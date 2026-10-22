"""Prespecified extreme-axis local VJP stress, using saved PLY parameters.

No images or full GPU model are loaded. The same already-built tiny CUDA
probe is used while the quality pilot runs; this is not a timing experiment.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from plyfile import PlyData
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.audit_world_preprocess import oracle
from research.world_preprocess_probe import load_probe,source_payloads,VARIANTS,RASTER


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finite_float(x):
    y=float(x)
    return y if np.isfinite(y) else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--real-audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    refstate=json.loads((args.real_audit/'manifest.json').read_text());ref=json.loads((args.real_audit/'results.json').read_text())
    if refstate['status']!='complete' or sha(args.real_audit/'results.json')!=refstate['results_sha256']:raise ValueError('Completed frozen reference required.')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'research/audit_world_preprocess.py',ROOT/'research/world_preprocess_probe.py',ROOT/'research/precise_preprocess_backend.py']
    sources+=[RASTER/'cuda_rasterizer'/n for n in ('render_forward.cu','render_backward.cu')]
    sources+=list((RASTER/'cuda_rasterizer').glob('*.h'))
    sources+=[x for x in (RASTER/'third_party/glm/glm').rglob('*') if x.is_file()]
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running','seed':101,
        'selection':'Eight largest filtered axis ratios among front-facing opacity>=1/255 primitives, plus eight largest within camera 0013 FoV, union without replacement per scene.',
        'gradient_scaled_gate':.001,'gradient_absolute_floor':1e-5,'forward_scaled_gate':1e-4,
        'source_sha256':{x.relative_to(ROOT).as_posix():sha(x) for x in sources},
        'reference_results_sha256':sha(args.real_audit/'results.json'),
        'limits':['Saved PLY parameters activated with torch CPU float32, then passed as device leaves; GPU activation rounding is not asserted identical.',
            'Extreme selection is a stress set, not an estimate of population failure prevalence or reconstruction contribution.',
            'Small probe runs concurrently with the full quality pilot; no latency or speed conclusions are drawn.',
            'Projection/front/opacity membership is not actual visibility. All selected cases, including forward failures, are retained.',
            'Oracle normalizes raw quaternions in float64; device normalization has float32 rounding. Large discrepancies require further localization.']}
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for x in sources:z.write(x,x.relative_to(ROOT).as_posix())
        for n,v in source_payloads().items():z.writestr('generated/'+n,v)
    state['source_archive_sha256']=sha(out/'sources.zip')
    def write():(out/'manifest.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    write();start=time.perf_counter()
    try:
        mod=load_probe();state['binary_sha256']=sha(Path(mod.__file__))
        torch.manual_seed(101);torch.set_num_threads(1)
        fields=('mean','logscale','raw_quaternion','opacity')
        components={'normal':slice(8,11),'plane_slope':slice(4,6),'ray_precision':slice(7,8),
                    'conic':slice(0,3),'opacity_coef':slice(3,4)}
        selected_variants=('legacy32','legacy64','all32','all64');cases=[];rows=[];population=[]
        plys=[Path(k) for k in refstate['input_sha256'] if k.endswith('point_cloud.ply')]
        for ply in plys:
            if sha(ply)!=refstate['input_sha256'][str(ply)]:raise ValueError('Frozen PLY changed.')
            state.setdefault('input_sha256',{})[str(ply)]=sha(ply);model=ply.parents[2].name
            camera=next(x for x in ref['cases'] if x['model']==model)
            view=torch.tensor(camera['view'],dtype=torch.float64);params=camera['params'];vg=view.float().cuda().contiguous()
            data=PlyData.read(ply)['vertex'].data
            means=torch.tensor(np.column_stack([data[n] for n in ('x','y','z')]),dtype=torch.float32)
            ls=torch.tensor(np.column_stack([data[f'scale_{i}'] for i in range(3)]),dtype=torch.float32)
            filt=torch.tensor(np.asarray(data['filter_3D']).copy(),dtype=torch.float32)[:,None]
            rawquats=torch.tensor(np.column_stack([data[f'rot_{i}'] for i in range(4)]),dtype=torch.float32)
            rawopacity=torch.tensor(np.asarray(data['opacity']).copy(),dtype=torch.float32)
            sq=ls.exp().square();smoothed=sq+filt.square();scale=smoothed.sqrt()
            opacity=rawopacity.sigmoid()*sq.prod(1).sqrt()*smoothed.prod(1).rsqrt()
            ratio=scale.max(1).values/scale.min(1).values
            cam=means@view.float()[:3,:3]+view.float()[3,:3]
            uv=cam[:,:2]/cam[:,2:3]
            front=(cam[:,2]>.2)&(opacity>=1/255)&torch.isfinite(ratio)
            inside=front&(uv[:,0].abs()<params[2])&(uv[:,1].abs()<params[3])
            choices={}
            for group,mask in (('front',front),('in_fov',inside)):
                ids=torch.where(mask)[0];order=ids[torch.argsort(ratio[ids],descending=True,stable=True)[:8]]
                for idx in order.tolist():choices.setdefault(idx,[]).append(group)
            population.append({'model':model,'total_primitives':len(means),'front_eligible':int(front.sum()),
                'in_fov_eligible':int(inside.sum()),'global_max_ratio':float(ratio.max()),
                'front_max_ratio':float(ratio[front].max()),'in_fov_max_ratio':float(ratio[inside].max()),'selected_count':len(choices)})
            for idx,groups in choices.items():
                mg=means[idx:idx+1].cuda().contiguous();sg=scale[idx:idx+1].cuda().contiguous()
                rq=rawquats[idx:idx+1].cuda().contiguous();qg=torch.nn.functional.normalize(rq,dim=1).contiguous()
                og=opacity[idx:idx+1].cuda().contiguous()
                leaves=(mg[0].cpu().double().requires_grad_(),sg[0].cpu().double().log().requires_grad_(),
                        rawquats[idx].double().requires_grad_(),og[0].cpu().double().requires_grad_())
                expected,rawlen=oracle(*leaves,view,params)
                actual=mod.forward(mg,sg,qg,og,vg,*params)[0].cpu().double()
                forward={k:{'finite':bool(torch.isfinite(actual[sl]).all() and torch.isfinite(expected[sl]).all()),
                    'scaled_error':finite_float(torch.linalg.vector_norm(actual[sl]-expected[sl].detach())/max(float(torch.linalg.vector_norm(expected[sl].detach())),1e-5))} for k,sl in components.items()}
                cases.append({'model':model,'id':idx,'selection_groups':groups,'axis_ratio':float(ratio[idx]),
                    'inside_fov':bool(inside[idx]),'mean':mg[0].cpu().tolist(),'filtered_scales':sg[0].cpu().tolist(),
                    'raw_quaternion':leaves[2].detach().tolist(),'normalized_device_quaternion':qg[0].cpu().tolist(),
                    'device_quaternion_norm64_error':float(abs(torch.linalg.vector_norm(qg[0].cpu().double())-1)),
                    'opacity':float(og[0]),'params':params,'view':view.tolist(),'forward':forward})
                for label,sl in components.items():
                    up=torch.zeros(11,dtype=torch.float64);up[sl]=torch.randn_like(up[sl])
                    if label=='ray_precision':up[sl]/=max(float(expected[sl].detach().abs().max()),1.)
                    reference=torch.autograd.grad((expected*up).sum(),leaves,retain_graph=True)
                    gc=up[:4].clone();gc[1]*=.5
                    incoming=[x.float().cuda()[None].contiguous() for x in (gc,up[4:8],up[8:11])];comparison={}
                    for name in selected_variants:
                        grads=mod.backward(VARIANTS.index(name),mg,sg,qg,og,vg,*incoming,*params)
                        gm,gscale,gq,go=[x[0].cpu().double() for x in grads]
                        raw=leaves[2].detach();unit=raw/torch.linalg.vector_norm(raw)
                        gq=(gq-unit*(gq*unit).sum())/torch.linalg.vector_norm(raw)
                        value=(gm,gscale*sg[0].cpu().double(),gq,go)
                        comparison[name]={f:{'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                            'reference_norm':finite_float(torch.linalg.vector_norm(a)),
                            'absolute_l2':finite_float(torch.linalg.vector_norm(a-b)),
                            'scaled_error':finite_float(torch.linalg.vector_norm(a-b)/max(float(torch.linalg.vector_norm(a)),1e-5))}
                            for f,a,b in zip(fields,reference,value)}
                    directions=[torch.randn_like(x) for x in leaves];dn=torch.sqrt(sum((x*x).sum() for x in directions));directions=[x/dn for x in directions]
                    derivative=sum((a*d).sum() for a,d in zip(reference,directions));fd=[]
                    for step in (1e-4,3e-5,1e-5):
                        def pert(sign):return (oracle(*[x.detach()+sign*step*d for x,d in zip(leaves,directions)],view,params)[0]*up).sum()
                        diff=(pert(1)-pert(-1))/(2*step)
                        fd.append({'step':step,'absolute_error':finite_float(abs(diff-derivative)),
                            'scaled_error':finite_float(abs(diff-derivative)/max(float(abs(derivative)),1e-5))})
                    rows.append({'case':len(cases)-1,'model':model,'id':idx,'component':label,'upstream':up.tolist(),
                        'raw_normal_length':finite_float(rawlen.detach()),'gradient_comparisons':comparison,'directional_fd':fd})
                print(f'{model}/{idx}: ratio={float(ratio[idx]):.3f}, complete',flush=True)
            write()
        def summarize(name,label):
            values=[e for r in rows if r['component']==label for e in r['gradient_comparisons'][name].values()]
            good=[e['scaled_error'] for e in values if e['scaled_error'] is not None]
            return {'max_finite_scaled_error':max(good) if good else None,'failed_groups':sum(not e['finite'] or e['scaled_error'] is None or e['scaled_error']>.001 for e in values),
                    'nonfinite_groups':sum(not e['finite'] for e in values),'total_groups':len(values)}
        summary={v:{k:summarize(v,k) for k in components} for v in selected_variants}
        report={'population_selection':population,'cases':cases,'rows':rows,'summary':summary,
            'forward_failed_components':sum(not e['finite'] or e['scaled_error'] is None or e['scaled_error']>1e-4 for c in cases for e in c['forward'].values()),
            'max_selected_axis_ratio':max(c['axis_ratio'] for c in cases),'limits':state['limits']}
        (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        state['results_sha256']=sha(out/'results.json');state['status']='complete'
        state['all64_stress_gate_pass']=all(r['failed_groups']==0 for r in summary['all64'].values())
        print(json.dumps({'case_count':len(cases),'all64_stress_gate_pass':state['all64_stress_gate_pass'],
            'summary_all64':summary['all64'],'forward_failed_components':report['forward_failed_components']},indent=2),flush=True)
    except BaseException as e:state['status']='failed';state['error']=repr(e);raise
    finally:state['wall_seconds']=time.perf_counter()-start;write()


if __name__=='__main__':main()
