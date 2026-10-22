"""Replay every saved stress case/upstream with the direct-factor FP64 branch."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import research.audit_world_preprocess as mathematical
from research.factor_preprocess_probe import load_probe,source_payloads
from research.world_preprocess_probe import RASTER


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def rotation_without_normalization(q):
    r,x,y,z=q
    return torch.stack((1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),
        2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),
        2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y))).reshape(3,3)


def metric(a,b):
    norm=float(torch.linalg.vector_norm(a));err=float(torch.linalg.vector_norm(a-b))
    return {'reference_norm':norm,'absolute_l2':err,'scaled_error':err/max(norm,1e-5)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stress',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    old=json.loads((args.stress/'manifest.json').read_text());saved=json.loads((args.stress/'results.json').read_text())
    if old['status']!='complete' or sha(args.stress/'results.json')!=old['results_sha256']:raise ValueError('Completed unmodified stress result required.')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'research/factor_preprocess_probe.py',ROOT/'research/stable_inverse_backend.py',
        ROOT/'research/world_preprocess_probe.py',ROOT/'research/precise_preprocess_backend.py',ROOT/'research/audit_world_preprocess.py']
    sources+=[RASTER/'cuda_rasterizer'/n for n in ('render_forward.cu','render_backward.cu')]
    sources+=list((RASTER/'cuda_rasterizer').glob('*.h'))
    sources+=[x for x in (RASTER/'third_party/glm/glm').rglob('*') if x.is_file()]
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'building',
        'source_sha256':{x.relative_to(ROOT).as_posix():sha(x) for x in sources},
        'stress_results_sha256':sha(args.stress/'results.json'),'primary_gradient_gate':.001,'absolute_floor':1e-5,
        'selection':f'Exact same {len(saved["cases"])} saved cases and {len(saved["rows"])} upstream blocks; no exclusion or resampling.',
        'input_artifact_kind':old.get('artifact_kind','stress_audit'),
        'references':['Primary: independent physical float64 model with normalized raw quaternion.',
                      'Secondary: float64 graph at the exact device quaternion value, without renormalization; same external normalization pullback.'],
        'limits':['Secondary device-value reference localizes node-rounding sensitivity and does not replace the primary admission gate.',
                  'Fixed-output world-preprocessing VJPs, not full-renderer or quality evidence.']}
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for x in sources:z.write(x,x.relative_to(ROOT).as_posix())
        for n,v in source_payloads().items():z.writestr('generated/'+n,v)
    state['source_archive_sha256']=sha(out/'sources.zip')
    def write():(out/'manifest.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    write();start=time.perf_counter()
    try:
        mod=load_probe();state['binary_sha256']=sha(Path(mod.__file__));state['status']='running';write()
        fields=('mean','logscale','raw_quaternion','opacity');rows=[];diagnostics=[]
        for i,case in enumerate(saved['cases']):
            mg=torch.tensor([case['mean']],dtype=torch.float32,device='cuda')
            sg=torch.tensor([case['filtered_scales']],dtype=torch.float32,device='cuda')
            qg=torch.tensor([case['normalized_device_quaternion']],dtype=torch.float32,device='cuda')
            og=torch.tensor([case['opacity']],dtype=torch.float32,device='cuda')
            view=torch.tensor(case['view'],dtype=torch.float64);vg=view.float().cuda().contiguous();params=case['params']
            raw=torch.tensor(case['raw_quaternion'],dtype=torch.float64);unit=raw/torch.linalg.vector_norm(raw)
            leaves=(mg[0].cpu().double().requires_grad_(),sg[0].cpu().double().log().requires_grad_(),raw.clone().requires_grad_(),og[0].cpu().double().requires_grad_())
            primary,_=mathematical.oracle(*leaves,view,params)
            deviceleaves=tuple(x.detach().clone().requires_grad_() for x in (leaves[0],leaves[1],qg[0].cpu().double(),leaves[3]))
            with patch.object(mathematical,'rotation',rotation_without_normalization):
                secondary,_=mathematical.oracle(*deviceleaves,view,params)
            R=rotation_without_normalization(qg[0].cpu().double());s=sg[0].cpu().double()
            cov=R@torch.diag(s*s)@R.T;precision=R@torch.diag(1/(s*s))@R.T
            residual=precision@cov-torch.eye(3,dtype=torch.float64)
            diagnostics.append({'case':i,'model':case['model'],'id':case['id'],'axis_ratio':case['axis_ratio'],
                'quaternion_norm_error':case['device_quaternion_norm64_error'],
                'rotation_orthogonality_spectral_error':float(torch.linalg.matrix_norm(R.T@R-torch.eye(3,dtype=torch.float64),ord=2)),
                'precision_times_covariance_minus_identity_spectral_norm':float(torch.linalg.matrix_norm(residual,ord=2))})
            for row in [r for r in saved['rows'] if r['case']==i]:
                up=torch.tensor(row['upstream'],dtype=torch.float64)
                gp=torch.autograd.grad((primary*up).sum(),leaves,retain_graph=True)
                gd=torch.autograd.grad((secondary*up).sum(),deviceleaves,retain_graph=True)
                gd=list(gd);gd[2]=(gd[2]-unit*(gd[2]*unit).sum())/torch.linalg.vector_norm(raw)
                gc=up[:4].clone();gc[1]*=.5
                incoming=[x.float().cuda()[None].contiguous() for x in (gc,up[4:8],up[8:11])]
                compare={}
                for name,mode in (('roundtrip64',5),('factor64',6)):
                    gm,gs,gq,go=[x[0].cpu().double() for x in mod.backward(mode,mg,sg,qg,og,vg,*incoming,*params)]
                    gq=(gq-unit*(gq*unit).sum())/torch.linalg.vector_norm(raw)
                    values=(gm,gs*sg[0].cpu().double(),gq,go)
                    compare[name]={'physical':{f:metric(a,b) for f,a,b in zip(fields,gp,values)},
                        'device_value':{f:metric(a,b) for f,a,b in zip(fields,gd,values)}}
                rows.append({'case':i,'component':row['component'],'model':case['model'],'id':case['id'],
                    'comparison':compare,'primary_vs_device_reference':{f:metric(a,b) for f,a,b in zip(fields,gp,gd)}})
            if i%8==0:print(f'Replayed {i+1}/{len(saved["cases"])} stress cases',flush=True)
        components=list(saved['summary']['all64'])
        def summarize(v,ref,label):
            vals=[e for r in rows if r['component']==label for e in r['comparison'][v][ref].values()]
            return {'max_scaled_error':max(e['scaled_error'] for e in vals),'failed_groups':sum(e['scaled_error']>.001 for e in vals),'total_groups':len(vals)}
        summary={v:{ref:{label:summarize(v,ref,label) for label in components} for ref in ('physical','device_value')}
                 for v in ('roundtrip64','factor64')}
        result={'rows':rows,'matrix_diagnostics':diagnostics,'summary':summary,
            'max_precision_covariance_identity_residual':max(r['precision_times_covariance_minus_identity_spectral_norm'] for r in diagnostics),
            'factor64_primary_gate_pass':all(r['failed_groups']==0 for r in summary['factor64']['physical'].values()),
            'limits':state['limits']}
        (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');state['results_sha256']=sha(out/'results.json')
        state['status']='complete';state['factor64_primary_gate_pass']=result['factor64_primary_gate_pass']
        print(json.dumps({'factor64_primary_gate_pass':result['factor64_primary_gate_pass'],'summary':summary,
            'max_precision_covariance_identity_residual':result['max_precision_covariance_identity_residual']},indent=2),flush=True)
    except BaseException as e:state['status']='failed';state['error']=repr(e);raise
    finally:state['wall_seconds']=time.perf_counter()-start;write()


if __name__=='__main__':main()
