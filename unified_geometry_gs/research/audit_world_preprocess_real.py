"""Independent local VJPs on prespecified frozen GGGS model parameters.

Three development scenes, camera 0013, sixteen projected in-FoV scale ratios
<=10 and eight >10 per scene. Projection membership is not visibility. These
are mechanism checks, not independent scene or reconstruction evaluations.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.audit_world_preprocess import oracle
from research.world_preprocess_probe import load_probe,source_payloads,VARIANTS,RASTER
from research.benchmark_median_cuda import load_config


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'building','seed':89,
        'camera':'0013','selection':'16 projected in-FoV filtered scale ratios <=10 and 8 >10, without using derivative errors',
        'forward_scaled_gate':1e-4,'vjp_scaled_gate':1e-3,'vjp_absolute_floor':1e-5,
        'limits':['One development camera in each of three previously used scenes; projected membership does not mean actual raster contribution.',
                  'Frozen filtered scales and opacity are independent leaves; Python 3D-filter activation chain is not part of this local audit.',
                  'Not an end-to-end renderer, training or quality result.']}
    sources=[Path(__file__),ROOT/'research/audit_world_preprocess.py',ROOT/'research/world_preprocess_probe.py',
        ROOT/'research/precise_preprocess_backend.py',ROOT/'research/benchmark_median_cuda.py',
        ROOT/'scene/gaussian_model.py',ROOT/'scene/cameras.py']
    sources+=[RASTER/'cuda_rasterizer'/n for n in ('render_forward.cu','render_backward.cu')]
    sources+=list((RASTER/'cuda_rasterizer').glob('*.h'))
    sources+=[x for x in (RASTER/'third_party/glm/glm').rglob('*') if x.is_file()]
    state['source_sha256']={x.relative_to(ROOT).as_posix():sha(x) for x in sources}
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for x in sources:z.write(x,x.relative_to(ROOT).as_posix())
        for n,value in source_payloads().items():z.writestr('generated/'+n,value)
    state['source_archive_sha256']=sha(out/'sources.zip')
    def write():(out/'manifest.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    write();start=time.perf_counter()
    try:
        mod=load_probe();state['binary_sha256']=sha(Path(mod.__file__));state['status']='running';write()
        from scene import GaussianModel,Scene
        torch.manual_seed(89)
        fields=('mean','logscale','raw_quaternion','opacity');rows=[];cases=[];forward_rows=[]
        components={'normal':slice(8,11),'plane_slope':slice(4,6),'ray_precision':slice(7,8),
                    'opacity_coef':slice(3,4),'conic':slice(0,3)}
        for cpu in args.cpu:
            ref=json.loads(cpu.read_text());model=Path(ref['model']);ply=model/'point_cloud/iteration_30000/point_cloud.ply'
            if [v for k,v in ref['source_sha256'].items() if k.endswith('point_cloud.ply')]!=[sha(ply)]:raise ValueError('Frozen model changed.')
            state.setdefault('input_sha256',{})[str(ply)]=sha(ply);state['input_sha256'][str(cpu.resolve())]=sha(cpu)
            ds,pipe=load_config(model)
            with tempfile.TemporaryDirectory(prefix='world-probe-',dir=out) as temp:
                scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch)
                gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
                cam=next(c for c in scene.getTrainCameras() if c.image_name==state['camera'])
                viewgpu=cam.world_view_transform.detach().contiguous();view=viewgpu.cpu().double()
                params=(cam.Fx,cam.Fy,float(torch.tan(torch.tensor(cam.FoVx/2))),float(torch.tan(torch.tensor(cam.FoVy/2))),ds.kernel_size)
                with torch.no_grad():
                    scale,opacity=gs.get_scaling_n_opacity_with_3D_filter
                    cam_points=gs.get_xyz@viewgpu[:3,:3]+viewgpu[3,:3]
                    uv=cam_points[:,:2]/cam_points[:,2:3]
                    eligible=(cam_points[:,2]>.2)&(uv[:,0].abs()<params[2])&(uv[:,1].abs()<params[3])
                    ratio=scale.max(1).values/scale.min(1).values
                    groups={'regular':torch.where(eligible&(ratio<=10))[0].cpu(),
                            'elongated':torch.where(eligible&(ratio>10))[0].cpu()}
                    selected=[]
                    for group,count in (('regular',16),('elongated',8)):
                        ids=groups[group]
                        if len(ids)<count:raise ValueError('Insufficient eligible points.')
                        selected.extend((group,int(x)) for x in ids[torch.randperm(len(ids))[:count]])
                for group,idx in selected:
                    mg=gs.get_xyz[idx:idx+1].detach().contiguous();sg=scale[idx:idx+1].detach().contiguous()
                    rawq=gs._rotation[idx].detach().cpu().double()
                    qg=gs.get_rotation[idx:idx+1].detach().contiguous();og=opacity[idx:idx+1].detach().flatten().contiguous()
                    leaves=(mg[0].cpu().double().requires_grad_(),sg[0].cpu().double().log().requires_grad_(),
                            rawq.requires_grad_(),og[0].cpu().double().requires_grad_())
                    expected,rawlength=oracle(*leaves,view,params)
                    actual=mod.forward(mg,sg,qg,og,viewgpu,*params)[0].cpu().double()
                    forward_errors={label:float(torch.linalg.vector_norm(actual[sl]-expected[sl].detach()))/max(float(torch.linalg.vector_norm(expected[sl].detach())),1e-5) for label,sl in components.items()}
                    forward_rows.append({'model':model.name,'id':idx,'group':group,'scaled_errors':forward_errors})
                    if max(forward_errors.values())>state['forward_scaled_gate']:raise AssertionError(f'Forward parity failed: {forward_rows[-1]}')
                    cases.append({'model':model.name,'camera':cam.image_name,'id':idx,'group':group,'ratio':float(ratio[idx]),
                        'mean':mg[0].cpu().tolist(),'scales':sg[0].cpu().tolist(),'raw_quaternion':rawq.detach().tolist(),
                        'normalized_device_quaternion':qg[0].cpu().tolist(),'opacity':float(og[0]),'params':params,'view':view.tolist()})
                    for label,sl in components.items():
                        up=torch.zeros(11,dtype=torch.float64);up[sl]=torch.randn_like(up[sl])
                        if label=='ray_precision':up[sl]/=max(float(expected[sl].detach().abs().max()),1.)
                        refgrads=torch.autograd.grad((expected*up).sum(),leaves,retain_graph=True)
                        conic=up[:4].clone();conic[1]*=.5
                        incoming=[x.float().cuda()[None].contiguous() for x in (conic,up[4:8],up[8:11])]
                        compare={}
                        for j,name in enumerate(VARIANTS):
                            gm,gscale,gq,go=[x[0].cpu().double() for x in mod.backward(j,mg,sg,qg,og,viewgpu,*incoming,*params)]
                            unit=rawq.detach()/torch.linalg.vector_norm(rawq.detach())
                            gq=(gq-unit*(gq*unit).sum())/torch.linalg.vector_norm(rawq.detach())
                            values=(gm,gscale*sg[0].cpu().double(),gq,go)
                            compare[name]={f:{'reference_norm':float(torch.linalg.vector_norm(a)),
                                'absolute_l2':float(torch.linalg.vector_norm(b-a)),
                                'scaled_error':float(torch.linalg.vector_norm(b-a))/max(float(torch.linalg.vector_norm(a)),state['vjp_absolute_floor'])} for f,a,b in zip(fields,refgrads,values)}
                        direction=[torch.randn_like(x) for x in leaves]
                        dn=torch.sqrt(sum((x*x).sum() for x in direction));direction=[x/dn for x in direction]
                        derivative=sum((a*d).sum() for a,d in zip(refgrads,direction));fd=[]
                        for step in (1e-4,3e-5,1e-5):
                            def pert(sign):return (oracle(*[x.detach()+sign*step*d for x,d in zip(leaves,direction)],view,params)[0]*up).sum()
                            value=(pert(1)-pert(-1))/(2*step)
                            fd.append({'step':step,'abs_error':float(abs(value-derivative)),
                                'scaled_error':float(abs(value-derivative))/max(float(abs(derivative)),1e-5)})
                        rows.append({'case':len(cases)-1,'model':model.name,'group':group,'id':idx,'component':label,
                            'raw_normal_length':float(rawlength.detach()),'upstream':up.tolist(),
                            'gradient_comparisons':compare,'directional_fd':fd})
                    print(f'{model.name} {group} {idx}: complete',flush=True)
            write()
        report={'cases':cases,'rows':rows,'forward':forward_rows,
            'summary':{name:{label:{'max_scaled_error':max(e['scaled_error'] for r in rows if r['component']==label for e in r['gradient_comparisons'][name].values()),
                'failed_groups':sum(e['scaled_error']>1e-3 for r in rows if r['component']==label for e in r['gradient_comparisons'][name].values())} for label in components} for name in VARIANTS},
            'max_forward_scaled_error':max(e for r in forward_rows for e in r['scaled_errors'].values()),
            'max_min_step_fd_scaled_error':max(r['directional_fd'][-1]['scaled_error'] for r in rows)}
        (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');state['results_sha256']=sha(out/'results.json')
        state['status']='complete';print(json.dumps(report['summary'],indent=2),flush=True)
    except BaseException as e:state['status']='failed';state['error']=repr(e);raise
    finally:state['wall_seconds']=time.perf_counter()-start;write()


if __name__=='__main__':main()
