"""Independent float64 AD and direction-FD oracle for GGGS CUDA preprocessing.

No scene training or quality claim. Random synthetic and optional frozen-model
cases are recorded separately; masks and clamp domains are explicit.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.world_preprocess_probe import load_probe, source_payloads, VARIANTS, RASTER


def rotation(q):
    # Conventional world rotation; GLM's constructor stores these as columns,
    # hence its R is the transpose of this matrix.
    r,x,y,z = q / torch.linalg.vector_norm(q)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),
        2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),
        2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y))).reshape(3,3)


def oracle(mean, logscale, quat, opacity, view, params):
    fx,fy,tx,ty,k = params
    R = rotation(quat)
    A = view.T[:3,:3]; b = view.T[:3,3]
    t = A @ mean + b; tc = torch.linalg.vector_norm(t)
    u = (t[0]/t[2]).clamp(-1.3*tx,1.3*tx)
    v = (t[1]/t[2]).clamp(-1.3*ty,1.3*ty)
    zero = u*0; one = zero+1
    ray = torch.stack((u,v,one)); L = (ray*ray).sum()
    Q = A @ R
    covariance = Q @ torch.diag(torch.exp(2*logscale)) @ Q.T
    precision = Q @ torch.diag(torch.exp(-2*logscale)) @ Q.T
    J = torch.stack((fx/t[2],zero,-fx*u/t[2],zero,fy/t[2],-fy*v/t[2])).reshape(2,3)
    C0 = J @ covariance @ J.T; C = C0 + k*torch.eye(2,dtype=mean.dtype)
    # Explicit 2x2 determinant avoids a distinct generic LU computation path.
    det0 = (C0[0,0]*C0[1,1]-C0[0,1]**2).clamp_min(1e-6)
    det1 = (C[0,0]*C[1,1]-C[0,1]**2).clamp_min(1e-6)
    det = C[0,0]*C[1,1]-C[0,1]**2
    conic = torch.stack((C[1,1]/det,-C[0,1]/det,C[0,0]/det,opacity*torch.sqrt(det0/det1)))
    h = precision @ ray; vb = (ray*h).sum()
    K = torch.stack((1+v*v,-u*v,-u,-u*v,1+u*u,-v,zero,zero,zero)).reshape(3,3)
    plane = K @ h / vb
    factor = t[2]/torch.sqrt(L)
    rp = torch.stack((plane[0]*factor/fx,plane[1]*factor/fy,tc,torch.sqrt(vb/L)))
    # Independently simplified normal, not the production nJ/plane chain.
    normal = -h / torch.linalg.vector_norm(h)
    raw_length = torch.linalg.vector_norm(h)*torch.sqrt(L)/vb
    return torch.cat((conic,rp,normal)), raw_length


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=48);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'research/world_preprocess_probe.py',ROOT/'research/precise_preprocess_backend.py']
    sources += [RASTER/'cuda_rasterizer'/n for n in ('render_forward.cu','render_backward.cu')]
    sources += list((RASTER/'cuda_rasterizer').glob('*.h'))
    # Include all inline GLM dependencies, not only .hpp; .inl contains arithmetic.
    sources += [x for x in (RASTER/'third_party/glm/glm').rglob('*') if x.is_file()]
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'building',
        'source_sha256':{x.relative_to(ROOT).as_posix():sha(x) for x in sources},
        'variants':list(VARIANTS),'seed':79,'count':args.count,
        'scope':'Isolated one-Gaussian production preprocessing. Fixed upstreams; no rasterizer, loss or optimizer.',
        'gate':{'per_case_parameter_scaled_error':1e-3,'absolute_scale_floor':1e-5},
        'conic_offdiagonal_convention':'Supply half of scalar packed-conic y upstream, as in the actual renderer.'}
    payloads=source_payloads()
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for x in sources:z.write(x,x.relative_to(ROOT).as_posix())
        for n,value in payloads.items():z.writestr('generated/'+n,value)
    state['source_archive_sha256']=sha(out/'sources.zip')
    def write():(out/'manifest.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    write();started=time.perf_counter()
    try:
        module=load_probe();state['binary_sha256']=sha(Path(module.__file__))
        state['status']='running';write()
        torch.manual_seed(79);params=(240.,260.,.8,.6,.3)
        qcam=torch.randn(4,dtype=torch.float64);Acam=rotation(qcam)
        mat=torch.eye(4,dtype=torch.float64);mat[:3,:3]=Acam;mat[:3,3]=torch.tensor([.1,-.2,.3])
        view=mat.T.contiguous();viewgpu=view.float().cuda().contiguous()
        # Oracle inputs use the exact quantized device values, promoted to F64.
        view=viewgpu.cpu().double()
        fields=('mean','logscale','raw_quaternion','opacity')
        components={'normal':slice(8,11),'plane_slope':slice(4,6),'center_distance':slice(6,7),
                    'ray_precision':slice(7,8),'conic_diag':[0,2],'conic_offdiag':slice(1,2),'opacity_coef':slice(3,4)}
        rows=[]
        for i in range(args.count):
            is_clipped=i>=2*args.count//3
            z=1.2+torch.rand((),dtype=torch.float64)*2
            uv=torch.rand(2,dtype=torch.float64)*.9-.45
            if is_clipped:uv[0]=1.3*params[2]+.1+torch.rand((),dtype=torch.float64)
            t=torch.cat((uv*z,z[None]));m=Acam.T@(t-mat[:3,3])
            s=torch.exp(-2+torch.rand(3,dtype=torch.float64)*1.5)
            q=torch.randn(4,dtype=torch.float64);q=q/torch.linalg.vector_norm(q)
            op=torch.tensor(.65,dtype=torch.float64)
            mg=m.float().cuda()[None].contiguous();sg=s.float().cuda()[None].contiguous()
            qg=q.float().cuda()[None].contiguous();og=op.float().cuda()[None].contiguous()
            leaves=(mg[0].cpu().double().requires_grad_(),sg[0].cpu().double().log().requires_grad_(),
                    qg[0].cpu().double().requires_grad_(),og[0].cpu().double().requires_grad_())
            # Scale activation and raw quaternion normalization are differentiated
            # outside CUDA, as they are in the actual Python model.
            qg=(leaves[2].detach().float().cuda()/torch.linalg.vector_norm(leaves[2].detach().float().cuda()))[None].contiguous()
            # The F64 oracle normalizes the same raw q. Quantization of normalized
            # q is a documented small forward error, not a different rotation.
            output,raw_length=oracle(*leaves,view,params)
            actual=module.forward(mg,sg,qg,og,viewgpu,*params)[0].cpu().double()
            forward_errors={key:float((actual[sl]-output[sl]).detach().abs().max()) for key,sl in components.items()}
            if max(forward_errors.values())>2e-4:raise AssertionError(f'Forward parity failed case {i}: {forward_errors}')
            for label,sl in components.items():
                upstream=torch.zeros(11,dtype=torch.float64);upstream[sl]=torch.randn_like(upstream[sl])
                ref=torch.autograd.grad((output*upstream).sum(),leaves,retain_graph=True)
                gc=upstream[:4].clone();gc[1]*=.5
                incoming=[x.float().cuda()[None].contiguous() for x in (gc,upstream[4:8],upstream[8:11])]
                comparisons={}
                for j,name in enumerate(VARIANTS):
                    grads=module.backward(j,mg,sg,qg,og,viewgpu,*incoming,*params)
                    gm,gs,gq,go=[x[0].cpu().double() for x in grads]
                    rawq=leaves[2].detach();unit=rawq/torch.linalg.vector_norm(rawq)
                    gq=(gq-unit*(gq*unit).sum())/torch.linalg.vector_norm(rawq)
                    pulled=(gm,gs*sg[0].cpu().double(),gq,go)
                    errors={}
                    for f,a,b in zip(fields,ref,pulled):
                        norm=float(torch.linalg.vector_norm(a))
                        errors[f]={'reference_norm':norm,'absolute_l2':float(torch.linalg.vector_norm(b-a)),
                            'scaled_error':float(torch.linalg.vector_norm(b-a))/max(norm,1e-5)}
                    comparisons[name]=errors
                fd=[]
                directions=[torch.randn_like(x) for x in leaves]
                length=torch.sqrt(sum((d*d).sum() for d in directions));directions=[d/length for d in directions]
                directional=sum((a*d).sum() for a,d in zip(ref,directions))
                for step in (1e-4,3e-5,1e-5):
                    def pert(sign):
                        vals=[x.detach()+sign*step*d for x,d in zip(leaves,directions)]
                        return (oracle(*vals,view,params)[0]*upstream).sum()
                    value=(pert(1)-pert(-1))/(2*step)
                    fd.append({'step':step,'value':float(value),'autograd_value':float(directional),
                        'scaled_error':float(abs(value-directional))/max(float(abs(directional)),1e-5)})
                rows.append({'case':i,'clamp_domain':'outside_x' if is_clipped else 'inside',
                    'component':label,'raw_normal_length':float(raw_length.detach()),'forward_max_abs':forward_errors[label],
                    'gradient_comparisons':comparisons,'directional_fd':fd,
                    'inputs':{'mean':mg[0].cpu().tolist(),'scales':sg[0].cpu().tolist(),'raw_quaternion':rawq.tolist(),
                              'upstream':upstream.tolist()}})
            if i%8==0:print(f'Completed case {i+1}/{args.count}',flush=True)
        report={'rows':rows,'params':params,'view_transposed_storage':view.tolist(),
            'summary':{name:{label:{'max_scaled_error':max(e['scaled_error'] for r in rows if r['component']==label
                for e in r['gradient_comparisons'][name].values()),
                'failed_groups':sum(e['scaled_error']>1e-3 for r in rows if r['component']==label
                for e in r['gradient_comparisons'][name].values())} for label in components} for name in VARIANTS},
            'max_directional_fd_scaled_error':max(d['scaled_error'] for r in rows for d in r['directional_fd']),
            'limitations':['Synthetic positive scales and normalized rotations, two FoV clamp domains; not a real reconstruction quality test.',
                'FP64 branches retain float32 inputs and final stores. World covariance fallback/eigendecomposition branch is not tested.',
                'Conic denominator and opacity coefficient production backward include stabilization/approximations; components audited separately.',
                'Directional FD validates the independent F64 oracle, not noisy finite differences of float32 CUDA output.']}
        (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        state['results_sha256']=sha(out/'results.json');state['status']='complete'
        print(json.dumps(report['summary'],indent=2),flush=True)
    except BaseException as e:
        state['status']='failed';state['error']=repr(e);raise
    finally:
        state['wall_seconds']=time.perf_counter()-started;write()


if __name__=='__main__':main()
