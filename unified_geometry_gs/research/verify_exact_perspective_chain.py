"""Small independent exact-ray VJP and regular-root checks, not a renderer test."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor(x,grad=False):return torch.tensor(x,dtype=torch.float64,requires_grad=grad)

def forward(F,mu,q,A):
    d=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    u=-torch.einsum('nij,nj->ni',F,mu);v=torch.einsum('nij,nj->ni',F,d)
    Q=(v*v).sum(-1);p=-(u*v).sum(-1)/Q;rho=torch.sqrt(Q)
    w=u+p[:,None]*v;energy=(w*w).sum(-1)
    raw_alpha=A*torch.exp(-energy/2);alpha=torch.minimum(raw_alpha,torch.full_like(raw_alpha,.99))
    bar=torch.cat((torch.clamp(mu[:,:2]/mu[:,2,None],-.65,.65),torch.ones_like(mu[:,2,None])),dim=1)
    P=F.transpose(1,2)@F;t=torch.einsum('nij,nj->ni',P,bar)
    normal=-t/torch.linalg.vector_norm(t,dim=-1,keepdim=True)
    return p,rho,alpha,normal,(d,u,v,Q,w,raw_alpha,bar,P,t)

def analytic(F,mu,q,A,seed):
    p,rho,alpha,n,aux=forward(F,mu,q,A);d,u,v,Q,w,raw_alpha,bar,P,t=aux
    gp,gr,ga,gn=seed
    active=(raw_alpha<.99).to(F.dtype);ga=ga*active
    gE=-.5*ga*raw_alpha;gA=ga*torch.exp(-(w*w).sum(-1)/2)
    gu=-gp[:,None]*v/Q[:,None]+2*gE[:,None]*w
    gv=gp[:,None]*(-u/Q[:,None]-2*p[:,None]*v/Q[:,None])+gr[:,None]*v/rho[:,None]+2*gE[:,None]*p[:,None]*w
    gF=gu[:,:,None]*(-mu[:,None,:])+gv[:,:,None]*d[:,None,:]
    gmu=-torch.einsum('nji,nj->ni',F,gu)
    gd=torch.einsum('nji,nj->ni',F,gv)
    gq=(gd-(gd*d).sum(-1,keepdim=True)*d)/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    gt=-(gn-(gn*n).sum(-1,keepdim=True)*n)/torch.linalg.vector_norm(t,dim=-1,keepdim=True)
    GP=gt[:,:,None]*bar[:,None,:]
    gF=gF+F@(GP+GP.transpose(1,2))
    gbar=torch.einsum('nij,nj->ni',P,gt)
    h=gbar[:,:2]*(abs(mu[:,:2]/mu[:,2,None])<.65).to(F.dtype)
    gmu=gmu+torch.cat((h/mu[:,2,None],-(h*mu[:,:2]).sum(-1,keepdim=True)/(mu[:,2,None]**2)),dim=1)
    return gF,gmu,gq,gA

def H(t,alpha,p,rho):
    delta=(t-p)*rho;G=alpha*torch.exp(-delta*delta/2)
    near=-torch.log1p(-G)/2
    return torch.where(t<=p,near,-torch.log1p(-alpha)-near).sum()

def root(data,camera_origin):
    a,p,rho=[tensor(x) for x in data]
    def f(t):return float(H(t,a,p,rho)-(H(0.,a,p,rho) if camera_origin else 0.)-np.log(2.))
    return brentq(f,0. if camera_origin else -20.,20.,xtol=1e-14,rtol=1e-14)

def roots_check():
    cases=[]
    for camera_origin in (False,True):
        for i,(a,p,rho) in enumerate([
            ([.8,.65,.5],[.05,.5,1.2],[.9,2.,1.5]),
            ([.75,.5,.55],[1.,1.5,2.],[2.1,1.4,1.]),
            ([.9,.6,.7],[.3,.8,1.6],[1.7,2.2,.8])]):
            data=np.array([a,p,rho],float);t=root(data,camera_origin)
            variables=[tensor(x,True) for x in data];tt=tensor(t,True)
            residual=H(tt,*variables)-(H(0.,*variables) if camera_origin else 0.)-np.log(2.)
            grads=torch.autograd.grad(residual,[tt,*variables]);slope=float(grads[0]);assert slope>1e-6
            analytical=-np.stack([g.detach().numpy() for g in grads[1:]])/slope
            fd=np.empty_like(data)
            for component in np.ndindex(data.shape):
                h=1e-5;plus=data.copy();minus=data.copy();plus[component]+=h;minus[component]-=h
                fd[component]=(root(plus,camera_origin)-root(minus,camera_origin))/(2*h)
            errors=abs(analytical-fd)/np.maximum(1e-8,abs(analytical))
            cases.append({'case':i,'camera_origin_conditioned':camera_origin,'root':t,'H_slope':slope,
                          'max_scaled_error':float(errors.max()),'pass':bool(errors.max()<1e-6)})
    return cases

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    torch.set_num_threads(1);rng=np.random.default_rng(13018);N=96
    rot=np.linalg.qr(rng.normal(size=(N,3,3)))[0]
    axes=10**rng.uniform(-.6,.3,(N,3));f=np.transpose(rot,(0,2,1))/axes[:,:,None]
    mu=np.c_[rng.uniform(-1.5,1.5,(N,2)),rng.uniform(.8,2.,N)]
    q=np.c_[rng.uniform(-.8,.8,(N,2)),np.ones(N)]
    amp=rng.uniform(.2,.95,N)
    # Legal high opacity, zero perpendicular distance -> active cap, away from kink.
    mu[:24]=q[:24]*rng.uniform(.5,2.,(24,1));amp[:24]=.9995
    F,means,rays,A=[tensor(x,True) for x in (f,mu,q,amp)]
    seed=[tensor(rng.normal(size=N)),tensor(rng.normal(size=N)),tensor(rng.normal(size=N)),tensor(rng.normal(size=(N,3)))]
    outputs=forward(F,means,rays,A)
    loss=sum((out*g).sum() for out,g in zip(outputs[:4],seed))
    reference=torch.autograd.grad(loss,[F,means,rays,A]);candidate=analytic(F,means,rays,A,seed)
    errors=[(abs(x-y)/torch.maximum(abs(x),torch.tensor(1e-8))).detach().numpy() for x,y in zip(reference,candidate)]
    rootrows=roots_check()
    # Keep evidence scoped to current source files actually audited.
    sources=[Path(__file__),ROOT/'research/EXACT_PERSPECTIVE_BACKWARD_AUDIT.md',ROOT/'research/clipped_packet.py',ROOT/'research/clipped_packet_torch.py']
    sources += [ROOT/'submodules/diff-gaussian-rasterization/cuda_rasterizer'/name for name in
                ('render_forward.cu','render_backward.cu','sample_forward.cu','sample_backward.cu','rasterizer_impl.cu','auxiliary.h','config.h')]
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','seed':13018,'torch':torch.__version__,
      'numpy':np.__version__,'inputs':N,'cap_active':int((outputs[4][5]>.99).sum()),
      'factor_normal_vjp_max_scaled_errors':{k:float(e.max()) for k,e in zip(('F','mean','query_ray','amplitude'),errors)},
      'factor_normal_vjp_components':sum(e.size for e in errors),
      'factor_normal_vjp_gate_pass':all(e.max()<1e-8 for e in errors),'root_cases':rootrows,
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
      'limits':['CPU float64 analytic local VJP versus shared Torch autograd, not independent CUDA/full renderer verification.',
        'Root finite differences cover six noncritical small fixed-parameter examples and frozen support only.',
        'No production packet preprocessing, shared-primitive atomic accumulation, NCC image warp or full-scene quality/timing test.',
        'Cap and clip branch boundary derivatives were not tested; all constructed cases are away from boundaries.']}
    result['verification_pass']=result['factor_normal_vjp_gate_pass'] and all(r['pass'] for r in rootrows)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','inputs','cap_active','factor_normal_vjp_components','factor_normal_vjp_max_scaled_errors','verification_pass')},indent=2))
    if not result['verification_pass']:raise ArithmeticError('Preserved verification failure')

if __name__=='__main__':main()
