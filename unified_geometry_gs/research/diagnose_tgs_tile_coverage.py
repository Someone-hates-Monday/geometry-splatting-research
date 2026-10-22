"""Exact rational rectangle counterexample plus independent native reproduction."""
import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_radius_contraction_v1/tile_coverage'


def exact_certificate():
    # Right 3-4-5 triangle with dyadic screen coordinates and rational cutoff.
    A=(F(317,8),F(124));B=(A[0]+6,A[1]);C=(A[0],A[1]+F(9,2))
    r=F(3,2);s=(A[0]+r,A[1]+r);opacity=F(1,2);cutoff=F(1,100)
    erosion=cutoff/opacity
    ymax=s[1]+(1-erosion)*(C[1]-s[1])
    legacy_upper=(ymax+15)//16;correct_upper=ymax//16+1
    pixel=(F(40),F(128));n=(F(3,5),F(4,5));offset=-(n[0]*B[0]+n[1]*B[1])
    phis=[A[1]-pixel[1],n[0]*pixel[0]+n[1]*pixel[1]+offset,A[0]-pixel[0]]
    phi=max(phis);alpha=opacity*(-phi/r)
    assert r==F(3,2) and ymax==F(3211,25)
    assert legacy_upper==8 and correct_upper==9
    assert pixel[1]//16==8 and alpha==F(7,120)>cutoff>F(1,255)
    result=dict(proven=True,vertices=[[str(x),str(y)] for x,y in [A,B,C]],inradius=str(r),
        inner_support_ymax=str(ymax),legacy_exclusive_upper_tile=int(legacy_upper),
        correct_exclusive_upper_tile=int(correct_upper),pixel=[40,128],pixel_tile_y=8,
        phi=[str(x) for x in phis],expected_alpha=str(alpha),cutoff=str(cutoff),
        claim='legacy integer-radius-style upper bound excludes a valid integer sample inside continuous eroded support',
        proof_scope='exact real/rational arithmetic; native reproduction is separately saved')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'rational_certificate.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


def native(upper_fixed=False):
    import torch
    from tgs_mixed_reference import Native,reference
    receipt=json.loads((ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/build_receipt.json').read_text())
    path=Path(receipt['module_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['module_sha256']
    if upper_fixed:
        from tgs_tile_upper_backend import load_upper_fix
        mod=load_upper_fix()
    else:
        spec=importlib.util.spec_from_file_location(path.stem,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    q=np.array([[39.625,124.],[45.625,124.],[39.625,128.5]])
    p=np.c_[((2*q+1)/256-1)*2,np.full(3,2.)].astype(np.float32)
    f=dict(name='rational_tile_counterexample',n=1,w=256,h=256,degree=-1,
           theta=np.r_[p.ravel(),1.,.5].astype(np.float32),pixels=np.array([[40,128],[40,127]]),
           pixel_grad=np.array([[1,0,0],[0,0,0]],dtype=np.float32),
           view=np.eye(4,dtype=np.float32),projection=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]],dtype=np.float32),
           colors=np.ones((1,3),dtype=np.float32),background=np.zeros(3,dtype=np.float32),campos=np.zeros(3,dtype=np.float32),tx=1.,ty=1.)
    call=Native(mod,f);data=call.forward(f['theta']);g,arrays=call.backward(data)
    x=torch.tensor(f['theta'].astype(float),requires_grad=True)
    loss,ideal,meta=reference(x,f)
    reference_gradient=torch.autograd.grad(loss,x)[0].numpy()
    alpha=data[1][2][1].cpu().numpy()
    assert int(data[1][3][0])>0
    if upper_fixed:assert data[-1][0].min()>.05 and alpha[128,40]>.05
    else:assert data[-1][0].max()==0 and alpha[128,40]==0
    assert data[-1][1].min()>.1
    result=dict(native_counterexample_confirmed=True,native_samples=data[-1].tolist(),
        ideal_samples=ideal.detach().numpy().tolist(),native_radius=int(data[1][3][0]),
        native_inradius=float(data[1][8][0]),native_gradient_max_abs=float(np.max(np.abs(g))),
        metadata=meta,module_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),training=False,
        upper_fixed=upper_fixed,gradient_relative=float(np.linalg.norm(g-reference_gradient)/np.linalg.norm(reference_gradient)),
        forward_max_abs=float(np.max(np.abs(data[-1]-ideal.detach().numpy()))))
    name='upper_fixed_reproduction' if upper_fixed else 'native_reproduction'
    np.savez_compressed(OUT/(name+'.npz'),**f,native_rgb=data[1][1].cpu().numpy(),
                        native_aux=data[1][2].cpu().numpy(),native_gradient=g,reference_gradient=reference_gradient)
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--native',action='store_true');parser.add_argument('--upper-fixed',action='store_true');args=parser.parse_args()
    if args.native:native(upper_fixed=args.upper_fixed)
    else:exact_certificate()
