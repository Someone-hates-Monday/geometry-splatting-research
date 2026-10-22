"""Count real backward-traversed pairs, with four one-pixel calibrations.

Instrumentation is never timed. Counts include traversed valid contributors
regardless of upstream magnitude; they are not counts of nonzero gradients.
"""
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path
import numpy as np
import torch

from benchmark_tgs_backward import ROOT,Workload
from tgs_mixed_reference import load_module

OUT=ROOT/'experiments/tgs_branch_exposure_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def raw_pixel(mod,kind):
    q=np.array([[-4.,-6.],[36.,-6.],[-4.,24.]])
    if kind=='tie':q=np.array([[-4.,-4.],[36.,-4.],[-4.,26.]])
    if kind in ['cap','early']:q=np.array([[-10.,-10.],[30.,-10.],[-10.,20.]])
    zs=np.array([2.,2.4,2.8,3.2,3.6]) if kind=='early' else np.array([2.])
    points=np.array([np.c_[q*2*z,np.full(3,z)] for z in zs],dtype=np.float32)
    n=len(points);sigma=np.full(n,.02 if kind in ['cap','early'] else 1.3,dtype=np.float32)
    opacity=np.full(n,.9999 if kind=='cap' else (.95 if kind=='early' else .65),dtype=np.float32)
    t=lambda x,dtype=torch.float32:torch.tensor(x,dtype=dtype,device='cuda').contiguous()
    bg=t([0,0,0]);p=t(points);sig=t(sigma);op=t(opacity);colors=t(np.full((n,3),.5))
    counts=t([3]*n,torch.int32);offset=t(np.arange(n)*3,torch.int32);view=t(np.eye(4));cam=t([0,0,0]);empty=t([])
    projection=t(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]],dtype=np.float32).T.copy())
    res=mod.rasterize_triangles(bg,p,sig,counts,offset,colors,op,t(np.zeros(n)),t(np.zeros(n)),view,projection,
                               n,1.,1.,1,1,empty,0,cam,False,False)
    grads=mod.rasterize_triangles_backward(bg,p,sig,counts,offset,res[3],colors,view,projection,n,1.,1.,
        t(np.ones((3,1,1))),t(np.zeros((7,1,1))),empty,0,cam,res[4],res[0],res[5],res[6],False)
    torch.cuda.synchronize()
    return res,grads,dict(points=points,sigma=sigma,opacity=opacity)


def main():
    assert not (OUT/'protocol.json').exists(),'Version new measurements explicitly; preserve this run'
    receipt=json.loads((OUT/'build_receipt.json').read_text());path=Path(receipt['module_file'])
    assert sha(path)==receipt['module_sha256']
    spec=importlib.util.spec_from_file_location(path.stem,path);trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
    control=load_module('control');work=Workload()
    benchmark=json.loads((ROOT/'experiments/tgs_backward_cost_v1/protocol.json').read_text())
    assert work.metadata==benchmark['workload']
    plan=dict(module_sha256=receipt['module_sha256'],workload=work.metadata,cameras=work.camera_metadata,
        counter_columns=receipt['counter_columns'],selftests=['untied','tie','cap','early'],
        scope='backward render-kernel valid contributor traversal, including zero upstream entries; per triangle exact unit counts',
        performance_claim=False,training=False,script_sha256=sha(Path(__file__)))
    (OUT/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    calibrations=[]
    for kind in plan['selftests']:
        a,ga,inputs=raw_pixel(control,kind);b,gb,_=raw_pixel(trace,kind)
        assert len(gb)==7 and torch.equal(a[1],b[1]) and torch.equal(a[2],b[2])
        c=gb[6].cpu().numpy();assert np.array_equal(c,np.rint(c))
        if kind=='untied':assert c.tolist()==[[1.,0.,0.,0.]]
        if kind=='tie':assert c.tolist()==[[1.,0.,1.,1.]]
        if kind=='cap':assert c[0,0]==1 and c[0,1]==1
        if kind=='early':assert c[:,0].tolist()==[1.,1.,1.,0.,0.] and c[:,1].sum()==0
        for x,y in zip(ga,gb[:6]):assert torch.allclose(x,y,rtol=1e-4,atol=1e-6)
        calibrations.append(dict(name=kind,counts=c.astype(int).tolist(),forward_equal=True))
        np.savez_compressed(OUT/('calibration_'+kind+'.npz'),**inputs,counts=c,control_rgb=a[1].cpu().numpy(),
                            trace_rgb=b[1].cpu().numpy())
        del a,b,ga,gb
    (OUT/'calibration.json').write_text(json.dumps(calibrations,indent=2)+'\n')
    print(json.dumps(dict(stage='counter_calibration_passed',cases=4)),flush=True)
    rows=[]
    for ci in range(2):
        base=work.forward(control,ci);g0=work.backward(control,base,ci)
        res=work.forward(trace,ci);g1=work.backward(trace,res,ci);torch.cuda.synchronize()
        assert torch.equal(base[1],res[1]) and torch.equal(base[2],res[2]) and torch.equal(base[3],res[3])
        counts=g1[6].cpu().numpy();assert counts.shape==(work.n,4)
        assert np.isfinite(counts).all() and np.array_equal(counts,np.rint(counts))
        assert counts.max()<=work.grad.shape[1]*work.grad.shape[2]<2**24
        counts=counts.astype(np.int32)
        assert (counts>=0).all() and (counts[:,1:]<=counts[:,0,None]).all() and (counts[:,3]<=counts[:,2]).all()
        current=base[9].cpu().numpy()>0
        assert np.array_equal(counts[:,0]>0,current)
        gradient_checks=[]
        for a,b in zip(g0,g1[:6]):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            norm=float(torch.linalg.vector_norm(a.double()))
            delta=float(torch.linalg.vector_norm((a.double()-b.double())))
            relative=delta/max(norm,1e-30)
            assert relative<3e-4
            gradient_checks.append(dict(reference_norm=norm,difference_norm=delta,relative=relative))
        baseline_path=ROOT/f'experiments/tgs_real_event_v1/baseline_{ci+2}.npz'
        with np.load(baseline_path) as z:cached=z['max_blend']>0
        totals=counts.sum(0,dtype=np.int64)
        row=dict(camera=ci,image=work.camera_metadata[ci]['img_name'],
                 totals={key:int(v) for key,v in zip(plan['counter_columns'],totals)},
                 triangles_with_each_event={key:int((counts[:,j]>0).sum()) for j,key in enumerate(plan['counter_columns'])},
                 forward_rgb_aux_radii_equal=True,gradient_checks=gradient_checks,
                 historical_cached_support_disagreements=int((current!=cached).sum()),
                 historical_baseline_sha256=sha(baseline_path))
        rows.append(row)
        np.savez_compressed(OUT/f'camera_{ci}_counts.npz',counts=counts,current_forward_contributing=current,
                            historical_cached_contributing=cached)
        (OUT/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        del base,res,g0,g1
    result=dict(protocol_sha256=sha(OUT/'protocol.json'),calibrations=calibrations,rows=rows,
                counter_module_sha256=receipt['module_sha256'],counter_scope=plan['scope'],training=False,
                quality_impact_established=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():(OUT/'execution_failure.txt').write_text(traceback.format_exc())
        raise
