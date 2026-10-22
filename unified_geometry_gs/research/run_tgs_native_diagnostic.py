"""Small fixed-input native forward tests. No optimizer, dataset, or training."""
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from tgs_native_diagnostic_backend import load_backend, ROOT, OUT
from tgs_native_reference import render, projected

W=H=256


def split(v):
    # Exact original main _update_params ordering, including winding differences.
    a,b,c=v
    ab,ac,bc=(a+b)/2,(a+c)/2,(b+c)/2
    return np.array([[a,ab,ac],[b,ab,bc],[c,ac,bc],[ab,ac,bc]],dtype=np.float32)


def screen_triangle(p):
    ndc=(2*np.asarray(p)+1)/[W,H]-1
    return np.c_[2*ndc,np.full(3,2)].astype(np.float32)


def call_native(mod,points,sigma,opacity,view,projection):
    n=len(points)
    def t(x,dtype=torch.float32):
        return torch.tensor(x,dtype=dtype,device='cuda').contiguous()
    with torch.no_grad():
        result=mod.rasterize_triangles(t([0,0,0]),t(points),t(sigma),
            t([3]*n,torch.int32),t(np.arange(n)*3,torch.int32),t(np.ones((n,3))),
            t(opacity),t(np.zeros(n)),t(np.zeros(n)),t(view.T.copy()),t(projection.T.copy()),
            n,1.,1.,H,W,t([]),0,t([0,0,0]),False,True)
        torch.cuda.synchronize()
    return result[1].cpu().numpy(),result[3].cpu().numpy(),result[8].cpu().numpy(),int(result[0])


def moments(image):
    yy,xx=np.mgrid[:H,:W]
    mass=float(image.astype(float).sum())
    return {'sum':mass,'centroid':[(image*xx).sum()/mass,(image*yy).sum()/mass] if mass else None}


def main():
    started=time.time()
    mod=load_backend()
    view=np.eye(4,dtype=np.float32)
    projection=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]],dtype=np.float32)
    base=screen_triangle([[40.25,40.25],[200.25,40.25],[40.25,160.25]])
    # Repeated identical launches establish nondeterministic variation before
    # choosing a reference tolerance. This is not an accuracy bound.
    repeats=[call_native(mod,base[None],np.array([1]),np.array([.5]),view,projection)[0] for _ in range(5)]
    repeat_max=max(float(np.abs(a-repeats[0]).max()) for a in repeats[1:])
    tolerance=max(3e-5,100*repeat_max)
    calibration={'repeats':5,'max_abs_repeat_difference':repeat_max,
                 'reference_abs_tolerance':tolerance,'tolerance_basis':'float32/fast-math allowance and repetition floor; not a proof bound'}
    (OUT/'cuda_calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    fixtures=[]
    for sig in [1.,1.16]:
        for label,pts in [('parent',base[None]),('children',split(base))]:
            fixtures.append((f'large_sigma{sig}_{label}',pts,sig,.5,view,projection))
    for size in [.75,1.5,2.5]:
        small=screen_triangle(np.array([[0,0],[4,0],[0,3]])*size+[112.25,112.25])
        for label,pts in [('parent',small[None]),('children',split(small))]:
            fixtures.append((f'inradius{size}_{label}',pts,1.,.5,view,projection))
    for op in [.009,.01,1.]:
        fixtures.append((f'opacity{op}',base[None],1.,op,view,projection))
    # High transmittance cutoff, away from an exact floating-point boundary.
    fixtures.append(('overlap_cutoff',np.repeat(base[None],5,axis=0),1.,.95,view,projection))
    # Same physical camera family as the certificate, with shared image scale
    # and principal point to keep the projected geometry resolvable in 256^2.
    d=100*np.sqrt(7)
    world=np.array([[0,0,0],[4,0,0],[0,4,0]],dtype=np.float32)
    c,q=.75,np.sqrt(7)/4
    for name,rot in [('A',[[1,0,0],[0,c,q],[0,-q,c]]),('B',[[c,0,q],[0,1,0],[-q,0,c]])]:
        vm=np.eye(4);vm[:3,:3]=rot;vm[2,3]=d
        pm=np.array([[2*35*d/W,0,(2*40.25+1)/W-1,0],
                     [0,2*35*d/H,(2*40.25+1)/H-1,0],[0,0,1,0],[0,0,1,0]])@vm
        for label,pts in [('parent',world[None]),('children',split(world))]:
            fixtures.append((f'finite_view{name}_{label}',pts,1.,.5,vm.astype(np.float32),pm.astype(np.float32)))
    rows=[]
    for name,points,sig,op,vm,pm in fixtures:
        n=len(points)
        sigmas=np.full(n,sig,dtype=np.float32);opacities=np.full(n,op,dtype=np.float32)
        native,radii,density,tiles=call_native(mod,points,sigmas,opacities,vm,pm)
        reference,ref_radii,details=render(points,sigmas,opacities,vm.astype(float),pm.astype(float),W,H)
        difference=np.abs(native[0]-reference)
        passed=bool(np.isfinite(native).all() and difference.max() <= tolerance and np.array_equal(radii,ref_radii))
        row={'name':name,'passed':passed,'max_abs_reference_difference':float(difference.max()),
             'mean_abs_reference_difference':float(difference.mean()),'native_radii':radii.tolist(),
             'reference_radii':ref_radii.tolist(),'native_inradius':density.tolist(),
             'tile_instances':tiles,'native_moments':moments(native[0]),
             'reference_moments':moments(reference),'reference_details':details}
        rows.append(row)
        np.savez_compressed(OUT/(name+'.npz'),points=points,sigma=sigmas,opacity=opacities,
                            view=vm,projection=pm,native=native,reference=reference)
        (OUT/'cuda_progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({k:row[k] for k in ['name','passed','max_abs_reference_difference','native_radii']}),flush=True)
    lookup={x['name']:x for x in rows}
    comparisons=[]
    for prefix in ['large_sigma1.0','large_sigma1.16','inradius0.75','inradius1.5','inradius2.5','finite_viewA','finite_viewB']:
        parent=np.load(OUT/(prefix+'_parent.npz'))['native'][0]
        child=np.load(OUT/(prefix+'_children.npz'))['native'][0]
        comparisons.append({'fixture':prefix,'parent':moments(parent),'children':moments(child),
                            'relative_L1':float(np.abs(parent-child).sum()/parent.sum()) if parent.sum() else None})
    report={'passed':all(x['passed'] for x in rows),'scope':'fixed-input forward renderer; no training/backward/quality evidence',
            'calibration':calibration,'fixtures':rows,'split_comparisons':comparisons,
            'seconds':time.time()-started,'native_module_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()}
    (OUT/'cuda_results.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['passed'], 'native/reference discrepancies: inspect persisted cases'
    print(json.dumps({'passed':True,'fixtures':len(rows),'seconds':report['seconds']}),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        (OUT/'cuda_failure.txt').write_text(traceback.format_exc())
        raise
