"""Isolated fixed-input probe of pinned native degree-one SH directions.

Compares unchanged native output to two explicitly different evaluations:
the true triangle center and vec3(center.z), the C++ comma-expression result.
No optimizer or training; all RGB values stay above the lower color clamp.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tgs_native_diagnostic_backend import load_backend, OUT
from run_tgs_native_diagnostic import screen_triangle, call_native

C1 = np.sqrt(3/(4*np.pi))


def color(position):
    d = np.array(position,dtype=float,copy=True)
    d /= np.linalg.norm(d)
    return np.array([.5-.3*C1*d[1],.5+.3*C1*d[2],.5-.3*C1*d[0]])


def main(shfix=False):
    target=OUT/'sh_direction_v1'
    target.mkdir(exist_ok=True)
    mod=load_backend(shfix=shfix)
    view=np.eye(4,dtype=np.float32)
    projection=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]],dtype=np.float32)
    base=screen_triangle([[40.25,40.25],[200.25,40.25],[40.25,160.25]])
    sh=np.zeros((1,4,3),dtype=np.float32)
    sh[0,1,0]=.3;sh[0,2,1]=.3;sh[0,3,2]=.3
    def t(x,dtype=torch.float32):
        return torch.tensor(x,dtype=dtype,device='cuda').contiguous()
    records=[]
    for label,delta in [('original',[0,0,0]),('translate_x',[.5,0,0]),('translate_y',[0,.4,0])]:
        pts=(base+np.array(delta,dtype=np.float32))[None]
        white=call_native(mod,pts,np.array([1.]),np.array([.5]),view,projection)[0]
        with torch.no_grad():
            result=mod.rasterize_triangles(t([0,0,0]),t(pts),t([1]),t([3],torch.int32),
                t([0],torch.int32),t([]),t([.5]),t([0]),t([0]),t(view.T.copy()),
                t(projection.T.copy()),1,1.,1.,256,256,t(sh),1,t([0,0,0]),False,True)
            torch.cuda.synchronize()
        rgb=result[1].cpu().numpy()
        mask=white[0]>.1
        assert mask.sum()>100
        observed=np.median(rgb[:,mask]/white[0,mask],axis=1)
        center=pts[0].astype(float).mean(0)
        expected_true=color(center)
        expected_comma=color(np.repeat(center[2],3))
        err_true=float(np.max(np.abs(observed-expected_true)))
        err_comma=float(np.max(np.abs(observed-expected_comma)))
        if shfix:
            assert err_true<2e-6 and err_comma>.01
        else:
            assert err_comma<2e-6 and err_true>.01
        # Exercise only the degree-one SH coefficient derivative. This does
        # not validate geometry, opacity, sigma, or camera derivatives.
        with torch.no_grad():
            grads=mod.rasterize_triangles_backward(t([0,0,0]),t(pts),t([1]),
                t([3],torch.int32),t([0],torch.int32),result[3],t([]),
                t(view.T.copy()),t(projection.T.copy()),1,1.,1.,
                t(np.ones((3,256,256))),t(np.zeros((7,256,256))),t(sh),1,t([0,0,0]),
                result[4],result[0],result[5],result[6],True)
            torch.cuda.synchronize()
        dsh=grads[4].cpu().numpy()[0]
        direction=np.array(center if shfix else np.repeat(center[2],3),copy=True)
        direction/=np.linalg.norm(direction)
        basis=np.array([np.sqrt(1/(4*np.pi)),-C1*direction[1],C1*direction[2],-C1*direction[0]])
        expected_grad=basis[:,None]*white[0].astype(float).sum()*np.ones((1,3))
        grad_rel=float(np.max(np.abs(dsh-expected_grad))/np.max(np.abs(expected_grad)))
        assert np.isfinite(dsh).all() and grad_rel<2e-4
        row={'case':label,'center':center.tolist(),'observed_rgb':observed.tolist(),
             'true_center_rgb':expected_true.tolist(),'comma_expression_rgb':expected_comma.tolist(),
             'max_abs_error_true':err_true,'max_abs_error_comma':err_comma,
             'samples':int(mask.sum()),'matches_comma_expression':not shfix,
             'sh_gradient_relative_error':grad_rel,'dSH':dsh.tolist(),
             'analytic_dSH':expected_grad.tolist()}
        records.append(row)
        np.savez_compressed(target/(label+('_fixed' if shfix else '')+'.npz'),points=pts,sh=sh,view=view,
                            projection=projection,white=white,native_sh=rgb,ratio_mask=mask)
        print(json.dumps(row),flush=True)
    result={'confirmed':True,'scope':'fixed degree-one SH forward and coefficient gradients; no training impact measured',
            'source_commit':'6d61f4c0a571c635e37ffb740250706fa30a5541',
            'source_forward_line':410,'source_backward_line':293,
            'expression':'(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)',
            'semantics':'comma expression yields center_triangle.z; scalar vec3 constructor repeats z',
            'native_source_modified':bool(shfix),'original_archive_modified':False,
            'backward_exercised':'SH coefficients only','fixed_variant':bool(shfix),
            'native_module_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
            'cases':records}
    (target/('fixed_results.json' if shfix else 'results.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'confirmed':True,'cases':len(records)}),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--shfix',action='store_true')
    main(parser.parse_args().shfix)
