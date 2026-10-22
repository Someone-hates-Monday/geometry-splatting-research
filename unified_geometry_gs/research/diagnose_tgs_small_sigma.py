"""Exact input boundary certificate and isolated cached-geometry diagnosis."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import sympy as sp
from probe_tgs_stable_window import SafeNative,rectangles

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_stable_window_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def previous(start,size):return start-((size+127)//128)*128


def main():
    target=OUT/'small_sigma_boundary_certificate.json';assert not target.exists()
    plan=json.loads((OUT/'input_plan.json').read_text());e=next(e for e in plan['old_cases'] if e['name']=='alpha_saturated')
    with np.load(ROOT/e['input']) as z:f={k:z[k].copy() for k in z.files}
    for k in ['n','w','h','degree']:f[k]=int(f[k])
    f['pixels']=np.array([[176,44],[188,35]]);f['pixel_grad']=np.ones((2,3),np.float32)
    p=sp.Matrix([[sp.Rational(float(x)) for x in row]+[1] for row in f['theta'][:9].reshape(3,3)])
    P=sp.Matrix([[sp.Rational(float(x)) for x in row] for row in f['projection']])
    hom=p*P.T;eps=sp.Rational(1,10**7)
    q=[sp.Matrix([((hom[i,0]/(hom[i,3]+eps)+1)*f['w']-1)/2,((hom[i,1]/(hom[i,3]+eps)+1)*f['h']-1)/2]) for i in range(3)]
    edges=[q[(i+1)%3]-q[i] for i in range(3)];length=[sp.sqrt(e.dot(e)) for e in edges]
    D=sp.det(sp.Matrix.hstack(q[1]-q[0],q[2]-q[0]));radius=abs(D)/sum(length)
    sigma=sp.Rational(float(f['theta'][9]));opacity=sp.Rational(float(f['theta'][10]));points=[]
    for x,y in f['pixels']:
        cross=[sp.det(sp.Matrix.hstack(edges[i],sp.Matrix([int(x),int(y)])-q[i])) for i in range(3)]
        phis=[-sp.sign(D)*cross[i]/length[i] for i in range(3)]
        phi=max(phis,key=lambda z:float(z));u=-phi/radius
        raw=opacity*u**sigma
        assert all(c>0 for c in cross) and phi<0
        points.append(dict(pixel=[int(x),int(y)],exact_signed_cross=[str(c) for c in cross],inside_proven=True,
            active_phi_80_digits=str(sp.N(phi,80)),normalized_u_80_digits=str(sp.N(u,80)),alpha_80_digits=str(sp.N(raw,80))))
    receipt=json.loads((OUT/'direct/build_receipt.json').read_text());path=Path(receipt['module_file']);assert sha(path)==receipt['module_sha256']
    spec=importlib.util.spec_from_file_location(path.stem,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    data=SafeNative(mod,f).forward(f['theta']);res=data[1];mn,mx=rectangles(res,1,f['w'],f['h'])
    raw=res[4].cpu().numpy();total=3;mxoff=len(raw)-128-8;mnoff=previous(mxoff,8)
    pwoff=previous(mnoff,4*total);normaloff=previous(pwoff,8*total);offsetoff=previous(normaloff,4*total);indexoff=previous(offsetoff,4*total);pointoff=previous(indexoff,8*total)
    local=raw[pointoff:pointoff+24].view(np.float32).reshape(3,2).copy()
    normals=raw[normaloff:normaloff+24].view(np.float32).reshape(3,2).copy();offsets=raw[offsetoff:offsetoff+12].view(np.float32).copy()
    # Prefix field offsets are independently determined by obtain() order.
    anchor=raw[384:392].view(np.float32).copy();phi_center=raw[640:648].view(np.float32).copy()
    q64=np.array([[float(v) for v in qi] for qi in q]);quant=np.abs(local.astype(float)+anchor-q64)
    for row in points:
        x,y=row['pixel'];pi=np.array([x,y],np.float32)-anchor
        cached_phi=(normals*pi).sum(1,dtype=np.float32)+offsets
        row.update(native_alpha=float(res[2][1,y,x]),native_rgb=res[1][:,y,x].cpu().numpy().tolist(),tile_included=bool(((np.array([x,y])//16>=mn[0])&(np.array([x,y])//16<mx[0])).all()),cached_edge_phi_numpy_float32_surrogate=cached_phi.tolist())
    # Concavity gives |a^s-b^s|<=|a-b|^s for nonnegative a,b and 0<s<=1.
    # For delta>0, derivative of (t+delta)^s-t^s is <=0, so max is delta^s at t=0.
    t,d,s=sp.symbols('t d s',positive=True)
    derivative=sp.diff((t+d)**s-t**s,t)
    assert sp.simplify(derivative-s*((t+d)**(s-1)-t**(s-1)))==0
    required_log10=sp.N(sp.log(sp.Rational(3,10**5)/opacity,10)/sigma,40)
    result=dict(script_sha256=sha(Path(__file__)),input_sha256=e['sha256'],module_sha256=receipt['module_sha256'],
        sigma=float(sigma),opacity=float(opacity),points=points,cached_anchor=anchor.tolist(),cached_local_vertices=local.tolist(),cached_normals=normals.tolist(),cached_offsets=offsets.tolist(),cached_phi_center=phi_center.tolist(),max_projected_quantization=float(quant.max()),
        holder_exponent_proof=dict(exponent_range='0 < sigma <= 1',bound='abs(min(.99,o*a**sigma)-min(.99,o*b**sigma)) <= o*abs(a-b)**sigma',assumptions='a,b nonnegative; identical geometry inclusion/threshold/compositing decisions, before discontinuous contribution cutoff; o positive',derivative=str(derivative),lipschitz_failure='d(u**sigma)/du = sigma*u**(sigma-1) diverges as u -> 0+ when sigma<1',log10_sufficient_delta_u_for_3e5_error=str(required_log10)),
        interpretation='A retained old small-sigma boundary is strictly inside by exact rational predicates. Local float32 vertex quantization still loses this margin; one tested pixel has native alpha zero while its binning tile is present. This disproves unconditional whole-image accuracy of v2. NumPy cached-phi surrogate is not bitwise CUDA FMA replay.',training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'small_sigma_native_cache.npz',geometry=raw,native_rgb=res[1].cpu().numpy(),native_aux=res[2].cpu().numpy(),radii=res[3].cpu().numpy(),local_vertices=local,normals=normals,offsets=offsets,anchor=anchor)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
