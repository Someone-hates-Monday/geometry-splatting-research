"""Causal upper-bound-only intervention on the unchanged frozen stress_40."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from tgs_mixed_reference import ROOT,Native,reference

OUT=ROOT/'experiments/tgs_radius_contraction_v1'


def main():
    target=OUT/'tile_coverage/stress40_upper_only.json';assert not target.exists()
    plan=json.loads((OUT/'input_plan.json').read_text());entry=next(e for e in plan['stress_cases'] if e['name']=='stress_40')
    path=ROOT/entry['input'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    with np.load(path) as z:f={k:z[k].copy() for k in z.files}
    for k in ['n','w','h','degree']:f[k]=int(f[k])
    receipt=json.loads((OUT/'tile_coverage/upper_fix_build/build_receipt.json').read_text())
    p=Path(receipt['module_file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['module_sha256']
    spec=importlib.util.spec_from_file_location(p.stem,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    f['pixel_grad']=f['patterns'][0];call=Native(mod,f);data=call.forward(f['theta'])
    rows=[];gradients=[];references=[]
    for pi,pattern in enumerate(f['patterns']):
        f['pixel_grad']=pattern;call.grad.zero_()
        for (x,y),g in zip(f['pixels'],pattern):call.grad[:,y,x]=call.t(g)
        x=torch.tensor(f['theta'].astype(float),requires_grad=True)
        loss,samples,meta=reference(x,f);exact=torch.autograd.grad(loss,x)[0].numpy()
        g,_=call.backward(data);gradients.append(g);references.append(exact)
        ge=float(np.linalg.norm(g[:9]-exact[:9])/np.linalg.norm(exact[:9]))
        se=float(np.linalg.norm(g[9:]-exact[9:])/np.linalg.norm(exact[9:]))
        fe=float(np.max(np.abs(data[-1]-samples.detach().numpy())))
        rows.append(dict(pattern=pi,geometry_relative=ge,scalar_relative=se,forward_max_abs=fe,
                         passes_same_numeric_gates=ge<=1e-3 and se<=1e-3 and fe<=3e-5))
    with np.load(OUT/'piecewise_stress_40.npz') as z:before=z['native_samples']
    np.savez_compressed(OUT/'tile_coverage/stress40_upper_only.npz',native_gradients=np.array(gradients),
                        reference_gradients=np.array(references),before_samples=before,after_samples=data[-1],
                        native_rgb=data[1][1].cpu().numpy(),native_aux=data[1][2].cpu().numpy())
    result=dict(input_sha256=entry['sha256'],module_sha256=receipt['module_sha256'],rows=rows,
                before_samples=before.tolist(),after_samples=data[-1].tolist(),
                max_geometry_relative=max(r['geometry_relative'] for r in rows),
                all_eight_same_numeric_gates=all(r['passes_same_numeric_gates'] for r in rows),
                scope='only four exclusive tile upper-bound expressions changed; contraction v1 outcome is not overwritten',training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
