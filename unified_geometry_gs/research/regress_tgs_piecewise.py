"""Re-evaluate nine frozen prior RGB VJPs with the piecewise native module.

No retraining or repeat finite-difference sweep; compares complete forward
images and gradients against already independently audited frozen references.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

from tgs_mixed_reference import ROOT,Native
from probe_tgs_mixed_backward import relative

OUT=ROOT/'experiments/tgs_branch_aux_v1'


def main():
    target=OUT/'prior_regression.json';assert not target.exists()
    receipt=json.loads((OUT/'piecewise_build/build_receipt.json').read_text())
    path=Path(receipt['module_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['module_sha256']
    spec=importlib.util.spec_from_file_location(path.stem,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    jobs=[]
    base=ROOT/'experiments/tgs_backward_v1/geometry_fixed'
    for row in json.loads((base/'results.json').read_text())['fixtures']:
        with np.load(base/(row['name']+'.npz')) as z:a={k:z[k].copy() for k in z.files}
        f=dict(a,n=1,degree=-1,colors=a['color'][None],background=np.zeros(3),campos=np.zeros(3),tx=1.,ty=1.,
               pixel_grad=(a['weights'][:,None]*a['channel_weights']).astype(np.float32))
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        jobs.append(('single_'+row['name'],f,np.array(row['exact_gradient']),a['native_rgb']))
    for relative_dir,variant in [('experiments/tgs_mixed_backward_v1','fixed'),('experiments/tgs_real_backward_v2','fixed')]:
        base=ROOT/relative_dir
        for row in json.loads((base/'results.json').read_text())['records']:
            if row['variant']!=variant:continue
            with np.load(base/(row['name']+'_input.npz')) as z:f={k:z[k].copy() for k in z.files}
            for k in ['n','w','h','degree']:f[k]=int(f[k])
            with np.load(base/(variant+'_'+row['name']+'.npz')) as z:image=z['native_rgb'].copy()
            jobs.append((row['name'],f,np.array(row['exact_gradient']),image))
    rows=[]
    for name,f,exact,image in jobs:
        native=Native(mod,f);data=native.forward(f['theta']);g,arrays=native.backward(data)
        n=f['n'];group=dict(geometry=relative(g[:9*n],exact[:9*n]),sigma_opacity=relative(g[9*n:11*n],exact[9*n:11*n]))
        if f['degree']>=0:group['SH']=relative(g[11*n:],exact[11*n:])
        difference=float(np.max(np.abs(data[1][1].cpu().numpy()-image)))
        row=dict(name=name,groups=group,forward_max_difference=difference,gate=all(e<1e-3 for e in group.values()) and difference==0,
                 native_gradient=g.tolist(),reference_gradient=exact.tolist())
        rows.append(row)
        np.savez_compressed(OUT/('regression_'+name+'.npz'),native_gradient=g,reference_gradient=exact,
                            native_rgb=data[1][1].cpu().numpy())
        print(json.dumps({k:row[k] for k in ['name','groups','forward_max_difference','gate']}),flush=True)
    target.write_text(json.dumps(dict(all_gates=all(r['gate'] for r in rows),rows=rows,training=False,
        module_sha256=receipt['module_sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
