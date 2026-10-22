"""Prespecified distance sensitivity of already-frozen coefficient conversions; CPU only."""
import json
from pathlib import Path
import numpy as np
from audit_tgs_sh_real_fields_v13 import ROOT,OUT,basis,sha,stats

DEST=OUT.parent/'radius_stress'

def main():
    DEST.mkdir(exist_ok=True);target=DEST/'results.json';assert not target.exists()
    provenance={n:sha(OUT/n) for n in ['plan.json','results.json','independent_audit.json']}
    assert json.loads((OUT/'independent_audit.json').read_text())['all_verified']
    with (DEST/'plan.json').open('x') as f:f.write(json.dumps(dict(inputs=provenance,
        script_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'TGS_SH_RADIUS_STRESS_PROTOCOL_V13.md'),
        radius_factors=[.5,2,4],posthoc_sensitivity=True,training=False),indent=2)+'\n')
    zz,w=np.polynomial.legendre.leggauss(37);phi=(np.arange(71)+.25)*2*np.pi/71
    z,p=np.meshgrid(zz,phi,indexing='ij');r=np.sqrt(1-z*z)
    d=np.stack([r*np.cos(p),r*np.sin(p),z],-1).reshape(-1,3);weights=np.repeat(w/142,71)
    Y=basis(d,4);rows=[];groups=[]
    for scene in ['flowers','room']:
        A=np.load(OUT/(scene+'_coefficients.npy'));flat=A.transpose(1,0,2).reshape(16,-1)
        ids=json.loads((OUT/(scene+'_selection.json')).read_text())['ids'];copy=(Y[:,:16]@flat+.5).reshape(-1,64,3)
        for axis,name in enumerate(['x','y','z']):
            for a in [.01,.05,.1,.25]:
                coef=np.load(OUT/f'{scene}_{name}_{a}_projection.npz')['fit32']
                fixed={f'L{L}':(Y[:,: (L+1)**2]@coef[: (L+1)**2].reshape((L+1)**2,-1)+.5).reshape(-1,64,3) for L in [3,4]}
                for k in [.5,2,4]:
                    e=d+a/k*np.eye(3)[axis];e/=np.linalg.norm(e,axis=1,keepdims=True)
                    truth=(basis(e,3)@flat+.5).reshape(-1,64,3)
                    metrics={}
                    for variant,pred in dict(copy=copy,**fixed).items():
                        diff=pred-truth;metrics[variant]=np.sqrt(np.einsum('i,inj,inj->n',weights,diff,diff)/3)
                    for i,primitive in enumerate(ids):rows.append(dict(scene=scene,axis=name,fit_a=a,
                        radius_factor=k,primitive=primitive,raw_RMS={v:float(x[i]) for v,x in metrics.items()}))
                    groups.append(dict(scene=scene,axis=name,fit_a=a,radius_factor=k,
                        raw_RMS={v:stats(x) for v,x in metrics.items()},
                        L3_worse_than_copy=int(np.sum(metrics['L3']>metrics['copy']+1e-12)),
                        L4_worse_than_copy=int(np.sum(metrics['L4']>metrics['copy']+1e-12)),
                        L3_over_copy=stats(metrics['L3']/metrics['copy']),
                        L4_over_copy=stats(metrics['L4']/metrics['copy'])))
        print(json.dumps(dict(scene=scene,cases=len(rows))),flush=True)
    assert len(rows)==4608
    with target.open('x') as f:f.write(json.dumps(dict(rows=rows,groups=groups,
        plan_sha256=sha(DEST/'plan.json'),script_sha256=sha(Path(__file__)),
        posthoc_sensitivity=True,real_captured_camera_or_image_metrics=False,training=False),indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),at_radius4={v:sum(g[v+'_worse_than_copy'] for g in groups if g['radius_factor']==4) for v in ['L3','L4']},training=False)),flush=True)

if __name__=='__main__':main()
