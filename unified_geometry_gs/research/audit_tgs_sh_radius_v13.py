"""Audit all stress errors independently, and summarize undefined zero-denominator ratios."""
import importlib.util,json
from pathlib import Path
import numpy as np
from audit_tgs_sh_real_fields_v13 import ROOT,OUT,sha,stats
DEST=OUT.parent/'radius_stress'

def main():
    target=DEST/'independent_audit.json';assert not target.exists()
    result=json.loads((DEST/'results.json').read_text());plan=json.loads((DEST/'plan.json').read_text())
    assert result['plan_sha256']==sha(DEST/'plan.json')
    assert result['script_sha256']==plan['script_sha256']==sha(ROOT/'stress_tgs_sh_radius_v13.py')
    for name,h in plan['inputs'].items():assert sha(OUT/name)==h
    helper=ROOT/'sources/tgs_real_checkpoint_v1/source/utils/sh_utils.py'
    assert sha(helper)==json.loads((OUT/'plan.json').read_text())['helper_sha256']
    spec=importlib.util.spec_from_file_location('independent_stress_polynomials_v13',helper)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    def basis(d,L):return mod.eval_sh(L,np.eye((L+1)**2),d)
    z,w=np.polynomial.legendre.leggauss(37);p=(np.arange(71)+.25)*2*np.pi/71
    Z,P=np.meshgrid(z,p,indexing='ij');r=np.sqrt(1-Z*Z)
    d=np.stack([r*np.cos(P),r*np.sin(P),Z],-1).reshape(-1,3);weights=np.repeat(w/142,71)
    Y=basis(d,4);verified=0;maxerror=0.;groups=[]
    for scene in ['flowers','room']:
        A=np.load(OUT/(scene+'_coefficients.npy'));flat=A.transpose(1,0,2).reshape(16,-1)
        ids=json.loads((OUT/(scene+'_selection.json')).read_text())['ids']
        copy=(Y[:,:16]@flat+.5).reshape(-1,64,3)
        for axis,name in enumerate(['x','y','z']):
            for a in [.01,.05,.1,.25]:
                coef=np.load(OUT/f'{scene}_{name}_{a}_projection.npz')['fit32']
                fixed={f'L{L}':(Y[:,: (L+1)**2]@coef[: (L+1)**2].reshape((L+1)**2,-1)+.5).reshape(-1,64,3) for L in [3,4]}
                for k in [.5,2,4]:
                    e=d+(a/k)*np.eye(3)[axis];e/=np.sqrt((e*e).sum(axis=1))[:,None]
                    truth=(basis(e,3)@flat+.5).reshape(-1,64,3)
                    rows=[r for r in result['rows'] if r['scene']==scene and r['axis']==name and r['fit_a']==a and r['radius_factor']==k]
                    assert len(rows)==64 and [r['primitive'] for r in rows]==ids
                    for variant,pred in dict(copy=copy,**fixed).items():
                        values=np.sqrt(np.sum(weights[:,None,None]*(pred-truth)**2,axis=(0,2))/3)
                        saved=np.array([r['raw_RMS'][variant] for r in rows]);assert np.isfinite(saved).all()
                        error=float(np.max(np.abs(values-saved)));maxerror=max(error,maxerror);assert error<=2e-12
                    positive=[r for r in rows if r['raw_RMS']['copy']>1e-14]
                    groups.append(dict(scene=scene,axis=name,fit_a=a,radius_factor=k,cases=64,
                        copy_near_zero=len(rows)-len(positive),
                        raw_RMS={v:stats([r['raw_RMS'][v] for r in rows]) for v in ['copy','L3','L4']},
                        ratios={v:stats([r['raw_RMS'][v]/r['raw_RMS']['copy'] for r in positive]) if positive else None for v in ['L3','L4']},
                        worse_than_copy={v:sum(r['raw_RMS'][v]>r['raw_RMS']['copy']+1e-12 for r in rows) for v in ['L3','L4']}))
                    verified+=64
        print(json.dumps(dict(scene=scene,stress_cases_independently_verified=verified)),flush=True)
    assert verified==4608
    out=dict(all_raw_errors_verified=True,cases=verified,groups=groups,maximum_RMS_disagreement=maxerror,
        raw_original_summary_contains_undefined_ratios=True,
        ratio_policy='Copy RMS<=1e-14 has no ratio; retained in raw RMS/counts, excluded only from ratios.',
        original_outputs_preserved=True,results_sha256=sha(DEST/'results.json'),
        script_sha256=sha(Path(__file__)),training=False,real_image_metrics=False)
    with target.open('x') as f:f.write(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(all_raw_errors_verified=True,cases=verified,maximum_RMS_disagreement=maxerror)),flush=True)

if __name__=='__main__':main()
