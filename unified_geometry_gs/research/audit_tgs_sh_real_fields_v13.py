"""Independent SciPy basis, checkpoint rows, all saved projections and weighted field metrics."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import sph_harm_y
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_sh_shell_v13/real_fields_v2'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def basis(d,L):
    theta=np.arccos(d[:,2]);phi=np.arctan2(d[:,1],d[:,0]);columns=[]
    for l in range(L+1):
        for m in range(-l,l+1):
            v=sph_harm_y(l,abs(m),theta,phi)
            columns.append(v.real if m==0 else np.sqrt(2)*(v.imag if m<0 else v.real))
    return np.stack(columns,1)

def stats(a):
    a=np.asarray(a,dtype=float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),
                minimum=float(a.min()),maximum=float(a.max()))

def main():
    target=OUT/'independent_audit.json';assert not target.exists()
    plan=json.loads((OUT/'plan.json').read_text());result=json.loads((OUT/'results.json').read_text())
    assert result['all_defined_checks_pass'] and result['plan_sha256']==sha(OUT/'plan.json')
    assert result['script_sha256']==plan['script_sha256']==sha(ROOT/'probe_tgs_sh_real_fields_v13_v2.py')
    assert plan['original_script_sha256']==sha(ROOT/'probe_tgs_sh_real_fields_v13.py')
    z,w=np.polynomial.legendre.leggauss(37);phi=(np.arange(71)+.25)*2*np.pi/71
    Z,P=np.meshgrid(z,phi,indexing='ij');r=np.sqrt(1-Z*Z)
    d=np.stack([r*np.cos(P),r*np.sin(P),Z],-1).reshape(-1,3);weights=np.repeat(w/142,71)
    Y=basis(d,4);groups=[];verified=0;max_metric_error=0
    for model in plan['models']:
        scene=model['scene'];path=ROOT/model['path'];assert sha(path)==model['sha256']
        selected=json.loads((OUT/(scene+'_selection.json')).read_text());ids=np.asarray(selected['ids'])
        assert ids.tolist()==np.sort(np.random.default_rng(plan['seed']).choice(selected['count'],64,replace=False)).tolist()
        if scene=='flowers':assert ids.tolist()==json.loads((OUT.parent/'real_fields/flowers_selection.json').read_text())['ids']
        state=torch.load(path,map_location='cpu',weights_only=True)
        actual=torch.cat([state['features_dc'][ids],state['features_rest'][ids]],1).detach().numpy().astype(float)
        A=np.load(OUT/(scene+'_coefficients.npy'));assert A.shape==(64,16,3) and np.array_equal(actual,A)
        del state
        flat=A.transpose(1,0,2).reshape(16,-1);copy=(Y[:,:16]@flat+.5).reshape(-1,64,3)
        for axis in range(3):
            for rho in plan['rho']:
                e=d+rho*np.eye(3)[axis];e/=np.sqrt(np.einsum('ij,ij->i',e,e))[:,None]
                truth=(basis(e,3)@flat+.5).reshape(-1,64,3)
                projected=np.load(OUT/f'{scene}_{plan["axes"][axis]}_{rho}_projection.npz')
                assert np.max(np.abs(projected['fit32']-projected['fit48']))<=1e-8
                predictions={'copy':copy}
                for degree in [3,4]:
                    for fit in ['fit32','fit48']:
                        cc=projected[fit][: (degree+1)**2].reshape((degree+1)**2,-1)
                        predictions[f'L{degree}_{fit}']=(Y[:,:len(cc)]@cc+.5).reshape(-1,64,3)
                cases=[r for r in result['rows'] if r['scene']==scene and r['axis']==plan['axes'][axis] and r['relative_displacement']==rho]
                assert len(cases)==64 and [r['primitive'] for r in cases]==ids.tolist()
                for i,row in enumerate(cases):
                    for name,pred in predictions.items():
                        diff=pred[:,i]-truth[:,i]
                        clipped=np.maximum(pred[:,i],0)-np.maximum(truth[:,i],0)
                        vals=dict(raw_RMS=float(np.sqrt(np.einsum('i,ij,ij->',weights,diff,diff)/3)),
                            lower_clamped_RMS=float(np.sqrt(np.einsum('i,ij,ij->',weights,clipped,clipped)/3)),
                            grid_max_abs=float(np.abs(diff).max()))
                        for key,value in vals.items():
                            error=abs(value-row['metrics'][name][key]);max_metric_error=max(max_metric_error,error)
                            assert error<=2e-12+2e-12*abs(value)
                    b=sum(np.linalg.norm(A[i,l*l:(l+1)**2],axis=0)*np.sqrt(l*(l+1)*(2*l+1)/(8*np.pi)) for l in [1,2,3])*np.arcsin(rho)
                    assert np.max(np.abs(b-row['copy_color_bound_per_channel']))<=2e-12
                    assert np.all(np.max(np.abs(copy[:,i]-truth[:,i]),axis=0)<=b+1e-10+1e-10*b)
                    verified+=1
                names=list(predictions);positive=[r for r in cases if r['metrics']['copy']['raw_RMS']>1e-14]
                groups.append(dict(scene=scene,axis=plan['axes'][axis],rho=rho,cases=len(cases),
                    metrics={name:stats([r['metrics'][name]['raw_RMS'] for r in cases]) for name in names},
                    copy_near_zero_count=len(cases)-len(positive),
                    L3_over_copy=stats([r['metrics']['L3_fit32']['raw_RMS']/r['metrics']['copy']['raw_RMS'] for r in positive]) if positive else None,
                    L4_over_L3=stats([r['metrics']['L4_fit32']['raw_RMS']/r['metrics']['L3_fit32']['raw_RMS'] for r in cases if r['metrics']['L3_fit32']['raw_RMS']>1e-14]),
                    L3_worse_than_copy_count=sum(r['metrics']['L3_fit32']['raw_RMS']>r['metrics']['copy']['raw_RMS']+1e-12 for r in cases),
                    L4_worse_than_L3_count=sum(r['metrics']['L4_fit32']['raw_RMS']>r['metrics']['L3_fit32']['raw_RMS']+1e-12 for r in cases)))
        print(json.dumps(dict(scene=scene,independently_verified_cases=verified)),flush=True)
    assert verified==1536
    audit=dict(all_verified=True,cases=verified,groups=groups,maximum_metric_disagreement=max_metric_error,
        independent_basis='SciPy spherical harmonics, separate code path',checkpoint_rows_verified=True,
        results_sha256=sha(OUT/'results.json'),plan_sha256=sha(OUT/'plan.json'),script_sha256=sha(Path(__file__)),
        geometry_or_real_image_quality_verified=False,training=False)
    with target.open('x') as f:f.write(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(all_verified=True,cases=verified,maximum_metric_disagreement=max_metric_error)),flush=True)

if __name__=='__main__':main()
