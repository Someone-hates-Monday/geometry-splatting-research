"""Independent MP240 scalar/world audit of saved short/strict hybrid results."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import hashlib, json, math, zipfile
import numpy as np
import mpmath as mp
from research.verify_guarded_precision import world, residual

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
BASE=R/'experiments/short_certificate_v1';OUT=R/'results/short_certificate_audit_v1.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def frame(row):
    view=np.eye(4);view[:3,:3]=row[20:29].reshape(3,3).T;view[3,:3]=row[29:32]
    return dict(mean=row[:3],scales=row[3:6],quat=row[6:10] if row[10] else None,
                R=row[11:20].reshape(3,3),view=view,qray=row[32:35],amplitude=row[35])
def same(a,b):return bool(a.shape==b.shape and a.dtype==b.dtype==np.dtype('float64') and np.array_equal(a.view(np.uint64),b.view(np.uint64)))

def main():
    assert not OUT.exists()
    mp.mp.dps=240;m=read(BASE/'manifest.json');s=read(BASE/'summary.json');assert m['status']==s['status']=='complete'
    counts={}
    for key,root in [('source_sha256',ROOT),('artifacts_sha256',BASE),('generated_sha256',BASE)]:
        for name,digest in m[key].items():assert sha(root/name)==digest,name
        counts[key]=len(m[key])
    with zipfile.ZipFile(BASE/'sources.zip') as z:
        assert set(z.namelist())==set(m['source_sha256'])
        for name,digest in m['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    with np.load(BASE/'inputs.npz') as d:source=d['source'];sections=d['sections'];scalar_inputs=d['scalar_inputs']
    with np.load(BASE/'outputs.npz') as d:
        packets=d['variant_packets'].copy();roots=d['variant_roots'].copy();scalars=d['scalar_intervals'].copy()
        assert same(packets,d['control_packets'])
        with np.load(R/'experiments/reciprocal_root_v2/outputs.npz') as prior:
            assert same(d['control_packets'],prior['variant_packets']) and same(d['control_roots'],prior['variant_roots'])
    assert source.shape==(7322,36) and sections.shape==(110,2) and roots.shape==(110,14)
    assert scalar_inputs.shape==(774,2) and scalars.shape==(774,2)
    scalar_bad=[];scalar_rows=[]
    # Independent elementary library oracle, distinct from the run's Decimal
    # + 128-term small-argument implementation. No double libm oracle.
    for i,(op,x) in enumerate(scalar_inputs):
        value=mp.mpf(float(x));kind=int(op)
        actual=mp.log(value) if kind==0 else mp.exp(value) if kind==1 else mp.log1p(value) if kind==2 else mp.expm1(value) if kind==3 else 1+value
        lo,hi=[mp.mpf(float(v)) for v in scalars[i]];okay=bool(mp.isfinite(lo) and mp.isfinite(hi) and lo<=actual<=hi)
        if not okay:scalar_bad.append(i)
        scalar_rows.append(dict(index=i,op=kind,pass_enclosure=okay))
    meta=read(BASE/'targets_posthoc_only.json');rows=[];bad=[];fieldchecks=0;tau=Fraction(1,1000)
    for i,(start,n) in enumerate(sections):
        label=meta['sections'][i];row=roots[i];mode=int(row[1]);params=[world(frame(source[j])) for j in range(start,start+n)]
        for j,c in enumerate(params):
            p=packets[start+j,mode];assert p[0]==0
            for name,a,b in [('p',1,2),('rho',3,4),('E',5,6),('alpha',24,25)]:
                value=c[name];allow=mp.mpf('1e-200')*max(1,abs(value));fieldchecks+=1
                if not mp.mpf(float(p[a]))-allow<=value<=mp.mpf(float(p[b]))+allow:bad.append(dict(name=label['name'],layer=j,field=name))
        if row[0]==1:
            assert np.isfinite(row).all() and row[12] in (1,2)
            z,lo,hi=map(Fraction,row[2:5]);assert hi-lo<=2*tau and z-lo<=tau and hi-z<=tau
            left=residual(params,mp.mpf(float(row[3])));right=residual(params,mp.mpf(float(row[4])))
            if not left<0<right:bad.append(dict(name=label['name'],field='world_endpoint_sign'))
            for actual,a,b,scale in [(left,5,6,9),(right,7,8,10)]:
                value=actual*mp.exp(-mp.mpf(float(row[scale])));lower=mp.mpf(float(row[a]));upper=mp.mpf(float(row[b]))
                allow=mp.mpf('1e-200')*max(abs(value),abs(lower),abs(upper))
                if not lower-allow<=value<=upper+allow:bad.append(dict(name=label['name'],field='world_scaled_enclosure'))
            target=mp.mpf(meta['targets'][label['name']]['camera_ray_depth'])
            assert mp.mpf(float(row[3]))<=target<=mp.mpf(float(row[4])) and abs(mp.mpf(float(row[2]))-target)<=mp.mpf('1e-3')
            record=dict(name=label['name'],classification='finite',mode=mode,route='short' if row[12]==1 else 'strict',strict_attempts=int(row[13]),left_H=mp.nstr(left,100),right_H=mp.nstr(right,100))
        else:
            exact=all(c['E']==0 for c in params);beta_exact=Fraction(1)
            if exact:
                for c in params:beta_exact*=1-c['A']
            beta=mp.fprod(1-c['alpha'] for c in params)
            classification='critical_no_finite_root' if exact and beta_exact==Fraction(1,2) else 'insufficient_mass_no_finite_root' if beta>mp.mpf('.5') else 'unresolved'
            if classification!=label['expected']:bad.append(dict(name=label['name'],field='world_mass_classification'))
            assert np.isnan(row[2:12]).all() and row[12]==row[13]==0
            record=dict(name=label['name'],classification=classification,mode=mode,terminal=mp.nstr(beta,100))
        rows.append(record)
    pairs=read(BASE/'paired_timing.json')['rows'];assert len(pairs)==5
    for i,p in enumerate(pairs):
        assert p['pair']==i and p['order']==(['control','variant'] if i%2==0 else ['variant','control'])
        for key in ['preprocess','root','combined']:
            c=p['results']['control'][key]['event_ms'];v=p['results']['variant'][key]['event_ms'];assert c>0 and v>0
            assert math.isclose(p['ratios'][key],v/c,rel_tol=1e-15)
    ratio=math.exp(sum(math.log(p['ratios']['combined']) for p in pairs)/5)
    assert math.isclose(ratio,s['paired_event_ratio_geometric_mean']['combined'],rel_tol=1e-15)
    numeric_pass=not scalar_bad and not bad and len(rows)==110 and sum(r['classification']=='finite' for r in rows)==106
    adoption=numeric_pass and s['development_numerical_timing_gate_pass'] and ratio<=.95
    result=dict(status='complete',checked_utc=datetime.now(timezone.utc).isoformat(),archive_checks=counts,independent_precision=240,
        scalar_checks=774,scalar_failure_indices=scalar_bad,world_endpoint_pairs=106,no_root_models=4,used_coefficient_checks=fieldchecks,
        failures=bad,numerical_gate_pass=numeric_pass,paired_combined_ratio=ratio,local_adoption_gate_pass=bool(adoption),rows=rows,scalar_rows=scalar_rows,
        scope='Independent MP240 covariance/elementary checks and exact widths on all fixed saved inputs. Local adoption gate only; no full-render/GT/backward/novelty evidence.')
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','scalar_rows','scope')},indent=2))
    if not numeric_pass:raise AssertionError('Independent numerical gate failed; audit saved.')
if __name__=='__main__':main()
