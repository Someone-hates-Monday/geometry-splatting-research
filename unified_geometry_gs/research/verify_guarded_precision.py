"""Independent world-covariance endpoint and exact bracket-width verification."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,zipfile
import numpy as np
import mpmath as mp
from research.audit_coherent_factor import groups
from research.exact_outlier_world_reference import matrix,vector,dot,rotation

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research/experiments/guarded_precision_v2'
OUT=ROOT/'research/results/guarded_precision_audit_v1.json'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mprational(x):return mp.mpf(x.numerator)/x.denominator


def world(item):
    mean=vector(item['mean']);A=matrix(np.asarray(item['view'])[:3,:3]).T;Ai=A**-1;t=vector(np.asarray(item['view'])[3,:3]);origin=-Ai*t
    q=vector(item['qray']);qn=mp.sqrt(dot(q,q));d=Ai*q/qn;v=mean-origin
    R=matrix(item['R']) if item['quat'] is None else rotation([mp.mpf(float(x)) for x in item['quat']])
    P=(R*mp.diag([mp.mpf(float(x))**2 for x in item['scales']])*R.T)**-1
    a=dot(d,P*d);p=dot(d,P*v)/a;r=v-p*d;E=dot(r,P*r)
    # Exact input collinearity gives exact E=0, independent of inverse-covariance error.
    Af=np.asarray(item['view'])[:3,:3].T;tf=np.asarray(item['view'])[3,:3]
    cam=[sum((Fraction(float(Af[i,j]))*Fraction(float(item['mean'][j])) for j in range(3)),Fraction(float(tf[i]))) for i in range(3)]
    qf=[Fraction(float(x)) for x in item['qray']]
    if all(cam[i]*qf[2]==cam[2]*qf[i] for i in (0,1)):p=mprational(cam[2]/qf[2])*qn;E=mp.mpf(0)
    assert a>0 and E>=0
    A0=Fraction(float(item['amplitude']));alpha=mprational(A0)*mp.exp(-E/2);cap=Fraction(float(np.float32(.99)))
    if alpha>=mprational(cap):A0=cap;E=mp.mpf(0);alpha=mprational(cap)
    return dict(A=A0,E=E,p=p,rho=mp.sqrt(a),alpha=alpha)


def residual(params,t):
    # Independent stable product-log evaluation: no candidate coefficients or interval helpers.
    beta=Fraction(1);shift=[];tails=[]
    for c in params:
        amp=mprational(c['A']);x=(t-c['p'])*c['rho'];tail=-mp.log1p(-c['alpha']*mp.exp(-x*x/2))/2
        if t>c['p']:
            beta*=1-c['A'];shift.append(-mp.log1p(amp/(1-amp)*(-mp.expm1(-c['E']/2))));tails.append(-tail)
        else:tails.append(tail)
    return mp.log1p(mprational((Fraction(1,2)-beta)/beta))+mp.fsum(shift)+mp.fsum(tails)


def interval_bounds(record):
    return [mp.mpf(tuple(int(v) for v in x)) for x in record['exact_endpoints']]


def main():
    if OUT.exists():raise FileExistsError(OUT)
    mp.mp.dps=240;manifest=read(BASE/'manifest.json');assert manifest['status']=='complete'
    with zipfile.ZipFile(BASE/'sources.zip') as z:
        for n,h in manifest['source_sha256'].items():assert sha(ROOT/n)==h and hashlib.sha256(z.read(n)).hexdigest()==h
    for n,h in manifest['artifacts_sha256'].items():assert sha(BASE/n)==h
    assert sha(BASE/'fresh_inputs.json')==manifest['fresh_input_sha256']
    original,_=groups();models={};lowmodels={}
    for name,items,sections in original:
        payload=read(BASE/(name+'_enclosures.json'))['rows'];assert len(payload)==len(items)
        for section in sections:
            a,b=section['start'],section['end'];models[section['name']]=items[a:b];lowmodels[section['name']]=payload[a:b]
    fresh=read(BASE/'fresh_inputs.json')['models']
    for model in fresh:models[model['name']]=model['items']
    rows=read(BASE/'root_rows.json');allrows=rows['old']+rows['fresh'];details=[];failures=[];coeffchecks=0
    for rr in allrows:
        name=rr['name'];params=[world(x) for x in models[name]]
        nodes=rr['high_enclosures'] if rr['fallback'] else lowmodels[name] if name in lowmodels else rr['low_enclosures']
        for j,(c,n) in enumerate(zip(params,nodes)):
            for field,key in (('p','p'),('rho','rho'),('alpha','alpha'),('energy','E')):
                lo,hi=interval_bounds(n[field]);value=c[key];coeffchecks+=1
                slack=mp.mpf('1e-200')*max(1,abs(value))
                if value<lo-slack or value>hi+slack:failures.append(dict(name=name,index=j,field=field))
        cert=rr['certificate'];record=dict(name=name,finite=rr['expected_class']=='finite',fallback=rr['fallback'])
        if cert:
            lo,hi=map(lambda v:mp.mpf(float(v)),(cert['lo'],cert['hi']));L=residual(params,lo);R=residual(params,hi)
            width=Fraction(cert['hi'])-Fraction(cert['lo']);z=Fraction(rr['candidate_depth']);tau=Fraction(1,1000)
            check=bool(L<0<R and width<=2*tau and z-Fraction(cert['lo'])<=tau and Fraction(cert['hi'])-z<=tau)
            for key,value in (('left_residual_bounds',L),('right_residual_bounds',R)):
                a,b=map(mp.mpf,cert[key]);slack=mp.mpf('1e-55')*max(abs(a),abs(b),abs(value))
                if not a-slack<=value<=b+slack:failures.append(dict(name=name,field=key,value=mp.nstr(value,60)))
            if not check:failures.append(dict(name=name,field='endpoint_sign_or_exact_width'))
            record.update(left_H=mp.nstr(L,70),right_H=mp.nstr(R,70),exact_width_numerator=str(width.numerator),exact_width_denominator=str(width.denominator),sign_and_width_pass=check)
        else:
            terminal=mp.fprod(1-c['alpha'] for c in params)
            exact=all(c['E']==0 for c in params)
            terminal_exact=np.prod([1-c['A'] for c in params]) if exact else None
            kind='critical_no_finite_root' if exact and terminal_exact==Fraction(1,2) else 'insufficient_mass_no_finite_root' if terminal>mp.mpf('.5') else 'finite' if terminal<mp.mpf('.5') else 'unresolved'
            if kind!=rr['expected_class']:failures.append(dict(name=name,field='terminal_classification',kind=kind))
            record.update(classification=kind,terminal=mp.nstr(terminal,70))
        details.append(record)
    summary=read(BASE/'summary.json')
    assert len(details)==110 and sum(x['finite'] for x in details)==106 and sum(x['fallback'] for x in details)==30
    assert summary['overall_cpu_gate_pass'] and not failures
    result=dict(status='complete',source_archive_checks=len(manifest['source_sha256']),artifact_checks=len(manifest['artifacts_sha256']),
        independent_dps=240,finite_endpoint_pairs=106,no_root_models=4,source_covariance_coefficient_checks=coeffchecks,
        failures=failures,rows=details,limits=['Independent MP240 numerical reference, not a proof assistant.',
            'Saved directed intervals are trusted mpmath.iv arithmetic; no GPU exponent range, full renderer or backward validation.',
            'V1 had 81 nominal bracket widths exceed exact 2e-3 by at most 6.67911e-16; preserved; V2 rounds endpoints inward without input or gate changes.'])
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','limits')},indent=2))


if __name__=='__main__':main()
