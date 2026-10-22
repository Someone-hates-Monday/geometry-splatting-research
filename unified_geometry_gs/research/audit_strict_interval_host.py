"""Locked scalar and all finite-model endpoint checks for the shared numeric core."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
from decimal import Decimal,localcontext
import argparse,hashlib,json,math,os,struct,subprocess,time,traceback,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def dec(x):return Decimal.from_float(float(x))
def finite(x):return float(x) if math.isfinite(x) else str(float(x))
def exact_bounds(node,key):
    result=[]
    for sign,mantissa,exponent,_ in node[key]['exact_endpoints']:
        f=Fraction((-1 if sign else 1)*int(mantissa))*Fraction(2)**int(exponent)
        value=float(f);assert Fraction(value)==f,'Endpoint not exactly representable in binary64'
        result.append(value)
    return result


def scalar_inputs():
    rng=np.random.default_rng(13023);rows=[]
    logs=[float.fromhex('0x0.0000000000001p-1022'),2.**-1022,1.,2.,np.finfo(float).max,math.nextafter(1.,0.),math.nextafter(1.,math.inf),2.**-700]
    logs += [math.ldexp(1+float(rng.random()),int(rng.integers(-1074,1024))) for _ in range(256-len(logs))]
    exps=[-60000.,-40000.,-1000.,-745.,-701.,math.nextafter(-700.,-math.inf),-700.,math.nextafter(-700.,math.inf),math.nextafter(-40.,-math.inf),-40.,math.nextafter(-40.,math.inf),0.,700.]
    count=256-len(exps);exps+=rng.uniform(-700.,700.,count//2).tolist()+rng.uniform(-60000.,-700.,count-count//2).tolist()
    tiny=2.**-1074
    lp=[0.,tiny,-tiny,2.**-1022,-2.**-1022,2.**-53,-2.**-53,2.**-100,-2.**-100,.124999,-.124999,.125,-.125,1e-36,-1e-36]
    lp+=rng.uniform(-.124999,.124999,128-len(lp)).tolist()
    em=[0.,tiny,-tiny,.125,-.125,-40.,-700.,-701.,-60000.,-1.,1e-36,-1e-36]
    em+=rng.uniform(-.124999,.124999,128-len(em)).tolist()
    for op,xs in enumerate((logs,exps,lp,em,[2.**-53,-2.**-54,tiny,-tiny,0.,2.**-52])):
        rows += [dict(op=op,x=float(x)) for x in xs]
    assert len(rows)==774
    return rows


def nodes_for_old():
    result={}
    for group in ('outliers772','critical18'):
        sections=read(R/f'experiments/coherent_factor_v2/{group}_oracle.json')['sections']
        rows=read(R/f'experiments/guarded_precision_v2/{group}_enclosures.json')['rows']
        for s in sections:result[s['name']]=rows[s['start']:s['end']]
    return result


def endpoint_inputs():
    modelrows=read(R/'experiments/guarded_precision_v2/root_rows.json');models=modelrows['old']+modelrows['fresh']
    truth={r['name']:r for r in read(R/'results/guarded_precision_audit_v1.json')['rows']};old=nodes_for_old();rows=[]
    for m in models:
        if m['expected_class']!='finite':continue
        nodes=m['high_enclosures'] if m['fallback'] else old[m['name']] if m['name'] in old else m['low_enclosures']
        layers=[exact_bounds(c,'p')+exact_bounds(c,'rho')+exact_bounds(c,'energy')+[float(c['amp']['bounds'][0])] for c in nodes]
        for side in ('left','right'):
            rows.append(dict(op=5,name=m['name'],side=side,x=m['certificate']['lo' if side=='left' else 'hi'],layers=layers,independent_MP240_H=truth[m['name']][side+'_H']))
    assert len(rows)==212
    return rows


def rational_constants():
    z=Fraction(1,3);lo=Fraction.from_float(float.fromhex('0x1.62e42fefa39efp-1'));hi=Fraction.from_float(float.fromhex('0x1.62e42fefa39f0p-1'))
    lower=2*sum((z**(2*j+1)/Fraction(2*j+1) for j in range(50)),Fraction(0))
    remainder=2*z**101/(101*(1-z*z));upper=lower+remainder
    assert lo<lower and upper<hi
    assert 1009*hi<700 and 57*hi<40 and Fraction(1,2**57)/(1-Fraction(1,2**57))<Fraction(1,2**56)
    return dict(ln2_binary64_lo_hex=float(lo).hex(),ln2_binary64_hi_hex=float(hi).hex(),
        rational_lower=dict(numerator=str(lower.numerator),denominator=str(lower.denominator)),
        rational_upper=dict(numerator=str(upper.numerator),denominator=str(upper.denominator)),
        all_exact_inequalities_pass=True,claims=['ln2 lies strictly between the two binary64 endpoints.',
            'exp(-700)<2^-1009 follows from 1009*ln2_upper<700.',
            'log-tail correction for logg<-40 is bounded by 2^-56.'])


def scalar_truth(op,x):
    x=dec(x)
    if op==0:return x.ln()
    if op==1:return x.exp()
    if op==4:return 1+x
    if abs(x)>=Decimal('.125'):return (1+x).ln() if op==2 else x.exp()-1
    term=x;value=Decimal(0)
    for j in range(1,129):
        if op==2:value+=term/j if j%2 else -term/j;term*=x
        else:value+=term;term*=x/(j+1)
    return value


def run(out,state,exe,scalars,endpoints):
    start=time.perf_counter();constants=rational_constants();save(out/'exact_constants.json',constants)
    tasks=scalars+endpoints
    with (out/'tasks.bin').open('wb') as f:
        for task in tasks:
            layers=task.get('layers',[]);f.write(struct.pack('<iid',task['op'],len(layers),task['x']))
            for row in layers:f.write(struct.pack('<7d',*row))
    result=subprocess.run([str(exe.resolve()),str((out/'tasks.bin').resolve()),str((out/'device_style_outputs.bin').resolve()),str(len(tasks))],capture_output=True,text=True)
    state['native_exit_code']=result.returncode;save(out/'manifest.json',state)
    if result.returncode:raise RuntimeError(dict(code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    values=np.fromfile(out/'device_style_outputs.bin',np.float64).reshape(-1,4);assert len(values)==986
    scalarrows=[];endrows=[];bad=[]
    with localcontext() as ctx:
        ctx.prec=180;ctx.Emax=999999999;ctx.Emin=-999999999
        for i,(task,v) in enumerate(zip(tasks,values)):
            if i<len(scalars):
                truth=scalar_truth(task['op'],task['x']);passed=bool(np.isfinite(v[:2]).all() and dec(v[0])<=truth<=dec(v[1]))
                row=dict(**task,output=list(map(finite,v)),reference=str(truth),gate_pass=passed);scalarrows.append(row)
            else:
                truth=Decimal(task['independent_MP240_H'])*(-dec(v[2])).exp() if math.isfinite(v[2]) else Decimal('NaN')
                # Stored independent MP240 diagnostics retain 70 significant decimal digits.
                allowance=abs(truth)*Decimal('1e-65')
                sign=bool(v[1]<0) if task['side']=='left' else bool(v[0]>0)
                passed=bool(v[3]==1 and np.isfinite(v).all() and sign and dec(v[0])-allowance<=truth<=dec(v[1])+allowance)
                row=dict(name=task['name'],side=task['side'],endpoint=task['x'],output=list(map(finite,v)),reference_scaled=str(truth),gate_pass=passed);endrows.append(row)
            if not passed:bad.append(dict(index=i,op=task['op'],name=task.get('name'),side=task.get('side')))
    save(out/'scalar_rows.json',dict(rows=scalarrows));save(out/'endpoint_rows.json',dict(rows=endrows))
    summary=dict(status='complete',scalar_checks=len(scalarrows),scalar_failures=sum(not r['gate_pass'] for r in scalarrows),endpoint_checks=len(endrows),endpoint_failures=sum(not r['gate_pass'] for r in endrows),
        all_exact_constant_inequalities_pass=True,host_numeric_core_gate_pass=not bad,failure_rows=bad,wall_seconds=time.perf_counter()-start,
        scope='Native Windows MSVC host only, same arithmetic algorithm. Not CUDA build/execution or full GPU budget/solver.',
        reused_reference='Previously independently checked world covariance MP240 endpoint residuals; no new world data or selection.')
    save(out/'summary.json',summary)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});save(out/'manifest.json',state);print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--exe',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    sources=[Path(__file__),R/'guarded_interval.cuh',R/'guarded_interval_host.cpp',R/'guarded_interval_cuda.py',R/'STRICT_INTERVAL_PROTOCOL.md',R/'GUARDED_CUDA_PROTOCOL.md',args.exe.resolve(),R/'results/strict_interval_host_build_v1.log',R/'results/guarded_precision_audit_v1.json',R/'experiments/guarded_precision_v2/root_rows.json',R/'experiments/guarded_precision_v2/manifest.json']
    for group in ('outliers772','critical18'):sources += [R/f'experiments/guarded_precision_v2/{group}_enclosures.json',R/f'experiments/coherent_factor_v2/{group}_oracle.json']
    scalars=scalar_inputs();endpoints=endpoint_inputs();out.mkdir(parents=True)
    save(out/'scalar_inputs.json',dict(seed=13023,rows=scalars));save(out/'endpoint_inputs.json',dict(rows=endpoints))
    state=dict(status='running',created_utc=now(),pid=os.getpid(),host_compiler='MSVC14.44.35207 /O2 /std:c++17 /fp:strict /EHsc',source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources})
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state,args.exe,scalars,endpoints)
    except BaseException as e:
        state.update(status='failed',failure_type=type(e).__name__,failure=str(e),failed_utc=now());save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
