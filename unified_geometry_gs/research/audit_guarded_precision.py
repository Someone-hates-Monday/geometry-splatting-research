"""Locked CPU conditional-precision development and fresh confirmation."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction
import argparse,hashlib,json,os,time,traceback,zipfile
import numpy as np
import mpmath as mp
from research.audit_coherent_factor import groups,frame
from research.coherent_factor import rotation,construct,store,decode
from research.guarded_precision import enclosure,bounds,certify,source_coefficients,double_root,exact_mpf
from research.exact_outlier_root_oracle import Coefficients
from research.exact_outlier_world_reference import ContinuousModel,matrix,vector,dot,rotation as mp_rotation

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def txt(v):return mp.nstr(v,140)
def pack_frame(item):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in item.items()}
def serialize_node(c):
    return {k:{'bounds':list(map(txt,bounds(v))),'exact_endpoints':[[int(x) for x in t] for t in v._mpi_]} if hasattr(v,'_mpi_') else v for k,v in c.items()}


def fresh():
    rng=np.random.default_rng(13019);models=[]
    def f(mean,scale,amp,quat=(1.,0.,0.,0.)):
        return frame(np.asarray(mean,np.float32),np.asarray(scale,np.float32),[0.,0.,1.],np.float32(amp),np.eye(4),quat=np.asarray(quat,np.float32))
    for j in range(64):
        items=[]
        for k,z in enumerate((.4,.7,1.1,1.5)):
            q=np.r_[1.,rng.uniform(-.025,.025,3)];q=q/np.linalg.norm(q)
            items.append(f([*rng.uniform(-.001,.001,2),z+rng.uniform(-.025,.025)],rng.uniform(.025,.07,3),rng.uniform(.75,.95),q))
        models.append(dict(name=f'ordinary_{j:02d}',kind='ordinary',items=items,expected_class='finite'))
    for j in range(16):
        A=np.float32(rng.uniform(.6,.9));sx=np.float32(rng.uniform(.05,.2));sy=np.float32(rng.uniform(.05,.2))
        y=np.float32(float(sy)*np.sqrt(2*np.log(float(A)/.5)));sz=np.float32(rng.uniform(.002,.015));gap=rng.uniform(.25,1.)
        items=[f([0.,y,.5],[sx,sy,sz],A),f([0.,0.,.5+gap],[sx,sy,sz],.5)]
        models.append(dict(name=f'near_threshold_{j:02d}',kind='near_threshold',items=items,expected_class='finite'))
    for j,(s1,s2) in enumerate(((.002,.002),(.005,.013),(.01,.003),(.015,.007))):
        items=[f([0.,0.,.5],[s1]*3,.5),f([0.,0.,1.5],[s2]*3,.5)]
        models.append(dict(name=f'exact_critical_{j}',kind='exact_critical_two_layer',items=items,expected_class='finite'))
    for j,y in enumerate((1e-20,1e-15,1e-10,1e-5)):
        items=[f([0.,y,.5],[.005]*3,.5),f([0.,0.,1.5],[.005]*3,.5)]
        models.append(dict(name=f'tiny_offset_{j}',kind='tiny_offset',items=items,expected_class='finite'))
    for j in range(4):
        items=[f([0.,0.,.5+j*.1],[.01]*3,.5)] if j<2 else [f([0.,0.,.5+k*.3],[.01]*3,.1) for k in range(3)]
        models.append(dict(name=f'no_root_{j}',kind='no_root',items=items,expected_class='critical_no_finite_root' if j<2 else 'insufficient_mass_no_finite_root'))
    assert len(models)==92 and sum(x['expected_class']=='finite' for x in models)==88
    return models


def independent_reference(item,dps):
    with mp.workdps(dps):
        A=matrix(item['view'][:3,:3]).T;Ai=A**-1;v=vector(item['view'][3,:3]);origin=-Ai*v
        q=vector(item['qray']);qn=mp.sqrt(dot(q,q));d=Ai*q/qn;delta=vector(item['mean'])-origin
        R=matrix(item['R']) if item['quat'] is None else mp_rotation([mp.mpf(float(v)) for v in item['quat']])
        S=mp.diag([mp.mpf(float(v))**2 for v in item['scales']]);P=(R*S*R.T)**-1
        a=dot(d,P*d);p=dot(d,P*delta)/a;r=delta-p*d;E=dot(r,P*r)
        # Exact dyadic collinearity proves E=0 independently of covariance rounding.
        from research.guarded_precision import camera_exact
        cam=camera_exact(item);qf=[Fraction(float(v)) for v in item['qray']]
        if all(cam[i]*qf[2]==cam[2]*qf[i] for i in (0,1)):
            ratio=cam[2]/qf[2];p=mp.mpf(ratio.numerator)/ratio.denominator*qn;E=mp.mpf(0)
        assert a>0 and E>=0
        rho=mp.sqrt(a);alpha=min(mp.mpf(float(np.float32(.99))),mp.mpf(item['amplitude'])*mp.exp(-E/2))
        return {k:mp.nstr(x,dps) for k,x in dict(peak=p,rho=rho,alpha=alpha,energy=E).items()}


def coverage(c,ref):
    failures=[]
    for k,field in (('p','peak'),('rho','rho'),('alpha','alpha'),('energy','energy')):
        lo,hi=bounds(c[k]);value=mp.mpf(ref[field])
        # Archived MP80 rows are rounded decimal diagnostics, not certified real endpoints.
        allowance=mp.mpf('1e-75')*max(mp.mpf(1),abs(value))
        if value<lo-allowance or value>hi+allowance:failures.append(k)
    return failures


def classify(nodes):
    interval=mp.iv.mpf(1)
    for c in nodes:interval*=1-c['alpha']
    lo,hi=bounds(interval)
    if hi<mp.mpf('.5'):kind='finite'
    elif lo>mp.mpf('.5'):kind='insufficient_mass_no_finite_root'
    elif lo==hi==mp.mpf('.5'):kind='critical_no_finite_root'
    else:kind='mass_unresolved'
    return kind,dict(terminal_bounds=list(map(txt,(lo,hi))))


def low_candidate(items,coeff=None):
    if coeff is None:
        rows=[]
        for item in items:
            U,mu,_=construct(item['mean'],item['scales'],item['R'],item['view'],item['quat']);up,m=store(U,mu,'qr_mixed')
            value,rho,_,_=decode(up,m,item['qray'],item['amplitude']);rows.append([value[0],rho,value[2]])
        coeff=np.asarray(rows,np.float32)
    if len(coeff)==2 and np.all(coeff[:,2]==np.float32(.5)):
        p,r=coeff[:,0].astype(float),coeff[:,1].astype(float)
        return float(np.dot(p,r)/sum(r)),'F32_two_equal_half'
    result=Coefficients(coeff[:,2],coeff[:,0],coeff[:,1]).root64()
    return result.get('depth'),result['status']


def reference_root(rows):
    alpha=[mp.mpf(r['alpha']) for r in rows];p=[mp.mpf(r['peak']) for r in rows];rho=[mp.mpf(r['rho']) for r in rows]
    if len(rows)==2 and all(a==mp.mpf('.5') for a in alpha):
        t=(rho[0]*p[0]+rho[1]*p[1])/(rho[0]+rho[1]);return {'status':'finite','camera_ray_depth':txt(t),'method':'analytic_equal_half_two_layer'}
    binary=[float(a) for a in alpha] if all(mp.mpf(float(a))==a for a in alpha) else None
    result=ContinuousModel(alpha,p,rho,binary_alpha=binary).root()
    if result['status']=='critical_no_finite_root':return result
    return result


def evaluate(name,items,low,expected,ref,coeff=None):
    low_error=None
    try:
        if any(c is None for c in low):raise ValueError('One or more low parameter enclosures unresolved')
        kind,mass=classify(low);z,solver=low_candidate(items,coeff) if kind=='finite' else (None,'mass_classification')
        passed,certificate=certify(low,z) if z is not None else (kind in ('critical_no_finite_root','insufficient_mass_no_finite_root'),{})
    except Exception as e:kind='unresolved';mass={};z=None;solver='low_exception';passed=False;certificate={};low_error=str(e)
    fallback=not passed
    high=[];high_error=None;high_cover=[]
    if fallback:
        try:
            high=[enclosure(x,high=True) for x in items];kind,mass=classify(high)
            if kind=='finite':z,solver=double_root([source_coefficients(x) for x in items]);passed,certificate=certify(high,z) if z is not None else (False,{})
            else:z=None;solver='high_mass_classification';passed=kind in ('critical_no_finite_root','insufficient_mass_no_finite_root');certificate={}
        except Exception as e:kind='unresolved';z=None;solver='high_exception';passed=False;high_error=str(e);certificate={}
    target=mp.mpf(ref['camera_ray_depth']) if ref['status']=='finite' else None
    error=abs(mp.mpf(z)-target) if z is not None and target is not None else None
    contains=bool(target is not None and certificate and mp.mpf(certificate['lo'])<=target<=mp.mpf(certificate['hi']))
    gate=bool(passed and kind==expected and (expected!='finite' or contains and error<=mp.mpf('1e-3')))
    return dict(name=name,expected_class=expected,classification=kind,fallback=fallback,certified=passed,certificate=certificate,mass=mass,
                candidate_depth=z,solver=solver,low_error=low_error,high_error=high_error,reference=ref,
                absolute_error=None if error is None else txt(error),reference_in_certificate=contains,gate_pass=gate,
                high_enclosures=[serialize_node(x) for x in high])


def run(out,state,data,new):
    start=time.perf_counter();mp.mp.dps=160;mp.iv.prec=53
    old=ROOT/'research/experiments/coherent_factor_v2';old_manifest=read(old/'manifest.json');assert old_manifest['status']=='complete'
    summaries=[];roots=[];allboxes={}
    for name,items,sections in data:
        oracle_path=old/(name+'_oracle.json');assert sha(oracle_path)==old_manifest['artifacts_sha256'][oracle_path.name]
        refs=read(oracle_path)['rows'];nodes=[];errors=[]
        for i,(item,ref) in enumerate(zip(items,refs)):
            try:
                c=enclosure(item);bad=coverage(c,ref);nodes.append(c)
                if bad:errors.append(dict(index=i,fields=bad))
            except Exception as e:nodes.append(None);errors.append(dict(index=i,exception=str(e)))
        allboxes[name]=nodes
        save(out/(name+'_enclosures.json'),dict(rows=[None if x is None else serialize_node(x) for x in nodes],failures=errors))
        summaries.append(dict(group=name,inputs=len(items),low_enclosure_failures=len(errors),failure_rows=errors))
        if sections:
            p=old/(name+'.npz');assert sha(p)==old_manifest['artifacts_sha256'][p.name]
            with np.load(p) as d:coeff=d['qr_mixed_F32_peak_rho_alpha'].copy()
            summary=read(old/'summary.json')
            for section in sections:
                a,b=section['start'],section['end'];target=next(r['reference_root'] for r in summary['root_rows'] if r['group']==name and r['section']['name']==section['name'])
                roots.append(evaluate(section['name'],items[a:b],nodes[a:b],'finite',target,coeff[a:b]))
        save(out/'progress.json',dict(utc=now(),stage=name,root_count=len(roots)));print(f'{name}: {len(items)} boxes; failures={len(errors)}',flush=True)
    newrows=[];coefficient_bad=[];dual_bad=[]
    for i,model in enumerate(new):
        refs80=[independent_reference(x,80) for x in model['items']];refs140=[independent_reference(x,140) for x in model['items']]
        for j,(a,b) in enumerate(zip(refs80,refs140)):
            if any(abs(mp.mpf(a[k])-mp.mpf(b[k]))>mp.mpf('1e-70')*max(1,abs(mp.mpf(b[k]))) for k in a):dual_bad.append([i,j])
        nodes=[]
        for j,(item,ref) in enumerate(zip(model['items'],refs140)):
            try:
                c=enclosure(item);bad=coverage(c,ref);nodes.append(c)
                if bad:coefficient_bad.append(dict(model=i,index=j,fields=bad))
            except Exception as e:nodes.append(None);coefficient_bad.append(dict(model=i,index=j,exception=str(e)))
        target=reference_root(refs140)
        rr=evaluate(model['name'],model['items'],nodes,model['expected_class'],target);rr.update(kind=model['kind'],reference80=refs80,reference140=refs140,low_enclosures=[None if x is None else serialize_node(x) for x in nodes]);newrows.append(rr)
        save(out/'fresh_progress.json',dict(utc=now(),completed_models=i+1,rows=newrows))
        if (i+1)%8==0:print(f'fresh {i+1}/92: failures={sum(not x["gate_pass"] for x in newrows)}',flush=True)
    save(out/'root_rows.json',dict(old=roots,fresh=newrows))
    allroots=roots+newrows
    coefpass=all(x['low_enclosure_failures']==0 for x in summaries) and not coefficient_bad
    rootpass=all(r['gate_pass'] for r in allroots)
    result=dict(status='complete',groups=summaries,fresh_coefficient_failures=coefficient_bad,dual_reference_failures=dual_bad,
                old_root_count=len(roots),fresh_model_count=len(newrows),finite_models=sum(r['expected_class']=='finite' for r in allroots),
                no_root_models=sum(r['expected_class']!='finite' for r in allroots),low_certified=sum(r['certified'] and not r['fallback'] for r in allroots),
                fallback_models=sum(r['fallback'] for r in allroots),root_failure_names=[r['name'] for r in allroots if not r['gate_pass']],
                coefficient_gate_pass=coefpass,root_gate_pass=rootpass,overall_cpu_gate_pass=coefpass and rootpass and not dual_bad,
                max_certified_absolute_error=txt(max([mp.mpf(r['absolute_error']) for r in allroots if r['gate_pass'] and r['absolute_error'] is not None],default=mp.mpf(0))),
                wall_seconds=time.perf_counter()-start,scope='CPU53 directed intervals with unbounded exponent; not IEEE FP64 GPU or full renderer/backward/GT/efficiency evidence')
    save(out/'summary.json',result)
    for name,h in state['source_sha256'].items():assert sha(ROOT/name)==h
    state.update(status='complete',completed_utc=now(),wall_seconds=result['wall_seconds'],artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','progress.json','fresh_progress.json')})
    save(out/'manifest.json',state);print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    data,files=groups();new=fresh();old=ROOT/'research/experiments/coherent_factor_v2'
    sources=[Path(__file__),ROOT/'research/guarded_precision.py',ROOT/'research/GUARDED_PRECISION_PROTOCOL.md',ROOT/'research/audit_coherent_factor.py',ROOT/'research/coherent_factor.py',ROOT/'research/exact_outlier_root_oracle.py',ROOT/'research/exact_outlier_world_reference.py',old/'manifest.json',old/'summary.json']+files
    sources+=[old/(name+suffix) for name,_,_ in data for suffix in ('.npz','_oracle.json')]
    out.mkdir(parents=True);save(out/'fresh_inputs.json',dict(seed=13019,models=[dict(**{k:v for k,v in x.items() if k!='items'},items=[pack_frame(f) for f in x['items']]) for x in new]))
    state=dict(created_utc=now(),status='running',pid=os.getpid(),numpy=np.__version__,mpmath=mp.__version__,source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources},fresh_input_sha256=sha(out/'fresh_inputs.json'))
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state,data,new)
    except BaseException as e:
        state.update(status='failed',failure_type=type(e).__name__,failure=str(e),failed_utc=now());save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
