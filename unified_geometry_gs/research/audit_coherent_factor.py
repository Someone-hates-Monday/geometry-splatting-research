"""Fixed coherent QR cache development on real, stress and critical layer inputs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,time,traceback,zipfile
import numpy as np
import mpmath as mp
from research.coherent_factor import rotation,construct,store,decode,VARIANTS
from research.exact_outlier_world_reference import ContinuousModel
from research.exact_outlier_root_oracle import Coefficients

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def txt(v):return mp.nstr(v,80)


def frame(mean,scales,qray,amplitude,view,quat=None,R=None):
    return dict(mean=np.asarray(mean,dtype=float),scales=np.asarray(scales,dtype=float),qray=np.asarray(qray,dtype=float),
                amplitude=float(amplitude),view=np.asarray(view,dtype=float),quat=None if quat is None else np.asarray(quat,dtype=float),
                R=rotation(quat) if quat is not None else np.asarray(R,dtype=float))


def groups():
    real=[];files=[]
    production=ROOT/'research/experiments/production_packet_v3'
    pm=read(production/'manifest.json');assert pm['status']=='complete'
    for i in range(24):
        p=production/f'point_{i:02d}.npz';assert sha(p)==pm['artifacts_sha256'][p.name];files.append(p)
        with np.load(p) as d:
            for j in range(len(d['means'])):
                real.append(frame(d['means'][j],d['filtered_scale'][j],d['qray'][j],d['conic_opacity'][j,3],d['world_view_transform'],d['normalized_quaternion'][j]))
    assert len(real)==1992
    outliers=[];sections=[]
    capture=ROOT/'research/experiments/exact_outlier_capture_v3';cm=read(capture/'manifest.json');assert cm['status']=='complete'
    for i in range(9):
        p=capture/f'ray_{i:02d}.npz';assert sha(p)==cm['artifacts_sha256'][p.name];files.append(p)
        begin=len(outliers)
        with np.load(p) as d:
            for j in np.flatnonzero(d['accepted_mask']):
                outliers.append(frame(d['raw_parameters'][j,:3],d['filtered_scales'][j],np.r_[d['query_xy'],1.],d['conic_opacity'][j,3],d['world_view_transform'],d['normalized_quaternions'][j]))
        sections.append(dict(name=f'outlier_{i}',start=begin,end=len(outliers)))
    assert len(outliers)==772
    stress=[];p=ROOT/'research/experiments/world_preprocess_stress_v1/results.json';files.append(p)
    for c in read(p)['cases']:
        view=np.asarray(c['view']);mean=np.asarray(c['mean']);camera=view[:3,:3].T@mean+view[3,:3];q=camera/camera[2]
        for query in (np.array([0.,0.,1.]),q,q+np.array([.1,0.,0.])):
            stress.append(frame(mean,c['filtered_scales'],query,c['opacity'],view,c['normalized_device_quaternion']))
    assert len(stress)==132
    synthetic=[];p=ROOT/'research/experiments/packet_input_confirmation_v1/inputs.npz';files.append(p)
    with np.load(p) as d:
        for j in range(4096):synthetic.append(frame(d['z'][j]*d['qmean'][j],d['axes'][j],d['qray'][j],d['amplitude'][j],np.eye(4),R=d['rotation'][j]))
    critical=[];critical_sections=[]
    for eps in np.array([.002,.005,.01],dtype=np.float32):
        y=np.float32(float(eps)*np.sqrt(2*np.log(1.5)))
        for label,offset in (('below',np.nextafter(y,np.float32(-np.inf))),('nearest',y),('above',np.nextafter(y,np.float32(np.inf)))):
            start=len(critical)
            for z,yy,amp in ((.5,float(offset),.75),(1.5,0.,.5)):
                critical.append(frame([0.,yy,z],[float(eps)]*3,[0.,0.,1.],amp,np.eye(4),quat=[1.,0.,0.,0.]))
            critical_sections.append(dict(name=f'critical_eps{float(eps):.9g}_{label}',start=start,end=len(critical)))
    files += [production/'manifest.json',capture/'manifest.json']
    return [('real1992',real,[]),('outliers772',outliers,sections),('stress132',stress,[]),('synthetic4096',synthetic,[]),('critical18',critical,critical_sections)],files


def oracle(item):
    from research.exact_outlier_world_reference import matrix,vector,rotation as mp_rotation,dot
    with mp.workdps(80):
        mean=vector(item['mean']);A=matrix(item['view'][:3,:3]).T;Ai=A**-1;origin=-Ai*vector(item['view'][3,:3])
        q=vector(item['qray']);direction=Ai*q/mp.sqrt(dot(q,q))
        R=matrix(item['R']) if item['quat'] is None else mp_rotation([mp.mpf(float(v)) for v in item['quat']])
        scales=[mp.mpf(float(v)) for v in item['scales']];cov=R*mp.diag([v*v for v in scales])*R.T;P=cov**-1
        delta=mean-origin;a=dot(direction,P*direction);b=dot(direction,P*delta);peak=b/a
        perpendicular=delta-peak*direction;energy=dot(perpendicular,P*perpendicular)
        assert a>0 and energy>=0
        rho=mp.sqrt(a);alpha=min(mp.mpf(float(np.float32(.99))),mp.mpf(item['amplitude'])*mp.exp(-energy/2))
        return {'peak':txt(peak),'rho':txt(rho),'sigma':txt(1/rho),'alpha':txt(alpha),'energy':txt(energy)}


def symbolic_inverse():
    import sympy as sp
    w,x,y,z=sp.symbols('w x y z',real=True);v=sp.Matrix([x,y,z]);k=w*w+x*x+y*y+z*z-1;D=1+4*k*(v.dot(v))
    R=sp.Matrix([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
    residual=(R.T+4*k*v*v.T)*R-D*sp.eye(3);checks=[sp.expand(t)==0 for t in residual]
    assert all(checks)
    return {'identity_components':9,'all_exact_zero':True,'domain':'D!=0; D=(1-2||v||²)²+4w²||v||²'}


def run(out,state,data):
    start=time.perf_counter();summaries=[];rootrows=[];mp.mp.dps=120
    state['symbolic_inverse']=symbolic_inverse();save(out/'manifest.json',state)
    for name,items,sections in data:
        refs=[];payloads={variant:[] for variant in VARIANTS};means={variant:[] for variant in VARIANTS}
        outputs={variant:[] for variant in VARIANTS};f32={variant:[] for variant in VARIANTS};valid={variant:[] for variant in VARIANTS}
        for i,item in enumerate(items):
            refs.append(oracle(item));U,mu,F=construct(item['mean'],item['scales'],item['R'],item['view'],item['quat'])
            for variant in VARIANTS:
                upper,m=store(U,mu,variant);value,rho,energy,ok=decode(upper,m,item['qray'],item['amplitude'])
                payloads[variant].append(upper);means[variant].append(m);outputs[variant].append(value);f32[variant].append(np.asarray([value[0],rho,value[2]],np.float32));valid[variant].append(ok)
        reference=np.array([[float(mp.mpf(r[k])) for k in ('peak','sigma','alpha')] for r in refs])
        rows=[];arrays={'reference':reference,'qray':np.array([x['qray'] for x in items]),'amplitude':np.array([x['amplitude'] for x in items])}
        for variant in VARIANTS:
            output=np.array(outputs[variant]);coeff=np.array(f32[variant]);actual=np.column_stack((coeff[:,0],1/coeff[:,1].astype(float),coeff[:,2]))
            errors=np.column_stack((abs(actual[:,0]-reference[:,0])/np.maximum(1e-8,abs(reference[:,0])),abs(actual[:,1]-reference[:,1])/np.maximum(1e-8,abs(reference[:,1])),abs(actual[:,2]-reference[:,2])))
            bad=(~np.array(valid[variant]))|np.any(~np.isfinite(errors),axis=1)|np.any(errors>np.array([1e-5,1e-4,1e-4]),axis=1)
            rows.append({'variant':variant,'cache_bytes_per_primitive':dict(qr32=36,qr_mixed=48,qr64=72)[variant],
                         'inputs':len(items),'bad':int(sum(bad)),'bad_ids':np.flatnonzero(bad).tolist(),'max_errors':np.max(errors,axis=0).tolist()})
            arrays.update({variant+'_upper':np.array(payloads[variant]),variant+'_mean':np.array(means[variant]),variant+'_double_coefficients':output,variant+'_F32_peak_rho_alpha':coeff,variant+'_errors':errors,variant+'_bad':bad})
        np.savez_compressed(out/(name+'.npz'),**arrays);save(out/(name+'_oracle.json'),{'rows':refs,'dps':80,'sections':sections})
        for section in sections:
            lo,hi=section['start'],section['end'];reference_model=ContinuousModel([mp.mpf(r['alpha']) for r in refs[lo:hi]], [mp.mpf(r['peak']) for r in refs[lo:hi]],[mp.mpf(r['rho']) for r in refs[lo:hi]])
            target=reference_model.root();rr={'group':name,'section':section,'reference_root':target,'variants':{}}
            assert target['status']=='finite'
            t=mp.mpf(target['camera_ray_depth'])
            for variant in VARIANTS:
                c=np.array(f32[variant])[lo:hi];model=Coefficients(c[:,2],c[:,0],c[:,1]);r=model.root()
                decoded=np.array(outputs[variant])[lo:hi];uncapped_rho=np.array([1/v[1] for v in decoded]);double_model=ContinuousModel([mp.mpf(float(v)) for v in decoded[:,2]],[mp.mpf(float(v)) for v in decoded[:,0]],[mp.mpf(float(v)) for v in uncapped_rho],binary_alpha=decoded[:,2]);dr=double_model.root()
                err=abs(mp.mpf(r['depth'])-t) if r['status']=='finite' else None
                derr=abs(mp.mpf(dr['camera_ray_depth'])-t) if dr['status']=='finite' else None
                rr['variants'][variant]={'F32_model_root':r,'F32_model_abs_error':None if err is None else txt(err),'F32_model_root_gate_pass':bool(err is not None and err<=mp.mpf('1e-3')),
                                         'unrounded_double_model_root':dr,'unrounded_double_model_abs_error':None if derr is None else txt(derr)}
            rootrows.append(rr)
        summaries.append({'group':name,'inputs':len(items),'variants':rows});save(out/'progress.json',{'utc':now(),'completed_group':name,'root_models':len(rootrows)*3});print(f'{name} complete: {[(r["variant"],r["bad"]) for r in rows]}',flush=True)
    primary_coef=all(next(r for r in g['variants'] if r['variant']=='qr_mixed')['bad']==0 for g in summaries)
    primary_root=all(r['variants']['qr_mixed']['F32_model_root_gate_pass'] for r in rootrows)
    result={'status':'complete','primary':'qr_mixed','symbolic_inverse':state['symbolic_inverse'],'groups':summaries,'root_rows':rootrows,
            'primary_coefficient_gate_pass':primary_coef,'primary_root_gate_pass':primary_root,'primary_development_gate_pass':primary_coef and primary_root,
            'root_failures':{v:sum(not r['variants'][v]['F32_model_root_gate_pass'] for r in rootrows) for v in VARIANTS},
            'wall_seconds':time.perf_counter()-start,'limits':['CPU construction and decoding, not CUDA or latency/backward/GT verification.',
                'Whole fixed sets and nearcritical fixtures; no prevalence estimate.',
                'MP target follows saved activated covariance and actual common amplitude; raw-physical activation is a different model.',
                'Critical fixtures test mathematical ray models, not complete renderer selection/bracket or image visibility.',
                'Unrounded coefficient roots are separate diagnostics and never replace the primary F32-root gate.']}
    save(out/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h
    state.update(status='complete',completed_utc=now(),wall_seconds=result['wall_seconds'],artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',state);print(json.dumps({k:v for k,v in result.items() if k not in ('groups','root_rows','limits')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    data,files=groups();sources=[Path(__file__),ROOT/'research/coherent_factor.py',ROOT/'research/COHERENT_FACTOR_PROTOCOL.md',ROOT/'research/exact_outlier_world_reference.py',ROOT/'research/exact_outlier_root_oracle.py']+files
    out.mkdir(parents=True);state={'created_utc':now(),'status':'running','pid':os.getpid(),'numpy':np.__version__,'mpmath':mp.__version__,'primary':'qr_mixed','source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},'scope':'CPU fixed development only'}
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for source in sources:z.write(source,source.relative_to(ROOT).as_posix())
    try:run(out,state,data)
    except BaseException as e:
        state.update(status='failed',failure_type=type(e).__name__,failure=str(e),failed_utc=now());save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
