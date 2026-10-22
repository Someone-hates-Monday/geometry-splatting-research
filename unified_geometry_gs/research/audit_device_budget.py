"""Device-built factor budgets on all fixed inputs; saved brackets remain CPU provided."""
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,localcontext
import argparse,hashlib,json,os,time,traceback,zipfile
import numpy as np
import mpmath as mp
import torch
from research.audit_coherent_factor import groups
from research.guarded_preprocess_cuda import load
from research.guarded_interval_cuda import FLAGS

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def finite(v):return float(v) if np.isfinite(v) else str(float(v))
def pack(x):
    quat=np.zeros(4) if x['quat'] is None else np.asarray(x['quat']);view=np.asarray(x['view'])
    return np.r_[x['mean'],x['scales'],quat,0. if x['quat'] is None else 1.,np.asarray(x['R']).reshape(-1),view[:3,:3].T.reshape(-1),view[3,:3],x['qray'],x['amplitude']].astype(np.float64)


def inputs():
    data,files=groups();items=[];refs=[];sections=[];localgroups=[];old=read(R/'experiments/coherent_factor_v2/manifest.json')
    for name,group,part in data:
        start=len(items);items+=group;path=R/f'experiments/coherent_factor_v2/{name}_oracle.json';assert sha(path)==old['artifacts_sha256'][path.name]
        refs+=read(path)['rows'];files.append(path);localgroups.append(dict(name=name,start=start,end=len(items)))
        for s in part:sections.append(dict(name=s['name'],start=start+s['start'],end=start+s['end'],expected='finite'))
    fresh=R/'experiments/guarded_precision_v2/fresh_inputs.json';fm=read(R/'experiments/guarded_precision_v2/manifest.json');assert sha(fresh)==fm['fresh_input_sha256']
    freshroots=R/'experiments/guarded_precision_v2/root_rows.json';assert sha(freshroots)==fm['artifacts_sha256'][freshroots.name]
    rfresh={r['name']:r for r in read(freshroots)['fresh']};models=read(fresh)['models'];begin=len(items)
    for model in models:
        start=len(items);items+=model['items'];refs+=rfresh[model['name']]['reference140']
        sections.append(dict(name=model['name'],start=start,end=len(items),expected=model['expected_class']))
    localgroups.append(dict(name='fresh312',start=begin,end=len(items)));files+=[fresh,freshroots,R/'experiments/coherent_factor_v2/manifest.json',R/'experiments/guarded_precision_v2/manifest.json']
    rows=np.array([pack(x) for x in items]);assert rows.shape==(7322,36) and len(refs)==7322 and len(sections)==110
    assert np.isfinite(rows).all() and np.all(rows[:,3:6]>0) and np.all((rows[:,35]>=0)&(rows[:,35]<1))
    cpu=read(R/'experiments/guarded_precision_v2/root_rows.json');cpu={r['name']:r for r in cpu['old']+cpu['fresh']}
    truthfile=R/'results/guarded_precision_audit_v1.json';truth={r['name']:r for r in read(truthfile)['rows']};files.append(truthfile)
    endpointrows=[]
    for s in sections:
        if s['expected']!='finite':continue
        for side in ('left','right'):
            endpointrows.append(dict(name=s['name'],side=side,start=s['start'],count=s['end']-s['start'],t=cpu[s['name']]['certificate']['lo' if side=='left' else 'hi'],H_MP240=truth[s['name']][side+'_H']))
    assert len(endpointrows)==212
    return rows,refs,sections,localgroups,endpointrows,files


def event_time(fn):
    fn();torch.cuda.synchronize();times=[]
    for _ in range(5):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();times.append(a.elapsed_time(b))
    return times


def run(out,state,rows,refs,sections,localgroups,endpointrows):
    start=time.perf_counter();module=load(out);src=torch.tensor(rows,dtype=torch.float64,device='cuda')
    sec=torch.tensor([[s['start'],s['end']-s['start']] for s in sections],dtype=torch.int32,device='cuda')
    esec=torch.tensor([[s['start'],s['count']] for s in endpointrows],dtype=torch.int32,device='cuda');ts=torch.tensor([s['t'] for s in endpointrows],dtype=torch.float64,device='cuda')
    packets=module.preprocess(src);mass=module.classify(packets,sec);ends=module.check(packets,esec,ts);torch.cuda.synchronize()
    p=packets.cpu().numpy();m=mass.cpu().numpy();e=ends.cpu().numpy();np.savez_compressed(out/'outputs.npz',packets=p,mass=m,endpoints=e)
    bad=[];summaries=[];mp.mp.dps=160
    for g in localgroups:
        local=[]
        for i in range(g['start'],g['end']):
            for mode in range(2):
                if p[i,mode,0]!=0:local.append(dict(index=i,mode=mode,status=finite(p[i,mode,0])));continue
                for field,a,b in (('peak',1,2),('rho',3,4),('energy',5,6),('alpha',24,25)):
                    value=mp.mpf(refs[i][field]);lo=mp.mpf(float(p[i,mode,a]));hi=mp.mpf(float(p[i,mode,b]));allow=mp.mpf('1e-75')*max(1,abs(value))
                    if not lo-allow<=value<=hi+allow:local.append(dict(index=i,mode=mode,field=field,value=refs[i][field],lo=finite(p[i,mode,a]),hi=finite(p[i,mode,b])))
        bad+=local;summaries.append(dict(group=g['name'],inputs=g['end']-g['start'],failure_records=len(local),failures=local))
        print(f'{g["name"]}: {g["end"]-g["start"]} inputs; failures={len(local)}',flush=True)
    classrows=[];classbad=[];codes={'finite':1,'critical_no_finite_root':2,'insufficient_mass_no_finite_root':3}
    for i,s in enumerate(sections):
        passes=[bool(m[i,j,0]==codes[s['expected']]) for j in range(2)]
        classrows.append(dict(name=s['name'],expected=s['expected'],outputs=[[finite(v) for v in m[i,j]] for j in range(2)],passes=passes))
        if not all(passes):classbad.append(s['name'])
    endrows=[]
    with localcontext() as ctx:
        ctx.prec=180;ctx.Emin=-999999999;ctx.Emax=999999999
        for i,s in enumerate(endpointrows):
            modechecks=[]
            for mode in range(2):
                v=e[i,mode];good=bool(v[3]==1 and np.isfinite(v).all());value=None
                if good:
                    value=Decimal(s['H_MP240'])*(-Decimal.from_float(float(v[2]))).exp();allow=abs(value)*Decimal('1e-65')
                    sign=v[1]<0 if s['side']=='left' else v[0]>0
                    good=bool(sign and Decimal.from_float(float(v[0]))-allow<=value<=Decimal.from_float(float(v[1]))+allow)
                modechecks.append(dict(pass_=good,output=list(map(finite,v)),reference_scaled=None if value is None else str(value)))
            endrows.append(dict(name=s['name'],side=s['side'],modes=modechecks))
    modelrows=[]
    for i in range(0,len(endrows),2):
        l,r=endrows[i:i+2];assert l['name']==r['name'];passes=[l['modes'][j]['pass_'] and r['modes'][j]['pass_'] for j in range(2)]
        modelrows.append(dict(name=l['name'],low_pass=passes[0],high_pass=passes[1],conditional_pass=any(passes),fallback=not passes[0]))
    times=dict(preprocess=event_time(lambda:module.preprocess(src)),mass=event_time(lambda:module.classify(packets,sec)),endpoint_both_modes=event_time(lambda:module.check(packets,esec,ts)))
    save(out/'local_rows.json',dict(groups=summaries));save(out/'class_rows.json',dict(rows=classrows));save(out/'endpoint_rows.json',dict(rows=endrows,models=modelrows));save(out/'timing.json',times)
    result=dict(status='complete',inputs=7322,coefficient_reference_checks=7322*2*4,local_failures=len(bad),local_gate_pass=not bad,mass_model_count=110,mass_mode_checks=220,mass_failure_names=classbad,mass_gate_pass=not classbad,
        endpoint_models=106,low_endpoint_certified=sum(r['low_pass'] for r in modelrows),high_endpoint_certified=sum(r['high_pass'] for r in modelrows),conditional_endpoint_failure_names=[r['name'] for r in modelrows if not r['conditional_pass']],conditional_endpoint_gate_pass=all(r['conditional_pass'] for r in modelrows),
        budget_mass_component_gate_pass=not bad and not classbad,whole_device_root_solver_complete=False,event_ms={k:float(np.median(v)) for k,v in times.items()},wall_seconds=time.perf_counter()-start,
        scope='Source-parameter budgets and mass fully on device; endpoints use prior CPU brackets. Dual-mode diagnostic outputs, not production cache or renderer/GT/backward/efficiency comparison.')
    save(out/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'},generated_sha256={p.name:sha(p) for p in (out/'generated').iterdir() if p.is_file()});save(out/'manifest.json',state);print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    rows,refs,sections,localgroups,endpointrows,files=inputs();sources=[Path(__file__),R/'guarded_preprocess.cuh',R/'guarded_preprocess_cuda.py',R/'guarded_interval.cuh',R/'guarded_interval_cuda.py',R/'audit_coherent_factor.py',R/'DEVICE_BUDGET_PROTOCOL.md',R/'exact_outlier_world_reference.py',R/'coherent_factor.py']+files
    out.mkdir(parents=True);np.savez_compressed(out/'inputs.npz',source=rows);save(out/'reference_and_sections.json',dict(reference=refs,sections=sections,groups=localgroups,endpoints=endpointrows))
    state=dict(status='running',created_utc=now(),pid=os.getpid(),torch=torch.__version__,numpy=np.__version__,cuda_build_flags=FLAGS,build_env={k:os.environ.get(k) for k in ('CUDA_HOME','CC','CXX','MAX_JOBS','TORCH_CUDA_ARCH_LIST')},source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources})
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in dict.fromkeys(sources):z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state,rows,refs,sections,localgroups,endpointrows)
    except BaseException as e:
        state.update(status='failed',failure_type=type(e).__name__,failure=str(e),failed_utc=now());save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
