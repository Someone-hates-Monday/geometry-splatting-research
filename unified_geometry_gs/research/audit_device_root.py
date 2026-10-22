"""Device generates and certifies all fixed world-model roots; no CPU brackets input."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction
import argparse,hashlib,json,os,time,traceback,zipfile
import numpy as np
import mpmath as mp
import torch
from research.guarded_root_cuda import load
from research.guarded_interval_cuda import FLAGS

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def finite(v):return float(v) if np.isfinite(v) else str(float(v))
def timing(fn):
    fn();torch.cuda.synchronize();values=[]
    for _ in range(5):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();values.append(a.elapsed_time(b))
    return values


def run(out,state,source,sections,targets):
    start=time.perf_counter();module=load(out);src=torch.tensor(source,dtype=torch.float64,device='cuda');sec=torch.tensor([[s['start'],s['end']-s['start']] for s in sections],dtype=torch.int32,device='cuda')
    assert Fraction(float.fromhex('0x1.0624dd2f1a9fbp-10'))<Fraction(1,1000)
    packets=module.preprocess(src);answer=module.solve(packets,sec);torch.cuda.synchronize();p=packets.cpu().numpy();a=answer.cpu().numpy();np.savez_compressed(out/'outputs.npz',packets=p,roots=a)
    mp.mp.dps=160;rows=[];errors=[];codes={'finite':1,'critical_no_finite_root':2,'insufficient_mass_no_finite_root':3}
    for i,s in enumerate(sections):
        v=a[i];classification=bool(v[0]==codes[s['expected']]);cert=False;contains=False;err=None;width=None;target=targets[s['name']]
        if s['expected']=='finite' and v[0]==1 and np.isfinite(v).all():
            z,lo,hi=map(Fraction,v[2:5]);tau=Fraction(1,1000);width=hi-lo
            cert=bool(width>0 and width<=2*tau and z-lo<=tau and hi-z<=tau and v[6]<0 and v[7]>0)
            t=mp.mpf(target['camera_ray_depth']);contains=bool(mp.mpf(float(v[3]))<=t<=mp.mpf(float(v[4])));err=abs(mp.mpf(float(v[2]))-t)
            passed=classification and cert and contains and err<=mp.mpf('1e-3')
        else:passed=classification and s['expected']!='finite'
        row=dict(name=s['name'],expected=s['expected'],output=list(map(finite,v)),classification_pass=classification,strict_sign_and_exact_width_pass=cert,reference_in_bracket=contains,
            abs_error=None if err is None else mp.nstr(err,100),width_numerator=None if width is None else str(width.numerator),width_denominator=None if width is None else str(width.denominator),reference=target,gate_pass=bool(passed))
        rows.append(row)
        if not passed:errors.append(s['name'])
    times=dict(preprocess=timing(lambda:module.preprocess(src)),root_with_certificate=timing(lambda:module.solve(packets,sec)),combined=timing(lambda:module.solve(module.preprocess(src),sec)))
    save(out/'root_rows.json',dict(rows=rows));save(out/'timing.json',times)
    result=dict(status='complete',models=len(rows),finite_models=sum(s['expected']=='finite' for s in sections),no_root_models=sum(s['expected']!='finite' for s in sections),failure_names=errors,
        fixed_device_root_gate_pass=not errors,low_certified_finite=sum(r['gate_pass'] and r['expected']=='finite' and r['output'][1]==0 for r in rows),high_certified_finite=sum(r['gate_pass'] and r['expected']=='finite' and r['output'][1]==1 for r in rows),
        max_certified_abs_error=mp.nstr(max([mp.mpf(r['abs_error']) for r in rows if r['gate_pass'] and r['abs_error'] is not None],default=mp.mpf(0)),100),event_ms={k:float(np.median(v)) for k,v in times.items()},wall_seconds=time.perf_counter()-start,
        scope='All fixed mathematical ray models from raw saved-activated source parameters, GPU budgets/mass/candidates/certificates. Not renderer prefix/initial-bracket/backward/GT or end-to-end efficiency validation.')
    save(out/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'},generated_sha256={p.name:sha(p) for p in (out/'generated').iterdir() if p.is_file()});save(out/'manifest.json',state);print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    old=R/'experiments/device_budget_v1';m=read(old/'manifest.json');assert m['status']=='complete';assert read(old/'summary.json')['budget_mass_component_gate_pass']
    inputs=old/'inputs.npz';meta=old/'reference_and_sections.json'
    for f in (inputs,meta):assert sha(f)==m['artifacts_sha256'][f.name]
    with np.load(inputs) as d:source=d['source'].copy()
    sections=read(meta)['sections'];assert source.shape==(7322,36) and len(sections)==110
    t1=R/'experiments/coherent_factor_v2/summary.json';t2=R/'experiments/guarded_precision_v2/root_rows.json';targets={r['section']['name']:r['reference_root'] for r in read(t1)['root_rows']}
    targets.update({r['name']:r['reference'] for r in read(t2)['fresh']})
    sources=[Path(__file__),R/'guarded_root.cuh',R/'guarded_root_cuda.py',R/'guarded_preprocess.cuh',R/'guarded_interval.cuh',R/'guarded_interval_cuda.py',R/'DEVICE_ROOT_PROTOCOL.md',R/'DEVICE_BUDGET_PROTOCOL.md',inputs,meta,old/'manifest.json',old/'summary.json',t1,t2]
    out.mkdir(parents=True);np.savez_compressed(out/'inputs.npz',source=source,sections=np.array([[s['start'],s['end']-s['start']] for s in sections],np.int32));save(out/'targets_posthoc_only.json',dict(sections=sections,targets=targets))
    state=dict(status='running',created_utc=now(),pid=os.getpid(),torch=torch.__version__,cuda_build_flags=FLAGS,build_env={k:os.environ.get(k) for k in ('CUDA_HOME','CC','CXX','MAX_JOBS','TORCH_CUDA_ARCH_LIST')},device_inputs='Source[7322,36] plus110 start/count; no CPU candidate/bracket/reference input to either device kernel.',source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources})
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state,source,sections,targets)
    except BaseException as e:
        state.update(status='failed',failure_type=type(e).__name__,failure=str(e),failed_utc=now());save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
