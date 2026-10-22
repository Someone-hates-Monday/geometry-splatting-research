"""Locked short certificate + strict fallback, scalar checks and paired timing."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import localcontext, Decimal
from fractions import Fraction
import argparse, hashlib, json, math, os, time, traceback, zipfile
import numpy as np
import mpmath as mp
import torch
from research.short_certificate_cuda import load
from research.guarded_interval_cuda import FLAGS
from research.audit_strict_interval_host import scalar_truth

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def same(a,b):return bool(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint64),b.view(np.uint64)))
def finite(x):return float(x) if np.isfinite(x) else str(float(x))
def event(fn):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);start=time.perf_counter();a.record();fn();b.record();b.synchronize()
    return dict(event_ms=a.elapsed_time(b),wall_ms=1000*(time.perf_counter()-start))

def run(out,state,old,source,sections,scalar_rows,meta):
    start=time.perf_counter();control=load(out,old,False);variant=load(out,old,True)
    src=torch.tensor(source,dtype=torch.float64,device='cuda');sec=torch.tensor(sections,dtype=torch.int32,device='cuda')
    inputs=torch.tensor([[r['op'],r['x']] for r in scalar_rows],dtype=torch.float64,device='cuda')
    cp=control.preprocess(src);vp=variant.preprocess(src);cr=control.solve(cp,sec);vr=variant.solve(vp,sec);sc=variant.scalars(inputs);torch.cuda.synchronize()
    ca,va,cra,vra,sa=[a.cpu().numpy() for a in (cp,vp,cr,vr,sc)]
    with np.load(old/'outputs.npz') as d:priorp=d['variant_packets'];priorr=d['variant_roots']
    bitchecks=dict(packets_control_variant=same(ca,va),control_packets_prior=same(ca,priorp),control_roots_prior=same(cra,priorr))
    np.savez_compressed(out/'outputs.npz',control_packets=ca,variant_packets=va,control_roots=cra,variant_roots=vra,scalar_intervals=sa)
    scalar_checks=[];scalar_bad=[]
    with localcontext() as ctx:
        ctx.prec=240
        for i,(entry,bounds) in enumerate(zip(scalar_rows,sa)):
            truth=scalar_truth(entry['op'],entry['x']);lo,hi=[Decimal.from_float(float(x)) for x in bounds]
            okay=bool(lo.is_finite() and hi.is_finite() and lo<=truth<=hi)
            scalar_checks.append(dict(index=i,op=entry['op'],x_hex=entry['x'].hex(),lo_hex=float(bounds[0]).hex(),hi_hex=float(bounds[1]).hex(),decimal240_value=str(truth),gate_pass=okay))
            if not okay:scalar_bad.append(i)
    save(out/'scalar_checks.json',dict(decimal_precision=240,rows=scalar_checks,failures=scalar_bad))
    mp.mp.dps=160;rows=[];bad=[];code={'finite':1,'critical_no_finite_root':2,'insufficient_mass_no_finite_root':3};tau=Fraction(1,1000)
    for i,label in enumerate(meta['sections']):
        row=vra[i];classification=bool(row[0]==code[label['expected']]);passcert=False;contains=False;error=None
        if label['expected']=='finite' and row[0]==1 and np.isfinite(row).all():
            z,lo,hi=map(Fraction,row[2:5]);passcert=bool(lo<z<hi and hi-lo<=2*tau and z-lo<=tau and hi-z<=tau and row[6]<0<row[7])
            target=mp.mpf(meta['targets'][label['name']]['camera_ray_depth']);contains=bool(mp.mpf(float(row[3]))<=target<=mp.mpf(float(row[4])))
            error=abs(mp.mpf(float(row[2]))-target);okay=classification and passcert and contains and error<=mp.mpf('1e-3')
        else:okay=classification and label['expected']!='finite' and np.isnan(row[2:12]).all() and row[12]==row[13]==0
        record=dict(name=label['name'],expected=label['expected'],output=list(map(finite,row)),classification_pass=classification,
                    exact_width_and_strict_sign_pass=passcert,reference_contains=contains,abs_error=None if error is None else mp.nstr(error,100),gate_pass=bool(okay))
        rows.append(record)
        if not okay:bad.append(label['name'])
    save(out/'root_rows.json',dict(rows=rows))
    fns={'control':dict(preprocess=lambda:control.preprocess(src),root=lambda:control.solve(cp,sec),combined=lambda:control.solve(control.preprocess(src),sec)),
         'variant':dict(preprocess=lambda:variant.preprocess(src),root=lambda:variant.solve(vp,sec),combined=lambda:variant.solve(variant.preprocess(src),sec))}
    for arm in fns:
        for fn in fns[arm].values():fn()
    torch.cuda.synchronize();pairs=[]
    for i in range(5):
        p=dict(pair=i,order=['control','variant'] if i%2==0 else ['variant','control'],results={})
        for arm in p['order']:p['results'][arm]={k:event(fn) for k,fn in fns[arm].items()}
        p['ratios']={k:p['results']['variant'][k]['event_ms']/p['results']['control'][k]['event_ms'] for k in fns['control']}
        pairs.append(p);save(out/'timing_progress.json',dict(completed_pairs=i+1,rows=pairs));print(f'pair{i}: combined ratio={p["ratios"]["combined"]:.6f}',flush=True)
    ratios={k:math.exp(sum(math.log(p['ratios'][k]) for p in pairs)/5) for k in fns['control']}
    medians={arm:{k:float(np.median([p['results'][arm][k]['event_ms'] for p in pairs])) for k in fns[arm]} for arm in fns}
    finite_mask=vra[:,0]==1
    result=dict(status='complete',source_rows=7322,models=110,bitwise_checks=bitchecks,scalar_inputs=774,scalar_failure_indices=scalar_bad,root_failure_names=bad,
        finite_models=int(finite_mask.sum()),no_root_models=int((~finite_mask).sum()),
        short_certified=int((vra[finite_mask,12]==1).sum()),strict_certified=int((vra[finite_mask,12]==2).sum()),
        strict_attempts=int(vra[:,13].sum()),finite_low=int((vra[finite_mask,1]==0).sum()),finite_high=int((vra[finite_mask,1]==1).sum()),
        max_certified_abs_error=mp.nstr(max([mp.mpf(r['abs_error']) for r in rows if r['gate_pass'] and r['abs_error'] is not None],default=mp.mpf(0)),100),
        paired_event_ratio_geometric_mean=ratios,median_event_ms=medians,preset_combined_ratio_gate=.95,
        development_numerical_timing_gate_pass=all(bitchecks.values()) and not scalar_bad and not bad and ratios['combined']<=.95,
        independent_MP240_world_audit='pending separate verifier; no adoption claim before it passes',wall_seconds=time.perf_counter()-start,
        scope='Fixed observed mathematical ray development. Includes output allocation/fill, excludes transfer/full renderer/backward/GT/training; not novel standard series bounds.')
    save(out/'paired_timing.json',dict(rows=pairs));save(out/'summary.json',result)
    for name,digest in state['source_sha256'].items():assert sha(ROOT/name)==digest,name
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','timing_progress.json')},
                 generated_sha256={f'{d}/{p.name}':sha(p) for d in ('control_generated','variant_generated') for p in (out/d).iterdir() if p.is_file()})
    save(out/'manifest.json',state);print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);out=p.parse_args().out.resolve()
    if out.exists():raise FileExistsError(out)
    old=R/'experiments/reciprocal_root_v2';prior=read(old/'manifest.json');assert prior['status']=='complete' and read(old/'summary.json')['local_optimization_gate_pass']
    for name,digest in prior['artifacts_sha256'].items():assert sha(old/name)==digest
    names=['root.cu','binding.cpp','guarded_interval.cuh','guarded_preprocess.cuh','guarded_root.cuh']
    for name in names:assert sha(old/'variant_generated'/name)==prior['generated_sha256']['variant_generated/'+name]
    with np.load(old/'inputs.npz') as d:source=d['source'].copy();sections=d['sections'].copy()
    scalar_path=R/'experiments/strict_interval_host_v1/scalar_inputs.json';scalar_rows=read(scalar_path)['rows'];assert len(scalar_rows)==774
    assert sha(scalar_path)==read(scalar_path.with_name('manifest.json'))['artifacts_sha256'][scalar_path.name]
    meta_path=R/'experiments/device_root_v1/targets_posthoc_only.json';meta=read(meta_path)
    sources=[Path(__file__),R/'short_certificate_cuda.py',R/'SHORT_CERTIFICATE_PROTOCOL.md',R/'audit_strict_interval_host.py',R/'guarded_interval_cuda.py',old/'manifest.json',old/'summary.json',old/'inputs.npz',old/'outputs.npz',scalar_path,meta_path]+[old/'variant_generated'/n for n in names]
    out.mkdir(parents=True);np.savez_compressed(out/'inputs.npz',source=source,sections=sections,scalar_inputs=np.array([[r['op'],r['x']] for r in scalar_rows],np.float64))
    save(out/'targets_posthoc_only.json',meta)
    state=dict(status='running',created_utc=now(),pid=os.getpid(),torch=torch.__version__,cuda_flags=FLAGS,
        build_env={k:os.environ.get(k) for k in ('CUDA_HOME','CC','CXX','MAX_JOBS','TORCH_CUDA_ARCH_LIST')},
        source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources},device_inputs='Source[7322,36], sections[110,2]; scalar op/value rows774. No CPU root/bracket/reference input.')
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state,old,source,sections,scalar_rows,meta)
    except BaseException as e:
        state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
