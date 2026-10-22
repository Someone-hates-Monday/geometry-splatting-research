"""All 212 fixed endpoint checks on device; not a whole GPU solver or renderer."""
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,localcontext
import argparse,hashlib,json,os,time,traceback,zipfile
import numpy as np
import torch
from research.guarded_interval_cuda import load,FLAGS

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def dec(x):return Decimal.from_float(float(x))
def finite(x):return float(x) if np.isfinite(x) else str(float(x))

def run(out,state,rows):
    start=time.perf_counter();module=load(out);device='cuda';layers=[];sections=[];endpoints=[]
    for r in rows:sections.append([len(layers),len(r['layers'])]);layers+=r['layers'];endpoints.append(r['x'])
    assert max(s[1] for s in sections)<=192 and len(rows)==212
    a=torch.tensor(layers,dtype=torch.float64,device=device);b=torch.tensor(sections,dtype=torch.int32,device=device);c=torch.tensor(endpoints,dtype=torch.float64,device=device)
    np.savez_compressed(out/'inputs.npz',layers=np.asarray(layers),sections=np.asarray(sections,np.int32),endpoints=np.asarray(endpoints))
    answer=module.forward(a,b,c);torch.cuda.synchronize();values=answer.cpu().numpy();np.save(out/'outputs.npy',values)
    details=[];bad=[]
    with localcontext() as ctx:
        ctx.prec=180;ctx.Emax=999999999;ctx.Emin=-999999999
        for i,(r,v) in enumerate(zip(rows,values)):
            truth=Decimal(r['independent_MP240_H'])*(-dec(v[2])).exp() if np.isfinite(v[2]) else Decimal('NaN');allow=abs(truth)*Decimal('1e-65')
            sign=v[1]<0 if r['side']=='left' else v[0]>0
            passed=bool(v[3]==1 and np.isfinite(v).all() and sign and dec(v[0])-allow<=truth<=dec(v[1])+allow)
            details.append(dict(name=r['name'],side=r['side'],output=list(map(finite,v)),reference_scaled=str(truth),gate_pass=passed))
            if not passed:bad.append(dict(index=i,name=r['name'],side=r['side']))
    save(out/'endpoint_rows.json',dict(rows=details))
    # Allocations/fill and endpoint kernel are included; source budgets/transfer/roots are excluded.
    for _ in range(5):module.forward(a,b,c)
    torch.cuda.synchronize();times=[]
    for _ in range(5):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
        for _ in range(5):module.forward(a,b,c)
        end.record();end.synchronize();times.append(begin.elapsed_time(end)/5)
    result=dict(status='complete',endpoint_checks=212,endpoint_failures=len(bad),endpoint_component_gate_pass=not bad,failure_rows=bad,
        event_ms_for_212_endpoints=times,median_event_ms=float(np.median(times)),wall_seconds=time.perf_counter()-start,
        scope='Device endpoint arithmetic on saved CPU parameter intervals and CPU-selected brackets. Not device budget construction, selection, root solver, backward, full image, GT or efficiency evidence.',
        standard_exp_log_library_used=False,device=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
    save(out/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'},generated_sha256={p.name:sha(p) for p in (out/'generated').iterdir() if p.is_file()})
    save(out/'manifest.json',state);print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    source=R/'experiments/strict_interval_host_v1';m=read(source/'manifest.json');assert m['status']=='complete'
    payload=source/'endpoint_inputs.json';assert sha(payload)==m['artifacts_sha256'][payload.name];rows=read(payload)['rows']
    sources=[Path(__file__),R/'guarded_interval_cuda.py',R/'guarded_interval.cuh',R/'GUARDED_CUDA_PROTOCOL.md',R/'STRICT_INTERVAL_PROTOCOL.md',payload,source/'manifest.json',R/'results/guarded_precision_audit_v1.json']
    out.mkdir(parents=True);state=dict(status='running',created_utc=now(),pid=os.getpid(),torch=torch.__version__,numpy=np.__version__,cuda_build_flags=FLAGS,build_env={k:os.environ.get(k) for k in ('CUDA_HOME','CC','CXX','MAX_JOBS','TORCH_CUDA_ARCH_LIST')},source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources})
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state,rows)
    except BaseException as e:
        state.update(status='failed',failure_type=type(e).__name__,failure=str(e),failed_utc=now());save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
