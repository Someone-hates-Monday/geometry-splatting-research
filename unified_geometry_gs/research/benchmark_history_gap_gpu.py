from pathlib import Path
import argparse,hashlib,json,os,time,traceback,zipfile
import numpy as np
import torch
from research.history_gap_gpu import Engine,build,metadata_array
R=Path(__file__).resolve().parent;ROOT=R.parent;RAW=R/'experiments/history_gap_raw_v1';WORLD=R/'experiments/world_root_vjp_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def errors(a,b):
    if not np.isfinite(a).all():return dict(relative_l2=None,blocks=None,gate_pass=False)
    rel=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300));blocks={k:float(np.linalg.norm(a[:,i:j]-b[:,i:j])/max(np.linalg.norm(b[:,i:j]),1e-30)) for k,i,j in [('xyz',0,3),('log_scale',3,6),('quaternion',6,10),('opacity_logit',10,11)]}
    return dict(relative_l2=rel,blocks=blocks,gate_pass=rel<=1e-5 and max(blocks.values())<=1e-4)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);out=p.parse_args().out.resolve();assert not out.exists();out.mkdir(parents=True)
    files=[Path(__file__),R/'history_gap_gpu.py',R/'history_gap_gpu.cu',R/'HISTORY_GAP_GPU_PROTOCOL.md',RAW/'manifest.json',WORLD/'manifest.json'];m=dict(status='running',pid=os.getpid(),created_epoch=time.time(),source_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files});save(out/'manifest.json',m)
    try:
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rawm=read(RAW/'manifest.json');wm=read(WORLD/'manifest.json');raws=[];sources=[];refs=[];metas=[];sections=[0];records=[]
        for f in sorted(RAW.glob('scan*_gradients.npz')):
            stem=f.name.removesuffix('_gradients.npz');wf=WORLD/f.name;meta=WORLD/(stem+'.json');rowpath=RAW/(stem+'.json')
            for q,mf in [(f,rawm),(rowpath,rawm),(wf,wm),(meta,wm)]:assert sha(q)==mf['artifacts_sha256'][q.name];m['source_sha256'][q.relative_to(ROOT).as_posix()]=sha(q)
            with np.load(f) as d:refs.append(d['reference']);ss=d['sections'];pixels=d['pixels']
            with np.load(wf) as d:raws.append(d['raw']);sources.append(d['source'])
            rows=read(rowpath)['rows'];metadata=read(meta)['meta']
            for j,row in enumerate(rows):
                sections.append(sections[-1]+int(ss[j+1]-ss[j]));metas.append(metadata_array(metadata));records.append(dict(view=stem,pixel=int(pixels[j]),old_anomaly=row['old_anomaly'],reference_loss=row['reference_loss'],reference_root=row['reference_root']))
        raw=np.concatenate(raws);source=np.concatenate(sources);reference=np.concatenate(refs);sections=np.array(sections,dtype=np.int64);assert raw.shape==(51524,12) and len(records)==891
        start=time.perf_counter();native=build(out/'generated');compile_wall=time.perf_counter()-start;save(out/'progress.json',dict(phase='compiled',compile_wall=compile_wall));print('compiled CUDA',flush=True)
        start=time.perf_counter();engine=Engine(raw,source,sections,np.array(metas),native);torch.cuda.synchronize();prepare_wall=time.perf_counter()-start
        observations=[]
        for repeat in range(4):
            theta=engine.base.detach().clone().requires_grad_(True);start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True);start_event.record();loss,info=engine.loss(theta);loss.sum().backward();end_event.record();torch.cuda.synchronize();observations.append(float(start_event.elapsed_time(end_event)));print(f'GPU full raw repeat{repeat}: {observations[-1]:.3f}ms',flush=True)
        actual=theta.grad.detach().cpu().numpy();values=loss.detach().cpu().numpy();debug=info.cpu().numpy();rays=[]
        for j,row in enumerate(records):
            a,b=map(int,sections[j:j+2]);metric=errors(actual[a:b],reference[a:b]);finite=bool(np.isfinite(values[j]));loss_error=abs(float(values[j])-row['reference_loss']) if finite else None;status=int(debug[j,0]);rays.append(dict(**row,status=status,loss=float(values[j]) if finite else None,loss_error=loss_error,gradient=metric,gate_pass=metric['gate_pass'] and finite and loss_error<=1e-9 and status>0))
        np.savez_compressed(out/'real_arrays.npz',sections=sections,reference=reference,actual=actual,loss=values,info=debug)
        result=dict(status='complete',rays=891,nodes=51524,pass_count=sum(r['gate_pass'] for r in rays),status_counts={str(int(v)):int(np.count_nonzero(debug[:,0]==v)) for v in np.unique(debug[:,0])},max_relative_l2=max((r['gradient']['relative_l2'] for r in rays if r['gradient']['relative_l2'] is not None),default=None),compile_wall_seconds=compile_wall,prepare_and_transfer_wall_seconds=prepare_wall,full_geometry_loss_raw_backward_ms=observations,timed_median_ms=float(np.median(observations[1:])),peak_allocated_bytes=torch.cuda.max_memory_allocated(),rows=rays,runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),numpy=np.__version__),scope='GPU Float64 numerical loss + complete captured-source raw chain; not certified root signs, training speed, GT or independent confirmation.')
        save(out/'summary.json',result)
        for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
        with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name in m['source_sha256']:z.write(ROOT/name,name)
        binaries=list((out/'generated').glob('*.so'));assert len(binaries)==1;m.update(status='complete',completed_epoch=time.time(),binary=dict(path=binaries[0].relative_to(ROOT).as_posix(),sha256=sha(binaries[0])),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(out/'manifest.json',m);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    except BaseException as exc:
        m.update(status='failed',failure=repr(exc));save(out/'manifest.json',m);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        with zipfile.ZipFile(out/'failure_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name in m['source_sha256']:z.write(ROOT/name,name)
        raise
if __name__=='__main__':main()
