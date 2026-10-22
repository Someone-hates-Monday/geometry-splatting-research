from pathlib import Path
import hashlib,json,os,time,traceback,zipfile
import numpy as np
import mpmath as mp
import torch
from research.history_gap_raw import torch_forward
from research.audit_history_gap_raw_micro import MPForward
R=Path(__file__).resolve().parent;ROOT=R.parent;SH=R/'experiments/history_gap_shared_v1';RAW=R/'experiments/history_gap_raw_v1';WORLD=R/'experiments/world_root_vjp_v1';OUT=R/'experiments/history_gap_shared_micro_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);torch.set_num_threads(1);files=[Path(__file__),R/'HISTORY_GAP_SHARED_MICRO_PROTOCOL.md',R/'audit_history_gap_raw_micro.py',R/'history_gap_raw.py',SH/'manifest.json',SH/'summary.json',SH/'shared_parameters.npz'];m=dict(status='running',pid=os.getpid(),created_epoch=time.time(),source_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files});save(OUT/'manifest.json',m)
    try:
        sh=read(SH/'summary.json');pred=-sh['scaled_gradient_norms']['full'];z=np.load(SH/'shared_parameters.npz');direction=z['full'];inverse=z['node_to_unique'];records=[];cursor=0;largest=[];oldfd=(np.load(SH/'full_plus_1e-05_loss.npy')-np.load(SH/'full_minus_1e-05_loss.npy'))/2e-5
        for f in sorted(RAW.glob('scan*_gradients.npz')):
            stem=f.name.removesuffix('_gradients.npz');wf=WORLD/f.name;meta=WORLD/(stem+'.json');rf=RAW/(stem+'.json')
            for p in [f,wf,meta,rf]:m['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
            with np.load(f) as d:sections=d['sections'];g=d['factor'];ref=d['reference']
            with np.load(wf) as d:raw=d['raw'];source=d['source']
            metadata=read(meta)['meta'];rows=read(rf)['rows']
            for j,row in enumerate(rows):
                a,b=map(int,sections[j:j+2]);delta=direction[inverse[cursor+a:cursor+b]];predicted=float(np.sum(g[a:b]*delta));index=len(records);record=dict(view=stem,pixel=row['pixel'],raw=raw[a:b],source=source[a:b],meta=metadata,direction=delta,reference=ref[a:b],reference_root=row['reference_root']);records.append(record);largest.append(dict(index=index,view=stem,pixel=row['pixel'],predicted=predicted,finite_difference=float(oldfd[index]),difference=float(oldfd[index]-predicted)))
            cursor+=len(raw)
        assert len(records)==891 and abs(sum(r['predicted'] for r in largest)/891-pred)<1e-13;largest.sort(key=lambda r:abs(r['difference']),reverse=True);assert largest[0]['view']=='scan37_0034' and largest[0]['pixel']==214675;save(OUT/'old_error_decomposition.json',dict(rows=largest))
        result=[];npz={}
        for h in [1e-7,1e-9,1e-11]:
            arrays=[]
            for sign in [-1,1]:
                losses=np.full(891,np.nan);statuses=[]
                for j,row in enumerate(records):
                    loss,info=torch_forward(row['raw'][:,:11]+sign*h*row['direction'],row['raw'],row['source'],row['meta']);statuses.append(info)
                    if loss is not None:losses[j]=loss
                    if (j+1)%128==0 or j==890:save(OUT/'progress.json',dict(h=h,sign=sign,rays=j+1,pid=os.getpid()));print(f'shared micro h={h} sign={sign}: {j+1}/891',flush=True)
                npz[f'{h}_{sign}']=losses;arrays.append(losses);save(OUT/f'roots_{h}_{sign}.json',dict(rows=statuses))
            finite=all(np.isfinite(a).all() for a in arrays);fd=float((arrays[1].mean()-arrays[0].mean())/(2*h)) if finite else None;result.append(dict(h=h,predicted=pred,derivative=fd,relative_error=abs(fd-pred)/abs(pred) if finite else None,gate_pass=finite and abs(fd-pred)<=max(1e-9,1e-5*abs(pred))))
        mp.mp.dps=240;row=records[largest[0]['index']];ctx=MPForward(row['raw'],row['source']);root=mp.mpf(row['reference_root']);baseline,t,caps=ctx.evaluate(row['direction'],mp.mpf(0),root);mp_pred=mp.fsum(mp.mpf(float(g))*mp.mpf(float(d)) for rr,dd in zip(row['reference'],row['direction']) for g,d in zip(rr,dd));mp_rows=[]
        for hh in ['1e-7','1e-9','1e-11']:
            h=mp.mpf(hh);minus,tm,cm=ctx.evaluate(row['direction'],-h,t);plus,tp,cp=ctx.evaluate(row['direction'],h,t);fd=(plus-minus)/(2*h);mp_rows.append(dict(h=hh,derivative=mp.nstr(fd,45),predicted=mp.nstr(mp_pred,45),relative_error=mp.nstr(abs(fd-mp_pred)/abs(mp_pred),30),cap_changed=cm!=caps or cp!=caps,gate_pass=abs(fd-mp_pred)<=max(mp.mpf('1e-9'),mp.mpf('1e-5')*abs(mp_pred))))
        np.savez_compressed(OUT/'losses.npz',**npz);summary=dict(status='complete',original_shared_gate_pass=False,original_gate_verdict_preserved=True,float64_all_ray_micro=result,posthoc_dominant_case=dict(view=row['view'],pixel=row['pixel'],mp_rows=mp_rows),scope='Additional micro-step diagnosis, original gate unchanged. Selected MP point is not full891 MP verification or training evidence.');save(OUT/'summary.json',summary)
        for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
        with zipfile.ZipFile(OUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for n in m['source_sha256']:archive.write(ROOT/n,n)
        m.update(status='complete',artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(OUT/'manifest.json',m);print(json.dumps(summary),flush=True)
    except BaseException as exc:m.update(status='failed',failure=repr(exc));save(OUT/'manifest.json',m);(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
