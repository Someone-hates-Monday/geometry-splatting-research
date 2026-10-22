"""Fixed raw-parameter audit and per-ray descent probes; no training."""
from pathlib import Path
import argparse,hashlib,json,os,time,traceback,zipfile
import numpy as np
import mpmath as mp
import torch
from research.history_gap_math import mp_root
from research.world_root_vjp import mp_geometry
from research.history_gap_raw import mp_raw_gradient,torch_gap_gradient,torch_forward,parameter_scale

R=Path(__file__).resolve().parent;ROOT=R.parent;BASE=R/'experiments/world_root_vjp_v1';COEF=R/'experiments/history_gap_math_v3'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def metrics(a,b):
    if not np.isfinite(a).all():return dict(relative_l2=None,blocks=None,gate_pass=False)
    relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300));blocks={}
    for name,i,j in [('xyz',0,3),('log_scale',3,6),('quaternion',6,10),('opacity_logit',10,11)]:blocks[name]=float(np.linalg.norm(a[:,i:j]-b[:,i:j])/max(np.linalg.norm(b[:,i:j]),1e-30))
    return dict(relative_l2=relative,blocks=blocks,gate_pass=relative<=1e-5 and max(blocks.values())<=1e-4)
def directional(raw,source,meta,gradient,scale,rng):
    weighted=scale*gradient;norm=float(np.linalg.norm(weighted));random=rng.normal(size=gradient.shape);random=scale*random/np.linalg.norm(random)
    directions=[('random',random)]
    if norm>0:directions.append(('negative_gradient',-scale*weighted/norm))
    rows=[]
    for name,d in directions:
        pred=float(np.sum(gradient*d));steps=[]
        for h in [1e-4,3e-5,1e-5]:
            minus,mi=torch_forward(raw[:,:11]-h*d,raw,source,meta);plus,pi=torch_forward(raw[:,:11]+h*d,raw,source,meta)
            if minus is None or plus is None:steps.append(dict(h=h,status='missing_root',minus=mi,plus=pi));continue
            fd=(plus-minus)/(2*h);steps.append(dict(h=h,status='finite',derivative=fd,absolute_error=abs(fd-pred),relative_error=abs(fd-pred)/max(abs(pred),1e-300)))
        last=steps[-1];passed=last['status']=='finite' and last['absolute_error']<=max(1e-9,1e-5*abs(pred))
        rows.append(dict(direction=name,predicted=pred,numerically_weak=abs(pred)<1e-10,steps=steps,gate_pass=passed))
    return rows
def descent(raw,source,meta,gradient,scale,baseline):
    norm=float(np.linalg.norm(scale*gradient));steps=[]
    if norm==0:return dict(scaled_gradient_norm=norm,status='zero_gradient',steps=[])
    direction=-scale*scale*gradient/norm
    for h in [1e-4,1e-3,1e-2]:
        loss,info=torch_forward(raw[:,:11]+h*direction,raw,source,meta)
        if loss is None:steps.append(dict(h=h,status='missing_root',**info));continue
        delta=loss-baseline;steps.append(dict(h=h,status='finite',loss=loss,delta=delta,relative_delta=delta/max(baseline,1e-300),armijo=loss<=baseline-1e-4*h*norm,measurable_decrease=delta< -max(1e-12,1e-8*baseline),**info))
    return dict(scaled_gradient_norm=norm,status='probed',steps=steps)
def synthetic():
    source=np.zeros((3,36));source[:,:3]=np.array([[.4,-.2,2],[.61,-.32,3],[.8,-.4,4]],np.float32).astype(float);source[:,3:6]=np.array([.08,.09,.11],np.float32).astype(float);source[:,6:10]=np.array([1.,.1,.05,-.02],np.float32);source[:,6:10]/=np.linalg.norm(source[:,6:10],axis=1)[:,None];source[:,10]=1;source[:,20:29]=np.eye(3).reshape(1,9);source[:,32:35]=[.2,-.1,1];source[:,35]=[.6,.35,.2]
    raw=np.zeros((3,12));raw[:,:3]=source[:,:3];raw[:,3:6]=np.log(source[:,3:6]);raw[:,6:10]=[1,.1,.05,-.02];raw[:,10]=[.8,-.4,-1.2];raw[:,11]=.006;rows=[];rng=np.random.default_rng(13017)
    for kernel in [0.,.3,1.]:
        meta=dict(fx=300.,fy=310.,limx=.8,limy=.6,kernel=kernel,floor=float(np.float32(1e-6)));loss,g,info=torch_gap_gradient(raw,source,meta,'factor');fd=directional(raw,source,meta,g,parameter_scale(raw,source),rng);rows.append(dict(kernel=kernel,synthetic=True,loss=loss,root=info['root'],directions=fd,gate_pass=all(v['gate_pass'] for v in fd)))
    return rows
def run(out,m):
    torch.set_num_threads(1);rng=np.random.default_rng(13017);oldm=read(BASE/'manifest.json');cm=read(COEF/'manifest.json');total=0;nodes_count=0;rows=[];selected=set(np.linspace(0,890,12,dtype=int).tolist());save(out/'synthetic.json',dict(rows=synthetic()))
    for f in sorted(BASE.glob('scan*_gradients.npz')):
        stem=f.name.removesuffix('_gradients.npz');meta_file=BASE/(stem+'.json');cof=COEF/(stem+'.json')
        for path,parent in [(f,oldm),(meta_file,oldm),(cof,cm)]:
            assert sha(path)==parent['artifacts_sha256'][path.name];m['source_sha256'][path.relative_to(ROOT).as_posix()]=sha(path)
        save(out/'manifest.json',m);v=read(meta_file);previous=read(cof)['rows'];meta=v['meta'];viewrows=[]
        with np.load(f) as d:source=d['source'];raw=d['raw'];sections=d['sections'];pixels=d['pixels'];ids=d['primitive_ids']
        arrays={k:np.full((len(source),11),np.nan) for k in ['reference','factor','dense']}
        for j,old in enumerate(previous):
            begin,end=map(int,sections[j:j+2]);src=source[begin:end];params=raw[begin:end];start=time.perf_counter();assert old['pixel']==int(pixels[j]);mp.mp.dps=140;geometry=[mp_geometry(row) for row in src];mp.mp.dps=240;t=mp_root([[g['alpha'],g['p'],g['rho']] for g in geometry],mp.mpf(old['root']));loss,reference,mi=mp_raw_gradient(geometry,src,params,meta,t);arrays['reference'][begin:end]=reference
            branches={};successful={}
            for arm in ['factor','dense']:
                try:
                    fl,g,info=torch_gap_gradient(params,src,meta,arm);arrays[arm][begin:end]=g;branches[arm]=dict(loss=fl,loss_error=abs(fl-float(loss)),**info,**metrics(g,reference));successful[arm]=(fl,g)
                except (ValueError,FloatingPointError,RuntimeError) as exc:branches[arm]=dict(status='failed',error=repr(exc),gate_pass=False)
            row=dict(index=total,view=stem,pixel=int(pixels[j]),nodes=end-begin,old_anomaly=old['old_anomaly'],reference_loss=float(loss),reference_root=mp.nstr(t,55),**mi,branches=branches)
            if 'factor' in successful:
                fl,g=successful['factor'];scale=parameter_scale(params,src);row['descent']=descent(params,src,meta,g,scale,fl)
                row['high_loss']=float(loss)>=.25;row['weak_gradient']=row['descent']['scaled_gradient_norm']<=1e-8
                if total in selected or old['old_anomaly'] is not None:row['directional']=directional(params,src,meta,g,scale,rng)
            else:row['descent']=dict(status='unavailable',steps=[]);row['high_loss']=float(loss)>=.25;row['weak_gradient']=None
            row['wall_seconds']=time.perf_counter()-start;rows.append(row);viewrows.append(row);total+=1;nodes_count+=end-begin
            if (j+1)%4==0 or j+1==len(previous):save(out/'progress.json',dict(status='running',pid=os.getpid(),rays=total,nodes=nodes_count,view=stem));print(f'raw {total}/891 nodes {nodes_count}, factor_pass={branches["factor"]["gate_pass"]}, loss={float(loss):.6g}',flush=True)
        np.savez_compressed(out/f.name,pixels=pixels,sections=sections,primitive_ids=ids,**arrays);save(out/(stem+'.json'),dict(meta=meta,rows=viewrows))
    assert total==891 and nodes_count==51524 and sum(r['old_anomaly'] is not None for r in rows)==9
    counts={k:sum(r['branches'][k]['gate_pass'] for r in rows) for k in ['factor','dense']};directions=[d for r in rows for d in r.get('directional',[])];steps={}
    for h in [1e-4,1e-3,1e-2]:
        values=[s for r in rows for s in r['descent']['steps'] if s['h']==h];finite=[s for s in values if s['status']=='finite'];steps[str(h)]=dict(all_ray_denominator=891,probed=len(values),finite=len(finite),missing_root=len(values)-len(finite),armijo=sum(s['armijo'] for s in finite),measurable_decrease=sum(s['measurable_decrease'] for s in finite),increased=sum(s['delta']>0 for s in finite),unchanged=sum(s['delta']==0 for s in finite))
    result=dict(status='complete',rays=891,nodes=51524,branch_pass_counts=counts,raw_gate_pass=all(v==891 for v in counts.values()),high_loss_count=sum(r['high_loss'] for r in rows),weak_gradient_count=sum(r['weak_gradient'] is True for r in rows),high_loss_weak_gradient_count=sum(r['high_loss'] and r['weak_gradient'] is True for r in rows),zero_gradient_count=sum(r['descent']['status']=='zero_gradient' for r in rows),directional_rays=sum('directional' in r for r in rows),direction_count=len(directions),directional_pass=sum(d['gate_pass'] for d in directions),directional_weak=sum(d['numerically_weak'] for d in directions),descent_steps=steps,rows=rows,scope='Fixed-source continuous raw chain and independent per-ray local descent; no shared-ID update, CUDA training, GT or novelty evidence.')
    save(out/'summary.json',result)
    for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in m['source_sha256']:z.write(ROOT/name,name)
    m.update(status='complete',completed_epoch=time.time(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(out/'manifest.json',m);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);out=p.parse_args().out.resolve();assert not out.exists();out.mkdir(parents=True)
    files=[Path(__file__),R/'history_gap_raw.py',R/'HISTORY_GAP_RAW_PROTOCOL.md',R/'history_gap_math.py',R/'world_root_vjp.py',R/'world_root_vjp_v2.py',BASE/'manifest.json',COEF/'manifest.json'];m=dict(status='running',created_epoch=time.time(),pid=os.getpid(),source_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files});save(out/'manifest.json',m)
    try:run(out,m)
    except BaseException as exc:m.update(status='failed',failure=repr(exc));save(out/'manifest.json',m);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
