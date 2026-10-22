from pathlib import Path
import argparse,hashlib,json,os,time,traceback,zipfile,importlib.util
import numpy as np
import torch
import mpmath as mp
from research.history_gap_warp import build
from research.history_gap_gpu_v2 import Engine
from research.history_gap_gpu import metadata_array
from research.benchmark_history_gap_gpu import errors
from research.history_gap_extension import mp_extended
from research.history_gap_math import mp_loss_gradient
R=Path(__file__).resolve().parent;ROOT=R.parent;GPU=R/'experiments/history_gap_gpu_v2';WORLD=R/'experiments/world_root_vjp_v1';RAW=R/'experiments/history_gap_raw_v1';CONF=R/'experiments/history_gap_gpu_confirm_v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load_binary(path):
    spec=importlib.util.spec_from_file_location(path.stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def time_full(engine,native):
    engine.native=native;theta=engine.base.detach().clone().requires_grad_(True);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();loss,info=engine.loss(theta);loss.sum().backward();b.record();torch.cuda.synchronize()
    return float(a.elapsed_time(b)),loss.detach().cpu().numpy(),theta.grad.cpu().numpy(),info.cpu().numpy()
def coefficient_checks(native,out):
    mp.mp.dps=240;cases=[('single_singular',[.75],[0.],[1.]),('common_singular',[.5,.5],[0.,0.],[1.,2.]),('missing',[.4],[0.],[1.]),('mass_boundary',[.5],[0.],[1.]),('before',[.9],[0.],[1.]),('strict_cap',[.6,.999],[-.2,.2],[5.,5.])]+[(f'thin_{r}',[.5,.5],[-1.,1.],[r,r]) for r in [5.,10.,1/.075,20.]]
    result=[]
    for name,amp,p,rho in cases+[(f'nodes{n}',[.001 if n==513 else .0004]*(n-1)+[.6 if n==513 else .7],[-1.]*(n-1)+[1.],[10.]*n) for n in [513,1025]]:
        alpha=[min(mp.mpf(float(np.float32(.99))),mp.mpf(float(a))) for a in amp];nodes=[[a,mp.mpf(float(pp)),mp.mpf(float(rr))] for a,pp,rr in zip(alpha,p,rho)]
        if len(nodes)>512:
            n=len(nodes);lo=mp.mpf(-2);hi=mp.mpf(2)
            def H(node,t):
                a,p,rho=node;ell=mp.log1p(-a*mp.exp(-((t-p)*rho)**2/2))/2
                return -mp.log1p(-a)+ell if t>p else -ell
            for _ in range(700):
                mid=(lo+hi)/2;v=(n-1)*H(nodes[0],mid)+H(nodes[-1],mid)-mp.log(2)
                if v<0:lo=mid
                else:hi=mid
                if hi-lo<mp.mpf('1e-170'):break
            reference_loss,g,_=mp_loss_gradient(nodes,(lo+hi)/2)
            assert mp.fprod(1-a for a,pp,rr in nodes)<mp.mpf('.5') and (1-alpha[0])**(n-1)>mp.mpf('.5')
        else:reference_loss,g,_=mp_extended(nodes)
        inputs=[torch.tensor(v,device='cuda',dtype=torch.float64) for v in [p,rho,np.zeros(len(p)),amp]];loss,seeds,info=native.solve(*inputs,torch.tensor([0,len(p)],device='cuda',dtype=torch.int64));actual=seeds.cpu().numpy();expected=[]
        for j,row in enumerate(g):
            logseed=row[0]*alpha[j];capped=amp[j]>float(np.float32(.99));expected.append([float(row[1]),float(row[2]),0. if capped else float(-logseed/2),0. if capped else float(logseed/mp.mpf(float(amp[j])) )])
        expected=np.array(expected);value=float(loss.cpu()[0]);finite=bool(np.isfinite(actual).all() and np.isfinite(value));rel=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-300)) if finite else None;error=abs(value-float(reference_loss)) if finite else None
        np.savez_compressed(out/(name+'_coefficient.npz'),amplitude=np.array(amp),peak=np.array(p),rho=np.array(rho),expected=expected,actual=actual,loss=loss.cpu().numpy(),reference_loss=np.array([float(reference_loss)]),info=info.cpu().numpy())
        result.append(dict(name=name,nodes=len(nodes),status=int(info.cpu()[0,0]),reported_nodes=int(info.cpu()[0,5]),relative_seed_error=rel,loss_error=error,gate_pass=bool(finite and rel<=1e-5 and error<=1e-9 and int(info.cpu()[0,5])==len(nodes))))
    invalid=native.solve(*[torch.tensor(v,device='cuda',dtype=torch.float64) for v in [[0.],[0.],[0.],[.5]]],torch.tensor([0,1],device='cuda',dtype=torch.int64));rejected=bool(torch.isnan(invalid[0]).all()) and int(invalid[2].cpu()[0,0])<0
    return result,rejected
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);out=parser.parse_args().out.resolve();assert not out.exists();out.mkdir(parents=True);files=[Path(__file__),R/'HISTORY_GAP_WARP_PROTOCOL.md',R/'history_gap_warp.cu',R/'history_gap_warp_body.inc',R/'history_gap_warp.py',R/'history_gap_gpu.py',R/'history_gap_gpu_v2.py',GPU/'manifest.json',WORLD/'manifest.json',RAW/'manifest.json',CONF/'manifest.json'];state=dict(status='running',created_epoch=time.time(),pid=os.getpid(),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files});save(out/'manifest.json',state)
    try:
        torch.set_num_threads(1);serial_m=read(GPU/'manifest.json');serial_path=ROOT/serial_m['binary']['path'];assert sha(serial_path)==serial_m['binary']['sha256'];serial=load_binary(serial_path);start=time.perf_counter();prior=R/'experiments/history_gap_warp_v1';prior_m=read(prior/'manifest.json');assert prior_m['status']=='failed' and sha(R/'history_gap_warp.cu')==prior_m['source_sha256']['research/history_gap_warp.cu'];prior_bins=list((prior/'generated').glob('*.so'));assert len(prior_bins)==1;warp=load_binary(prior_bins[0]);compile_wall=time.perf_counter()-start;state['reused_native']=dict(path=prior_bins[0].relative_to(ROOT).as_posix(),sha256=sha(prior_bins[0]),reason='Original compilation completed; failed runner summary only.');print('reused unchanged compiled warp CUDA',flush=True)
        raws=[];sources=[];refs=[];metadata=[];sections=[0];records=[];wm=read(WORLD/'manifest.json');rm=read(RAW/'manifest.json')
        for f in sorted(WORLD.glob('scan*_gradients.npz')):
            stem=f.name.removesuffix('_gradients.npz');rf=RAW/f.name;mf=WORLD/(stem+'.json');rowf=RAW/(stem+'.json')
            for p,m in [(f,wm),(mf,wm),(rf,rm),(rowf,rm)]:assert sha(p)==m['artifacts_sha256'][p.name];state['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
            with np.load(f) as d:raws.append(d['raw']);sources.append(d['source']);ss=d['sections']
            with np.load(rf) as d:refs.append(d['reference'])
            meta=read(mf)['meta']
            for j,row in enumerate(read(rowf)['rows']):sections.append(sections[-1]+int(ss[j+1]-ss[j]));metadata.append(metadata_array(meta));records.append(dict(view=stem,pixel=row['pixel'],old_anomaly=row['old_anomaly'],reference_loss=row['reference_loss']))
        raw=np.concatenate(raws);src=np.concatenate(sources);ref=np.concatenate(refs);sections=np.array(sections,dtype=np.int64);start=time.perf_counter();engine=Engine(raw,src,sections,np.array(metadata),serial);torch.cuda.synchronize();prepare_wall=time.perf_counter()-start
        timings=[];kernel_times=[];outputs={}
        for iteration in range(10):
            for arm in (['serial','warp'] if iteration%2==0 else ['warp','serial']):
                native=serial if arm=='serial' else warp;ms,loss,g,info=time_full(engine,native);timings.append(dict(iteration=iteration,arm=arm,warmup=iteration<2,ms=ms));outputs[arm]=(loss,g,info)
            save(out/'timings.json',dict(full=timings,kernel=kernel_times));print(f'paired iteration{iteration} full serial/warp done',flush=True)
        with torch.no_grad():fields=engine.fields(engine.base)
        for iteration in range(10):
            for arm in (['serial','warp'] if iteration%2==0 else ['warp','serial']):
                native=serial if arm=='serial' else warp;a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();native.solve(*fields,engine.sections);b.record();torch.cuda.synchronize();kernel_times.append(dict(iteration=iteration,arm=arm,warmup=iteration<2,ms=float(a.elapsed_time(b))));save(out/'timings.json',dict(full=timings,kernel=kernel_times))
        rayrows=[]
        for j,record in enumerate(records):
            a,b=map(int,sections[j:j+2]);branches={}
            for arm,(loss,g,info) in outputs.items():
                metric=errors(g[a:b],ref[a:b]);finite=bool(np.isfinite(loss[j]));error=abs(float(loss[j])-record['reference_loss']) if finite else None;branches[arm]={**metric,'loss_error':error,'status':int(info[j,0]),'gate_pass':bool(metric['gate_pass'] and finite and error<=1e-9 and info[j,0]>0)}
            rayrows.append(dict(**record,branches=branches))
        np.savez_compressed(out/'real_arrays.npz',sections=sections,reference=ref,**{f'{arm}_{key}':v for arm,data in outputs.items() for key,v in zip(['loss','gradient','info'],data)})
        cf=CONF/'new_arrays.npz';state['source_sha256'][cf.relative_to(ROOT).as_posix()]=sha(cf)
        with np.load(cf) as d:cr=d['raw'];cs=d['source'];ct=d['sections'];cm=d['metadata'];cref=d['reference']
        confirm_rows=read(CONF/'summary.json')['rows'];ce=Engine(cr,cs,ct,cm,warp);_,cl,cg,ci=time_full(ce,warp);newrows=[]
        for j,row in enumerate(confirm_rows):
            a,b=map(int,ct[j:j+2]);metric=errors(cg[a:b],cref[a:b]);error=abs(float(cl[j])-row['reference_loss']);newrows.append({**metric,'case':j,'new_confirmation':row['new_confirmation'],'loss_error':error,'gate_pass':bool(metric['gate_pass'] and error<=1e-9 and ci[j,0]>0)})
        np.savez_compressed(out/'confirm_arrays.npz',sections=ct,reference=cref,actual=cg,loss=cl,info=ci);special,invalid=coefficient_checks(warp,out)
        # Two retained real absent updates, independent MP directional reference.
        shared=R/'experiments/history_gap_shared_v1';wf=WORLD/'scan37_0020_gradients.npz';z=np.load(shared/'shared_parameters.npz');meta=read(WORLD/'scan37_0020.json')['meta'];absent=[]
        with np.load(wf) as d:wr=d['raw'];ws=d['source'];ss=d['sections'];pixels=d['pixels']
        jj=int(np.flatnonzero(pixels==57513)[0]);a,b=map(int,ss[jj:jj+2]);global_start=sum(int(np.load(p)['sections'][-1]) for p in sorted(WORLD.glob('scan*_gradients.npz')) if p.name<wf.name)+a;inverse=z['node_to_unique'][global_start:global_start+b-a]
        for row in read(R/'experiments/history_gap_extension_v1/summary.json')['real_missing_updates']:
            delta=z[row['arm']][inverse];local=Engine(wr[a:b],ws[a:b],np.array([0,b-a]),np.array([metadata_array(meta)]),warp);theta=torch.tensor(wr[a:b,:11]+.01*delta,device='cuda',dtype=torch.float64,requires_grad=True);loss,info=local.loss(theta);loss.sum().backward();pred=float(np.sum(theta.grad.cpu().numpy()*delta));error=abs(pred-row['predicted_raw_directional'])/abs(row['predicted_raw_directional']);le=abs(float(loss.detach().cpu()[0])-float(row['extended_loss']));absent.append(dict(arm=row['arm'],relative_directional_error=error,loss_error=le,gate_pass=error<=1e-5 and le<=1e-9 and int(info.cpu()[0,0])>0))
        for p in [shared/'shared_parameters.npz',R/'experiments/history_gap_extension_v1/summary.json',CONF/'summary.json']:state['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
        med={arm:float(np.median([r['ms'] for r in timings if r['arm']==arm and not r['warmup']])) for arm in ['serial','warp']};kmed={arm:float(np.median([r['ms'] for r in kernel_times if r['arm']==arm and not r['warmup']])) for arm in ['serial','warp']};counts={arm:sum(r['branches'][arm]['gate_pass'] for r in rayrows) for arm in ['serial','warp']};correct=counts['warp']==counts['serial']==891 and all(r['gate_pass'] for r in newrows+special+absent) and invalid
        result=dict(status='complete',real_pass_counts=counts,confirm_pass=sum(r['gate_pass'] for r in newrows),confirm_total=64,special=special,invalid_input_rejected=invalid,absent_updates=absent,correctness_gate_pass=bool(correct),full_median_ms=med,kernel_median_ms=kmed,full_ratio=med['warp']/med['serial'],adoption_gate_pass=bool(correct and med['warp']/med['serial']<=.95),timings=timings,kernel_timings=kernel_times,real_rows=rayrows,confirm_rows=newrows,compile_wall_seconds=compile_wall,prepare_wall_seconds=prepare_wall,scope='Same-process alternating loss-kernel and complete GPU raw-chain costs, not full training/GT/newness evidence. Ordinary F64 signs, not interval certification.');save(out/'summary.json',result)
        for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h,n
        with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for n in state['source_sha256']:archive.write(ROOT/n,n)
        state.update(status='complete',binary=state['reused_native'],artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(out/'manifest.json',state);print(json.dumps({k:v for k,v in result.items() if k not in ['timings','kernel_timings','real_rows','confirm_rows','special']}),flush=True)
    except BaseException as exc:state.update(status='failed',failure=repr(exc));save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
