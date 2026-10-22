"""Fixed new synthetic inputs + retained absent updates; no training."""
from pathlib import Path
import hashlib,json,os,time,traceback,zipfile,importlib.util
import numpy as np
import mpmath as mp
import torch
from research.history_gap_gpu import Engine,metadata_array
from research.world_root_vjp import mp_geometry,mp_rotation_derivatives,mp_screen_derivatives,inner
from research.history_gap_extension import mp_extended,mp_path_nodes
from research.exact_outlier_world_reference import dot,vector
from research.benchmark_history_gap_gpu import errors
R=Path(__file__).resolve().parent;ROOT=R.parent;GPU=R/'experiments/history_gap_gpu_v1';OUT=R/'experiments/history_gap_gpu_confirm_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')
def reference(raw,source,meta):
    mp.mp.dps=140;geometry=[mp_geometry(row) for row in source];mp.mp.dps=240;nodes=[[g['alpha'],g['p'],g['rho']] for g in geometry];loss,seeds,info=mp_extended(nodes);result=[];cap=mp.mpf(float(np.float32(.99)))
    for g,row,r,seed in zip(geometry,source,raw,seeds):
        ha=seed[0]*g['alpha'];hp=seed[1];hr=seed[2]
        if g['amp']*mp.exp(-g['E']/2)>cap:ha=mp.mpf(0)
        he=-ha/2;P=g['P'];d=g['d'];w=g['w'];a=g['a'];mu=hp*(P*d)/a+2*he*(P*w);HP=hp*(d*w.T+w*d.T)/(2*a)+hr*(d*d.T)/(2*g['rho'])+he*(w*w.T);Ri=g['R']**-1;F=g['S']**-1*Ri;ss=[-2*dot(F[j,:].T,HP*F[j,:].T) for j in range(3)];GR=-2*P*HP*Ri.T;qq=[inner(GR,v) for v in mp_rotation_derivatives([mp.mpf(float(x)) for x in row[6:10]])]
        gm,gs,gq,_=mp_screen_derivatives(g,row,meta);mu+=ha*gm;radius=mp.mpf(float(r[11]));kv=[mp.exp(2*mp.mpf(float(r[3+j])))/(mp.exp(2*mp.mpf(float(r[3+j])))+radius*radius) for j in range(3)];scale=[ss[j]*kv[j]+ha*(1-kv[j]+gs[j]*kv[j]) for j in range(3)];qraw=vector(r[6:10]);norm=mp.sqrt(dot(qraw,qraw));J=(mp.eye(4)-qraw*qraw.T/(norm*norm))/norm;quat=J.T*mp.matrix([qq[j]+ha*gq[j] for j in range(4)]);result.append(list(mu)+scale+list(quat)+[ha/(1+mp.exp(mp.mpf(float(r[10]))))])
    return float(loss),np.array([[float(v) for v in row] for row in result]),info
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);files=[Path(__file__),R/'history_gap_gpu.py',R/'history_gap_gpu.cu',R/'HISTORY_GAP_GPU_PROTOCOL.md',R/'world_root_vjp.py',R/'history_gap_extension.py',GPU/'manifest.json'];m=dict(status='running',pid=os.getpid(),created_epoch=time.time(),source_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files});save(OUT/'manifest.json',m)
    try:
        manifest=read(GPU/'manifest.json');path=ROOT/manifest['binary']['path'];assert sha(path)==manifest['binary']['sha256'];spec=importlib.util.spec_from_file_location(path.stem,path);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native);rng=np.random.default_rng(13018);records=[];raws=[];sources=[];sections=[0];metas=[];refs=[];losses=[]
        for case in range(32):
            n=[1,2,4,8][case%4];source=np.zeros((n,36));source[:,:3]=np.stack([np.zeros(n),np.zeros(n),rng.uniform(1.8,2.2,n)],1).astype(np.float32);source[:,3:6]=rng.uniform(.06,.2,(n,3)).astype(np.float32);qraw=rng.normal(size=(n,4));qnorm=qraw/np.linalg.norm(qraw,axis=1)[:,None];source[:,6:10]=qnorm.astype(np.float32);source[:,10]=1;source[:,20:29]=np.eye(3).reshape(1,9);source[:,32:35]=[0,0,1];target=[.3,.6,.8,1.2,1.6,2.5,1.,2.][case//4];source[:,35]=np.float32(1-np.exp(-target/n));raw=np.zeros((n,12));raw[:,:3]=source[:,:3];raw[:,3:6]=np.log(source[:,3:6]);raw[:,6:10]=qraw;raw[:,10]=rng.uniform(-1,2,n);raw[:,11]=.006;meta=dict(fx=300.,fy=310.,limx=.8,limy=.6,kernel=[0.,.3,1.][case%3],floor=float(np.float32(1e-6)))
            loss,g,info=reference(raw,source,meta);raws.append(raw);sources.append(source);refs.append(g);losses.append(loss);sections.append(sections[-1]+n);metas.append(metadata_array(meta));records.append(dict(case=case,nodes=n,kernel=meta['kernel'],reference_status=info['status'],reference_loss=loss));print(f'MP new model {case+1}/32',flush=True)
        raw=np.concatenate(raws);source=np.concatenate(sources);ref=np.concatenate(refs);sections=np.array(sections,dtype=np.int64);engine=Engine(raw,source,sections,np.array(metas),native);theta=engine.base.detach().clone().requires_grad_(True);actual_loss,info=engine.loss(theta);actual_loss.sum().backward();actual=theta.grad.cpu().numpy();values=actual_loss.detach().cpu().numpy();debug=info.cpu().numpy();rows=[]
        for j,record in enumerate(records):
            a,b=sections[j:j+2];metric=errors(actual[a:b],ref[a:b]);error=abs(float(values[j])-losses[j]);rows.append(dict(**record,gpu_status=int(debug[j,0]),gradient=metric,loss_error=error,gate_pass=metric['gate_pass'] and error<=1e-9 and debug[j,0]>0))
        np.savez_compressed(OUT/'new_arrays.npz',raw=raw,source=source,sections=sections,metadata=np.array(metas),reference=ref,actual=actual,loss=values,info=debug)
        # Retained two real absent-root updates, exact same directions and base.
        shared=R/'experiments/history_gap_shared_v1';ex=read(R/'experiments/history_gap_extension_v1/summary.json')['real_missing_updates'];wf=R/'experiments/world_root_vjp_v1/scan37_0020_gradients.npz';rf=R/'experiments/history_gap_raw_v1/scan37_0020.json';meta=read(R/'experiments/world_root_vjp_v1/scan37_0020.json')['meta'];z=np.load(shared/'shared_parameters.npz')
        for p in [wf,rf,shared/'shared_parameters.npz',R/'experiments/history_gap_extension_v1/summary.json']:m['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
        with np.load(wf) as d:wr=d['raw'];ws=d['source'];ss=d['sections'];pixels=d['pixels']
        j=int(np.flatnonzero(pixels==57513)[0]);a,b=map(int,ss[j:j+2]);global_start=sum(int(np.load(p)['sections'][-1]) for p in sorted(wf.parent.glob('scan*_gradients.npz')) if p.name<wf.name)+a;inverse=z['node_to_unique'][global_start:global_start+b-a];absent=[]
        for record in ex:
            delta=z[record['arm']][inverse];local=Engine(wr[a:b],ws[a:b],np.array([0,b-a]),np.array([metadata_array(meta)]),native);theta=torch.tensor(wr[a:b,:11]+.01*delta,device='cuda',dtype=torch.float64,requires_grad=True);val,di=local.loss(theta);val.sum().backward();g=theta.grad.cpu().numpy();pred=float(np.sum(g*delta));loss_error=abs(float(val.detach().cpu()[0])-float(record['extended_loss']));relative=abs(pred-record['predicted_raw_directional'])/abs(record['predicted_raw_directional']);absent.append(dict(arm=record['arm'],loss_error=loss_error,relative_directional_error=relative,status=int(di.cpu()[0,0]),gate_pass=loss_error<=1e-9 and relative<=1e-5))
        summary=dict(status='complete',new_cases=32,new_nodes=len(raw),new_pass=sum(r['gate_pass'] for r in rows),rows=rows,real_absent_updates=absent,scope='New fixed synthetic captured-source raw models, including nonzero screen kernels; two retained real absent updates. Neither new real scenes nor training/GT evidence.');save(OUT/'summary.json',summary)
        for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
        with zipfile.ZipFile(OUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for name in m['source_sha256']:archive.write(ROOT/name,name)
        m.update(status='complete',artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'});save(OUT/'manifest.json',m);print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
    except BaseException as exc:m.update(status='failed',failure=repr(exc));save(OUT/'manifest.json',m);(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
