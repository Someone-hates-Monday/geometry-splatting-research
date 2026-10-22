"""A common actual-ID update for the frozen 891-ray diagnostic objective."""
from pathlib import Path
import hashlib,json,os,time,traceback,zipfile
import numpy as np
import torch
from research.history_gap_raw import parameter_scale,torch_forward
R=Path(__file__).resolve().parent;ROOT=R.parent;RAW=R/'experiments/history_gap_raw_v1';WORLD=R/'experiments/world_root_vjp_v1';MICRO=R/'experiments/history_gap_raw_micro_v1';OUT=R/'experiments/history_gap_shared_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);torch.set_num_threads(1)
    files=[Path(__file__),R/'HISTORY_GAP_SHARED_PROTOCOL.md',R/'history_gap_raw.py',R/'history_gap_math.py',R/'world_root_vjp_v2.py',RAW/'manifest.json',RAW/'summary.json',MICRO/'manifest.json',MICRO/'summary.json'];m=dict(status='running',pid=os.getpid(),created_epoch=time.time(),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files});save(OUT/'manifest.json',m)
    try:
        rawm=read(RAW/'manifest.json');worldm=read(WORLD/'manifest.json');assert read(RAW/'summary.json')['branch_pass_counts']['factor']==891 and read(MICRO/'summary.json')['mp_micro_pass']==7
        records=[];allkeys=[];allparams=[];allscales=[];allg=[];baseline=[];node_offset=0
        for f in sorted(RAW.glob('scan*_gradients.npz')):
            stem=f.name.removesuffix('_gradients.npz');scan=int(stem[4:6]);wf=WORLD/f.name;meta=WORLD/(stem+'.json');rowfile=RAW/(stem+'.json')
            for p,manifest in [(f,rawm),(rowfile,rawm),(wf,worldm),(meta,worldm)]:assert sha(p)==manifest['artifacts_sha256'][p.name];m['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
            with np.load(f) as d:g=d['factor'];ids=d['primitive_ids'];sections=d['sections'];pixels=d['pixels']
            with np.load(wf) as d:raw=d['raw'];source=d['source']
            metadata=read(meta)['meta'];rows=read(rowfile)['rows'];allkeys.append(np.int64(scan)*np.int64(2**32)+ids.astype(np.int64));allparams.append(raw);allscales.append(parameter_scale(raw,source));allg.append(g)
            for j,row in enumerate(rows):
                a,b=map(int,sections[j:j+2]);records.append(dict(scan=scan,view=stem,pixel=int(pixels[j]),raw=raw[a:b],source=source[a:b],meta=metadata,node_begin=node_offset+a,node_end=node_offset+b));baseline.append(row['branches']['factor']['loss'])
            node_offset+=len(source)
        assert len(records)==891 and node_offset==51524
        keys=np.concatenate(allkeys);params=np.concatenate(allparams);scales=np.concatenate(allscales);g=np.concatenate(allg);unique,first,inverse,counts=np.unique(keys,return_index=True,return_inverse=True,return_counts=True);canonical=params[first];S=scales[first]
        assert np.array_equal(params,canonical[inverse]) and np.array_equal(scales,S[inverse]);summed=np.zeros((len(unique),11));np.add.at(summed,inverse,g/891)
        directions={};norms={}
        for arm in ['full','geometry','opacity']:
            active=summed.copy()
            if arm=='geometry':active[:,10]=0
            elif arm=='opacity':active[:,:10]=0
            norm=float(np.linalg.norm(S*active));assert norm>0;directions[arm]=-S*S*active/norm;norms[arm]=norm
        energy={name:float(np.sum((S[:,a:b]*summed[:,a:b])**2)) for name,a,b in [('xyz',0,3),('log_scale',3,6),('quaternion',6,10),('opacity_logit',10,11)]}
        np.savez_compressed(OUT/'shared_parameters.npz',keys=unique,raw=canonical,scale=S,counts=counts,node_to_unique=inverse,mean_gradient=summed,**directions)
        control=np.array(baseline);np.save(OUT/'baseline.npy',control)
        variants=[('zero','full',0.)]+[(f'{arm}_{h}',arm,h) for arm in ['full','geometry','opacity'] for h in [1e-4,1e-3,1e-2]]+[(f'full_minus_{h}','full',-h) for h in [1e-4,1e-5]]+[('full_plus_1e-05','full',1e-5)]
        results=[];loss_arrays={};statuses={}
        for label,arm,h in variants:
            losses=np.full(891,np.nan);states=[]
            for j,record in enumerate(records):
                a,b=record['node_begin'],record['node_end'];delta=directions[arm][inverse[a:b]];theta=record['raw'][:,:11]+h*delta
                loss,info=torch_forward(theta,record['raw'],record['source'],record['meta']);states.append(info)
                if loss is not None:losses[j]=loss
                if (j+1)%64==0 or j==890:save(OUT/'progress.json',dict(pid=os.getpid(),branch=label,completed_rays=j+1,branches_completed=len(results)));print(f'shared {label}: {j+1}/891',flush=True)
            np.save(OUT/(label+'_loss.npy'),losses);save(OUT/(label+'_roots.json'),dict(rows=states));loss_arrays[label]=losses;statuses[label]=states
            finite=bool(np.isfinite(losses).all());row=dict(branch=label,arm=arm,step=h,finite_rays=int(np.isfinite(losses).sum()),all_rays_finite=finite)
            if finite:
                mean=float(losses.mean());delta=mean-float(control.mean());row.update(mean_loss=mean,mean_delta=delta,relative_mean_delta=delta/float(control.mean()),armijo=(h>0 and mean<=float(control.mean())-1e-4*h*norms[arm]),measurable_decrease=delta< -max(1e-12,1e-8*float(control.mean())),increased_rays=int(np.count_nonzero(losses>control)),decreased_rays=int(np.count_nonzero(losses<control)),unchanged_rays=int(np.count_nonzero(losses==control)),scene_means={str(scan):float(losses[[r['scan']==scan for r in records]].mean()) for scan in [37,55,63]})
            results.append(row);save(OUT/'partial.json',dict(rows=results))
        assert np.array_equal(loss_arrays['zero'].view(np.uint64),control.view(np.uint64))
        fd=[];pred=-norms['full']
        for h in [1e-4,1e-5]:
            plus=loss_arrays['full_0.0001' if h==1e-4 else 'full_plus_1e-05'];minus=loss_arrays[f'full_minus_{h}'];finite=bool(np.isfinite(plus).all() and np.isfinite(minus).all());v=float((plus.mean()-minus.mean())/(2*h)) if finite else None;fd.append(dict(h=h,derivative=v,predicted=pred,absolute_error=abs(v-pred) if finite else None,relative_error=abs(v-pred)/abs(pred) if finite else None,gate_pass=finite and abs(v-pred)<=max(1e-9,1e-5*abs(pred))))
        result=dict(status='complete',rays=891,nodes=51524,unique_primitives=len(unique),repeated_primitives=int(np.count_nonzero(counts>1)),repeated_occurrences=int(np.sum(counts[counts>1])),maximum_multiplicity=int(counts.max()),raw_and_scale_consistency=True,baseline_mean=float(control.mean()),zero_control_bits_equal=True,scaled_gradient_norms=norms,scaled_gradient_energy=energy,rows=results,directional=fd,original_raw_gate_pass=False,original_gate_verdict_preserved=True,scope='Actual shared-ID update on the fixed continuous ray objective; three models, no full renderer visibility update, CUDA training, GT or novelty evidence.')
        save(OUT/'summary.json',result)
        for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
        with zipfile.ZipFile(OUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name in m['source_sha256']:z.write(ROOT/name,name)
        m.update(status='complete',completed_epoch=time.time(),artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(OUT/'manifest.json',m);print(json.dumps(result),flush=True)
    except BaseException as exc:m.update(status='failed',failure=repr(exc));save(OUT/'manifest.json',m);(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
