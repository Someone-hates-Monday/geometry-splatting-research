"""Full-budget pilot; scoped live loss, streaming records, original state guards."""
from pathlib import Path
import argparse,hashlib,inspect,json,os,sys,time,traceback,zipfile
import numpy as np
import torch
from research.checkpoint_state import verify_binding,supplemental,state_digest
from research.history_gap_live_adapter import LiveAdapter,load_native
from research.smoke_history_gap_adapter import rng,same_rng,params

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
class DiagnosticComplete(Exception):pass
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def training_frame():
    f=inspect.currentframe().f_back
    for _ in range(20):
        if f is None:return None
        if f.f_code.co_name=='training' and Path(f.f_code.co_filename).resolve()==ROOT/'train.py':return f
        f=f.f_back
    return None
def finite(gs):
    opaque=None
    for name,p in params(gs).items():
        if name=='opacity:0':
            opaque=torch.isposinf(p)
            if not bool(torch.isfinite(p[~opaque]).all()):raise FloatingPointError('Invalid raw opacity')
            if p.grad is not None and bool((p.grad[opaque]!=0).any()):raise FloatingPointError('Nonzero opaque gradient')
        elif not bool(torch.isfinite(p).all()):raise FloatingPointError('Nonfinite parameter '+name)
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('Nonfinite gradient '+name)
    for state in gs.optimizer.state.values():
        for value in state.values():
            if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite Adam state')
    return opaque
def scalar(x):return float(x.detach()) if isinstance(x,torch.Tensor) else float(x)

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--hg-weight',required=True)
    p.add_argument('--hg-updates',type=int,default=0)
    p.add_argument('--hg-samples',type=int,default=128)
    p.add_argument('--hg-label',required=True)
    opts,args=p.parse_known_args()
    weight=None if opts.hg_weight=='disabled' else float(opts.hg_weight)
    assert weight is None or np.isfinite(weight) and weight>=0
    assert opts.hg_samples==128 and opts.hg_updates==0
    model=Path(args[args.index('-m')+1]).resolve();assert not model.exists()
    assert '--branch-sidecar' in args and '--start_checkpoint' in args and '--eval' in args
    torch.set_num_threads(1)
    # Load all three pinned private binaries before parent restores training RNG.
    qm=read(R/'experiments/csr_decode_v1/manifest.json')['capture_backend']
    cm=read(R/'experiments/csr_capture_v1/manifest.json')['csr_backend']
    wm=read(R/'experiments/history_gap_warp_v2/manifest.json')['binary']
    adapter=LiveAdapter(load_native(qm['binary_path'],qm['binary_sha256']),
        load_native(ROOT/cm['binary_relative_path'],cm['binary_sha256']),load_native(ROOT/wm['path'],wm['sha256']))
    files=[Path(__file__),R/'HISTORY_GAP_PILOT_PROTOCOL.md',R/'HISTORY_GAP_FULL_TRAIN_PROTOCOL.md',R/'history_gap_live_adapter.py',
        R/'history_gap_gpu.py',R/'history_gap_gpu_v2.py',R/'smoke_history_gap_adapter.py',
        R/'train_solver_branch.py',R/'train_holdout_branch.py',R/'checkpoint_state.py',
        R/'median_adjquad_backend.py',R/'median_cuda_backend.py',ROOT/'train.py',
        ROOT/'arguments/__init__.py',ROOT/'scene/gaussian_model.py',ROOT/'gaussian_renderer/__init__.py']
    record=dict(status='running',pid=os.getpid(),epoch=time.time(),label=opts.hg_label,
        weight=weight,updates_requested=opts.hg_updates,argv=args,steps=[],steps_observed=0,
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        binaries=dict(query=qm,csr=cm,loss=wm),
        scope='Original full recipe through30000; first6 rows in manifest, every backward/update in JSONL.')
    # Parent requires a fresh output directory, so write sources/manifest after Scene initializes.
    original_backward=torch.Tensor.backward;original_step=torch.optim.Adam.step;original_save=torch.save
    pending=None;artifact_seconds=0.;loop_start=None;started=time.perf_counter()
    def backward(self,*a,**kw):
        nonlocal pending,artifact_seconds,loop_start
        frame=training_frame()
        if frame is None or self is not frame.f_locals.get('loss'):return original_backward(self,*a,**kw)
        loc=frame.f_locals;gs=loc['gaussians'];cam=loc['viewpoint_cam'];scene=loc['scene']
        assert any(cam is c for c in scene.getTrainCameras())
        assert pending is None
        if loop_start is None:
            torch.cuda.synchronize();loop_start=time.perf_counter()
        row=dict(iteration=loc['iteration'],camera=cam.image_name,gaussians_before_backward=len(gs._xyz),
            opaque_before_backward=int(torch.isposinf(gs._opacity).sum()),
            baseline_loss=scalar(self),rgb=scalar(loc['rgb_loss']),ncc=scalar(loc['ncc_loss']),
            geo=scalar(loc['geo_loss']),depth_normal=scalar(loc['depth_normal_loss']))
        verify_binding(gs);finite(gs)
        total=self
        if weight is not None:
            rng_before=rng()
            seed=np.random.SeedSequence([13023,int(loc['iteration']),int(cam.image_name)])
            pixels=np.random.default_rng(seed).choice(cam.image_width*cam.image_height,opts.hg_samples,replace=False)
            capture=adapter.capture(cam,gs,loc['pipe'],loc['background'],loc['kernel_size'],keep_outputs=record['steps_observed']==0)
            if record['steps_observed']==0:
                public=loc['render_pkg'];query,csr=capture.outputs;bits={}
                for key,index in [('render',1),('mask',2),('normal',3),('radii',5)]:
                    bits[key]=all(torch.equal(public[key].detach().view(torch.int32),n[index].view(torch.int32)) for n in (query,csr))
                depth=torch.cat([public['median_depth'],public['interface_slope']],0).detach()
                bits['depth_slope']=all(torch.equal(depth.view(torch.int32),n[4][:2].view(torch.int32)) for n in (query,csr))
                assert all(bits.values()),bits
                row['initial_query_CSR_public_bit_parity']=bits
            extra,detail=adapter.loss(gs,capture,pixels)
            assert same_rng(rng_before,rng())
            total=self+weight*extra
            row.update(regularizer=scalar(extra),pixels=pixels.tolist(),empty=detail['empty'],
                nodes=detail['node_count'],status_counts={str(int(k)):int(v) for k,v in zip(*torch.unique(detail['info'][:,0],return_counts=True))},
                capture=capture.times,prepare_loss_wall_ms=detail['prepare_loss_wall_ms'],RNG_unchanged=True)
            del capture,detail
        else:row.update(regularizer=None)
        row['objective']=scalar(total)
        torch.cuda.synchronize();back_start=time.perf_counter()
        result=original_backward(total,*a,**kw)
        torch.cuda.synchronize();row['combined_backward_wall_ms']=1000*(time.perf_counter()-back_start)
        finite(gs)
        if not record['steps']:
            artifact_start=time.perf_counter();arrays={};absent=[]
            for name,param in params(gs).items():
                if param.grad is None:absent.append(name)
                else:arrays[name]=param.grad.detach().cpu().numpy()
            np.savez_compressed(model/'history_gap_first_gradients.npz',**arrays)
            row['first_gradients_absent']=absent
            row['first_gradient_artifact_seconds']=time.perf_counter()-artifact_start
            artifact_seconds+=row['first_gradient_artifact_seconds']
        pending=row
        return result
    def step(optimizer,*a,**kw):
        nonlocal pending,loop_start
        frame=training_frame()
        if frame is None:return original_step(optimizer,*a,**kw)
        gs=frame.f_locals['gaussians']
        if optimizer is not gs.optimizer:return original_step(optimizer,*a,**kw)
        assert pending is not None
        verify_binding(gs);opaque=finite(gs).clone()
        pending.update(gaussians_at_optimizer=len(gs._xyz),opaque_at_optimizer=int(opaque.sum()),
            gradient_groups_absent_at_optimizer=[name for name,param in params(gs).items() if param.grad is None])
        result=original_step(optimizer,*a,**kw)
        verify_binding(gs);after=finite(gs);assert torch.equal(opaque,after)
        torch.cuda.synchronize()
        pending['iteration_tail_wall_ms']=1000*(time.perf_counter()-loop_start)-1000*pending.get('first_gradient_artifact_seconds',0.)
        pending['opacity_mask_unchanged_by_Adam']=True
        row=pending;pending=None
        record['steps_observed']+=1
        if len(record['steps'])<6:record['steps'].append(row)
        record['last_step']=row
        with (model/'history_gap_steps.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        if record['steps_observed']<=6 or record['steps_observed']%100==0:
            save(model/'history_gap_training_integration.json',record)
            print(f'[history-gap] {opts.hg_label} iter{row["iteration"]} P={row["gaussians_at_optimizer"]} base={row["baseline_loss"]:.7g} reg={row["regularizer"]} interval_ms={row["iteration_tail_wall_ms"]:.1f}',flush=True)
        torch.cuda.synchronize();loop_start=time.perf_counter()
        return result
    def checkpoint_save(obj,path,*a,**kw):
        result=original_save(obj,path,*a,**kw)
        if Path(path).resolve()==model/'chkpnt30000.pth':
            frame=training_frame();assert frame is not None
            loc=frame.f_locals;gs=loc['gaussians'];scene=loc['scene']
            supp=supplemental(gs,scene,loc['viewpoint_stack'])
            side=dict(iteration=obj[1],supplemental=supp,checkpoint_sha256=sha(Path(path)),
                initial_state_sha256=state_digest((obj[0],supp)))
            target=Path(path).with_suffix('.runtime.pth');original_save(side,target)
            record['final_checkpoint']=dict(path=str(path),sha256=side['checkpoint_sha256'],
                sidecar=str(target),sidecar_sha256=sha(target),state_sha256=side['initial_state_sha256'])
        return result
    torch.Tensor.backward=backward;torch.optim.Adam.step=step;torch.save=checkpoint_save
    try:
        from research import train_holdout_branch as inner
        sys.argv=[str(Path(inner.__file__)),'--holdout-mode','warmup',*args]
        inner.main()
        parent=read(model/'solver_training_integration.json');holdout=read(model/'holdout_camera_integration.json')
        assert parent['resume_state_identical'] and parent['resumed_iteration']==6999
        assert parent['depth_render_calls']>0 and parent['ncc_calls']>0
        assert holdout['scene_initializations']==1 and holdout['filter_calls']>0
        assert len(holdout['pool']['train_names'])==42 and len(holdout['pool']['test_names'])==7
        assert holdout['ncc_calls']>0 and holdout['final_test_report_render_calls']==(0 if opts.hg_updates else 7)
        assert parent['completed'] and holdout['status']=='complete'
        assert record['steps_observed']==23000 and record['last_step']['iteration']==29999
        assert pending is not None and pending['iteration']==30000
        record['final_backward_without_optimizer']=pending
        assert record['final_checkpoint']['path']==str(model/'chkpnt30000.pth')
        record.update(status='complete',parent_guards_verified=True,initial_state_sha256=parent['initial_state_sha256'],
            full_training_completed=not bool(opts.hg_updates))
    except BaseException as exc:
        record.update(status='failed',failure=repr(exc));model.mkdir(parents=True,exist_ok=True)
        (model/'history_gap_failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        torch.Tensor.backward=original_backward;torch.optim.Adam.step=original_step;torch.save=original_save
        model.mkdir(parents=True,exist_ok=True)
        record.update(wall_seconds=time.perf_counter()-started,gradient_artifact_seconds=artifact_seconds)
        with zipfile.ZipFile(model/'history_gap_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name,expected in record['sources'].items():
                assert sha(ROOT/name)==expected,name
                z.write(ROOT/name,name)
        record['artifacts_sha256']={p.name:sha(p) for p in model.glob('history_gap_*') if p.is_file() and p.suffix!='.json'}
        save(model/'history_gap_training_integration.json',record)
if __name__=='__main__':main()
