from pathlib import Path
import argparse,copy,hashlib,json,os,random,tempfile,time,traceback,zipfile
import numpy as np
import torch
from research.history_gap_live_adapter import LiveAdapter,load_native,measured,ROOT
from research.benchmark_median_cuda import load_config
from research.checkpoint_state import verify_binding
from research.benchmark_history_gap_gpu import errors
R=ROOT/'research';WORLD=R/'experiments/world_root_vjp_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rng():return random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),[t.clone() for t in torch.cuda.get_rng_state_all()]
def same_rng(a,b):return a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2]) and all(torch.equal(x,y) for x,y in zip(a[3],b[3]))
def params(gs):return {g['name']+':'+str(j):p for g in gs.optimizer.param_groups for j,p in enumerate(g['params'])}
def grads(gs):return {name:None if p.grad is None else p.grad.detach().clone() for name,p in params(gs).items()}
def differences(a,b):
    out={}
    for k in a:
        if a[k] is None or b[k] is None:out[k]=0. if a[k] is None and b[k] is None else None
        else:
            mask=torch.isposinf(b[k])
            if not torch.equal(torch.isposinf(a[k]),mask):out[k]=None
            else:
                aa=torch.where(mask,torch.zeros_like(a[k]),a[k]);bb=torch.where(mask,torch.zeros_like(b[k]),b[k]);out[k]=float(torch.linalg.vector_norm((aa-bb).double())/max(float(torch.linalg.vector_norm(bb.double())),1e-30))
    return out
def finite(gs):
    for k,p in params(gs).items():
        if k=='opacity:0':
            mask=gs._history_gap_opaque_mask
            if not torch.equal(torch.isposinf(p),mask) or not bool(torch.isfinite(p[~mask]).all()):raise FloatingPointError('Changed or invalid opacity boundary')
            if p.grad is not None and bool((p.grad[mask]!=0).any()):raise FloatingPointError('Saturated opacity must have exactly zero gradient')
        elif not bool(torch.isfinite(p).all()):raise FloatingPointError('Nonfinite parameter '+k)
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('Nonfinite gradient '+k)
    for state in gs.optimizer.state.values():
        for value in state.values():
            if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite Adam state')
def rgb(cam,gs,pipe,bg,kernel,opt):
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss,ssim
    package=render(cam,gs,pipe,bg,kernel,require_depth=True);image=package['render'];gt=cam.original_image.cuda();loss=(1-opt.lambda_dssim)*l1_loss(image,gt)+opt.lambda_dssim*(1-ssim(image.unsqueeze(0),gt.unsqueeze(0)));return loss,package
def replay(out,stem,public,capture):
    q,c=capture.outputs;arrays={};passed={}
    fields=[('rgb','render',1),('alpha','mask',2),('normal','normal',3),('radii','radii',5)]
    for name,key,index in fields:
        arrays['public_'+name]=public[key].detach().cpu().numpy()
        for label,n in [('query',q),('csr',c)]:arrays[label+'_'+name]=n[index].cpu().numpy();passed[label+'_'+name]=bool(np.array_equal(arrays[label+'_'+name].view(np.uint32),arrays['public_'+name].view(np.uint32)))
    arrays['public_depth']=torch.cat([public['median_depth'],public['interface_slope']],0).detach().cpu().numpy()
    for label,n in [('query',q),('csr',c)]:arrays[label+'_depth']=n[4][:2].cpu().numpy();passed[label+'_depth']=bool(np.array_equal(arrays[label+'_depth'].view(np.uint32),arrays['public_depth'].view(np.uint32)))
    arrays['query_counters']=q[4][:6].cpu().numpy();arrays['csr_counters']=c[4][:6].cpu().numpy();passed['counters']=bool(np.array_equal(arrays['query_counters'].view(np.uint32),arrays['csr_counters'].view(np.uint32)))
    np.savez_compressed(out/(stem+'_replay.npz'),**arrays);return passed
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);out=parser.parse_args().out.resolve();assert not out.exists();out.mkdir(parents=True);files=[Path(__file__),R/'history_gap_live_adapter.py',R/'HISTORY_GAP_ADAPTER_PROTOCOL.md',R/'HISTORY_GAP_ADAPTER_OPACITY_AMENDMENT.md',R/'history_gap_gpu.py',R/'history_gap_gpu_v2.py',R/'checkpoint_state.py',ROOT/'gaussian_renderer/__init__.py',ROOT/'scene/gaussian_model.py',ROOT/'arguments/__init__.py',WORLD/'manifest.json'];m=dict(status='running',created_epoch=time.time(),pid=os.getpid(),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files});save(out/'manifest.json',m)
    try:
        from scene import Scene,GaussianModel
        from arguments import OptimizationParams
        torch.set_num_threads(1);qm=read(R/'experiments/csr_decode_v1/manifest.json')['capture_backend'];cm=read(R/'experiments/csr_capture_v1/manifest.json')['csr_backend'];wm=read(R/'experiments/history_gap_warp_v2/manifest.json')['binary'];query=load_native(qm['binary_path'],qm['binary_sha256']);csr=load_native(ROOT/cm['binary_relative_path'],cm['binary_sha256']);native=load_native(ROOT/wm['path'],wm['sha256']);adapter=LiveAdapter(query,csr,native);results=[]
        m['binaries']={'query':dict(path=qm['binary_path'],sha256=qm['binary_sha256']),'csr':dict(path=cm['binary_relative_path'],sha256=cm['binary_sha256']),'loss':wm};save(out/'manifest.json',m)
        for scan in [37,55,63]:
            stem=f'scan{scan}_0000';model=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=model/'point_cloud/iteration_30000/point_cloud.ply';expected=read(R/'experiments/full_topology_v1/manifest.json')['model_inputs'][str(scan)]['ply_sha256'];assert sha(ply)==expected;m['source_sha256'][ply.relative_to(ROOT).as_posix()]=expected;ds,pipe=load_config(model)
            with tempfile.TemporaryDirectory(prefix='live-gap-scene-',dir=out) as temp:
                scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch);gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False);cam=next(c for c in scene.getTrainCameras()+scene.getTestCameras() if c.image_name=='0000');bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda');p=argparse.ArgumentParser();op=OptimizationParams(p);opt=op.extract(p.parse_args([]));gs.spatial_lr_scale=scene.cameras_extent;gs.training_setup(opt);gs.update_learning_rate(30000);verify_binding(gs)
                gs._history_gap_opaque_mask=torch.isposinf(gs._opacity).clone();finite(gs);np.savez_compressed(out/(stem+'_opaque.npz'),ids=torch.nonzero(gs._history_gap_opaque_mask.flatten()).flatten().cpu().numpy())
                with torch.no_grad():_,public=rgb(cam,gs,pipe,bg,ds.kernel_size,opt)
                before=rng();bundle=adapter.capture(cam,gs,pipe,bg,ds.kernel_size,True);assert same_rng(before,rng());replay_bits=replay(out,stem,public,bundle);assert all(replay_bits.values());print(f'{stem}: public/query/CSR bit replay passed',flush=True)
                probe=WORLD/(stem+'_gradients.npz');assert sha(probe)==read(WORLD/'manifest.json')['artifacts_sha256'][probe.name];m['source_sha256'][probe.relative_to(ROOT).as_posix()]=sha(probe)
                with np.load(probe) as d:pp=d['pixels'];ps=d['sections'];pid=d['primitive_ids'];source=d['source'];raw=d['raw']
                gradient_probe=R/'experiments/history_gap_raw_v1'/probe.name;assert sha(gradient_probe)==read(R/'experiments/history_gap_raw_v1/manifest.json')['artifacts_sha256'][gradient_probe.name];m['source_sha256'][gradient_probe.relative_to(ROOT).as_posix()]=sha(gradient_probe)
                with np.load(gradient_probe) as d:ref=d['reference'];assert np.array_equal(d['pixels'],pp) and np.array_equal(d['sections'],ps) and np.array_equal(d['primitive_ids'],pid)
                gs.optimizer.zero_grad(set_to_none=True);loss,detail=adapter.loss(gs,bundle,pp);assert detail['empty']==0;assert np.array_equal(detail['ids'].cpu().numpy(),pid) and np.array_equal(detail['sections'].cpu().numpy(),ps) and np.array_equal(detail['source'].cpu().numpy().view(np.uint64),source.view(np.uint64)) and np.array_equal(detail['raw'].cpu().numpy().view(np.uint64),raw.view(np.uint64));loss.backward();finite(gs)
                unique,inverse=np.unique(pid,return_inverse=True);expected_grad=np.zeros((len(unique),11));np.add.at(expected_grad,inverse,ref/len(pp));indices=torch.tensor(unique.astype(np.int64),device='cuda');actual=torch.cat([gs._xyz.grad[indices],gs._scaling.grad[indices],gs._rotation.grad[indices],gs._opacity.grad[indices]],1).cpu().numpy();scatter=errors(actual,expected_grad)
                np.savez_compressed(out/(stem+'_scatter.npz'),pixels=pp,sections=ps,ids=pid,source=source,raw=raw,reference_per_node=ref,unique_ids=unique,reference=expected_grad,actual=actual);save(out/(stem+'_scatter_metrics.json'),scatter);assert scatter['gate_pass'];unintended=[name for name,p0 in params(gs).items() if name.split(':')[0] not in ['xyz','scaling','rotation','opacity'] and p0.grad is not None and bool((p0.grad!=0).any())];assert not unintended
                indices_random=np.random.default_rng(13021+scan).choice(cam.image_width*cam.image_height,128,replace=False);pixels=np.unique(np.concatenate([indices_random,pp]));before=rng();sample_loss,sample=adapter.loss(gs,bundle,pixels);assert same_rng(before,rng());empty_pixels=torch.nonzero(bundle.counts==0).flatten();all_empty_check=None
                if empty_pixels.numel():
                    empty_loss,ed=adapter.loss(gs,bundle,empty_pixels[:1]);all_empty_check=dict(empty=ed['empty'],loss=float(empty_loss.detach()),expected=float(np.log(4)**2));assert ed['empty']==1 and abs(all_empty_check['loss']-all_empty_check['expected'])<1e-14
                np.savez_compressed(out/(stem+'_sample.npz'),pixels=pixels,counts=bundle.counts[torch.tensor(pixels,device='cuda')].cpu().numpy(),active_pixels=sample['active_pixels'].cpu().numpy(),ids=sample.get('ids',torch.empty(0,device='cuda')).cpu().numpy(),sections=sample.get('sections',torch.tensor([0],device='cuda')).cpu().numpy(),values=sample.get('values',torch.empty(0,device='cuda')).detach().cpu().numpy(),empty=np.array([sample['empty']]),loss=np.array([float(sample_loss.detach())]))
                controls=[];control_losses=[]
                for label in ['disabled_a','disabled_b','computed_zero']:
                    gs.optimizer.zero_grad(set_to_none=True);base_loss,_=rgb(cam,gs,pipe,bg,ds.kernel_size,opt)
                    if label=='computed_zero':captured=adapter.capture(cam,gs,pipe,bg,ds.kernel_size);extra,_=adapter.loss(gs,captured,pixels);objective=base_loss+0*extra
                    else:objective=base_loss
                    objective.backward();finite(gs);controls.append(grads(gs));control_losses.append(float(base_loss.detach()))
                repeat_diff=differences(controls[1],controls[0]);zero_diff=differences(controls[2],controls[0]);zero_gate=all(zero_diff[k] is not None and repeat_diff[k] is not None and zero_diff[k]<=max(1e-5,5*repeat_diff[k]) for k in zero_diff);assert zero_gate
                initial={name:p.detach().clone() for name,p in params(gs).items()};adam=copy.deepcopy(gs.optimizer.state_dict());arm_states={};arm_rows=[]
                for label,weight in [('disabled_a',None),('disabled_b',None),('computed_zero',0.),('positive',.01)]:
                    with torch.no_grad():
                        for name,p0 in params(gs).items():p0.copy_(initial[name])
                    gs.optimizer.load_state_dict(copy.deepcopy(adam));verify_binding(gs)
                    for step in range(3):
                        torch.cuda.synchronize();started=time.perf_counter();gs.optimizer.zero_grad(set_to_none=True);(base_loss,_),render_time=measured(lambda:rgb(cam,gs,pipe,bg,ds.kernel_size,opt));extra=None;capture_times=None;prepare_ms=None;sampled_empty=None
                        if weight is not None:
                            before=rng();current=adapter.capture(cam,gs,pipe,bg,ds.kernel_size);extra,dd=adapter.loss(gs,current,pixels);assert same_rng(before,rng());capture_times=current.times;prepare_ms=dd['prepare_loss_wall_ms'];sampled_empty=dd['empty'];objective=base_loss+weight*extra
                        else:objective=base_loss
                        _,back_time=measured(lambda:objective.backward());finite(gs);_,opt_time=measured(gs.optimizer.step);finite(gs);verify_binding(gs);torch.cuda.synchronize();wall=1000*(time.perf_counter()-started);row=dict(arm=label,weight=weight,step=step,rgb_loss=float(base_loss.detach()),regularizer=None if extra is None else float(extra.detach()),sampled_empty=sampled_empty,render=render_time,capture=capture_times,prepare_loss_wall_ms=prepare_ms,backward=back_time,optimizer=opt_time,total_diagnostic_wall_ms=wall);arm_rows.append(row);save(out/(stem+'_steps.json'),arm_rows);save(out/'progress.json',dict(scan=scan,arm=label,step=step,wall_ms=wall));print(f'{stem} {label} step{step}: RGB={row["rgb_loss"]:.6g} reg={row["regularizer"]} wall={wall:.1f}ms',flush=True)
                    arm_states[label]={name:p0.detach().clone() for name,p0 in params(gs).items()}
                    kept=torch.unique(sample.get('ids',torch.empty(0,device='cuda',dtype=torch.int64))).long();np.savez_compressed(out/(stem+'_'+label+'_selected.npz'),ids=kept.cpu().numpy(),raw=torch.cat([gs._xyz[kept],gs._scaling[kept],gs._rotation[kept],gs._opacity[kept]],1).detach().cpu().numpy())
                pd=differences(arm_states['disabled_b'],arm_states['disabled_a']);zd=differences(arm_states['computed_zero'],arm_states['disabled_a']);pgate=all(zd[k]<=max(1e-5,5*pd[k]) for k in zd);assert pgate
                with torch.no_grad():
                    for name,p0 in params(gs).items():p0.copy_(initial[name])
                gs.optimizer.load_state_dict(copy.deepcopy(adam));gs.optimizer.zero_grad(set_to_none=True);assert all(torch.equal(p0,initial[name]) for name,p0 in params(gs).items());assert sha(ply)==expected
                result=dict(scan=scan,camera='0000',primitive_count=int(gs._xyz.shape[0]),preserved_positive_inf_opacity=int(gs._history_gap_opaque_mask.sum()),saturated_opacity_gradient_exactly_zero=True,other_parameters_gradients_and_Adam_finite=True,replay_bits=replay_bits,scatter=scatter,unintended_regularizer_gradients=unintended,sample_count=len(pixels),sample_empty=sample['empty'],all_empty_check=all_empty_check,RNG_unchanged=True,zero_gradient_gate=zero_gate,control_repeat_gradient_difference=repeat_diff,zero_gradient_difference=zero_diff,control_rgb_losses=control_losses,zero_parameter_gate=pgate,control_repeat_parameter_difference=pd,zero_parameter_difference=zd,parameter_snapshot_restored=True,optimizer_binding_verified=True,learning_rates={g['name']:g['lr'] for g in gs.optimizer.param_groups},steps=arm_rows,gate_pass=True);results.append(result);save(out/(stem+'.json'),result)
                del gs,scene,bundle,controls,arm_states,initial;torch.cuda.empty_cache()
        summary=dict(status='complete',scenes=results,gate_pass=all(r['gate_pass'] for r in results),scope='Actual live PLY GaussianModels, native GPU capture and shared Float32 parameter scatter; fresh-Adam RGB-only fixed-topology memory smoke, not full training recipe, resumed state, GT or heldout quality.');save(out/'summary.json',summary)
        for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
        with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
            for n in m['source_sha256']:
                if not n.endswith('.ply'):z.write(ROOT/n,n)
        m.update(status='complete',artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(out/'manifest.json',m)
    except BaseException as exc:m.update(status='failed',failure=repr(exc));save(out/'manifest.json',m);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
