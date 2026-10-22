"""Compare exact-formula sampling grouping and complete frozen loss/backward.

No model updates. Includes actual PatchMatch geometry/NCC in full-path timings.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.benchmark_median_cuda import load_config
from research.precise_preprocess_backend import activate,load_backend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',nargs='+',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=10)
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'building','steps':[],'source_sha256':{},
        'scope':'Three frozen scenes, two fixed development cameras each; root formulas/five passes unchanged; FP64 world backward and per-thread query batch differ; original float32 control retained.'}
    def write():
        (out/'manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    sources=[Path(__file__),ROOT/'research/precise_preprocess_backend.py',ROOT/'gaussian_renderer/__init__.py',ROOT/'utils/loss_utils.py']
    source=ROOT/'submodules/diff-gaussian-rasterization'
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        sources.extend(source.glob(pattern))
    state['source_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in sources:archive.write(p,p.relative_to(ROOT).as_posix())
    state['source_archive_sha256']=hashlib.sha256((out/'sources.zip').read_bytes()).hexdigest()
    write()
    started=time.perf_counter()
    try:
        backends={v:load_backend(v) for v in ('control','precise1','precise2')}
        state['status']='running';write()
        from scene import GaussianModel,Scene
        from gaussian_renderer import render,sample_depth
        from utils.loss_utils import PatchMatch,l1_loss,ssim
        fields=('_xyz','_scaling','_rotation','_opacity')
        all_fields=fields+('_features_dc','_features_rest','_sg_axis','_sg_sharpness','_sg_color')
        rng=np.random.default_rng(0)
        rows=[]
        for cpu_path in args.cpu:
            ref=json.loads(cpu_path.read_text());model=Path(ref['model'])
            ply=model/'point_cloud/iteration_30000/point_cloud.ply'
            sha=hashlib.sha256(ply.read_bytes()).hexdigest()
            if [v for k,v in ref['source_sha256'].items() if k.endswith('point_cloud.ply')]!=[sha]:raise ValueError('Frozen PLY changed.')
            state['source_sha256'][str(ply)]=sha
            ds,pipe=load_config(model)
            with tempfile.TemporaryDirectory(prefix='sampling-batch-',dir=out) as temp:
                scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True)
                ds.model_path=str(scratch)
                gs=GaussianModel(ds.sh_degree,ds.sg_degree)
                scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
                cameras=scene.getTrainCameras();lookup={c.image_name:c for c in cameras}
                bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
                pm=PatchMatch(3,1.,ds.kernel_size,pipe,debug=False,model_path=str(scratch))
                names=list(dict.fromkeys(r['camera'] for r in ref['rows']))[::4]
                for name in names:
                    cam=lookup[name]
                    if not cam.nearest_id:raise ValueError('No nearest camera.')
                    near=cameras[cam.nearest_id[0]]
                    with torch.no_grad(),activate(backends['control']):
                        package=render(cam,gs,pipe,bg,ds.kernel_size,require_depth=True)
                        z=package['median_depth'].squeeze()
                        ix=(torch.arange(cam.image_width,device='cuda')-cam.Cx)/cam.Fx
                        iy=(torch.arange(cam.image_height,device='cuda')-cam.Cy)/cam.Fy
                        local=torch.stack((z*ix[None,:],z*iy[:,None],z),dim=-1)
                        points=((local-cam.T)@cam.R.T).detach().requires_grad_(True)
                        del package,z,local
                    def clear():
                        points.grad=None
                        for f in all_fields:getattr(gs,f).grad=None
                    # Full-image primitive outputs, with a fixed 256-point signed loss.
                    count=points.shape[0]*points.shape[1]
                    selected=rng.choice(count,size=min(256,count),replace=False)
                    ids=torch.as_tensor(selected,device='cuda')
                    signs=torch.as_tensor(rng.choice([-1.,1.],size=len(selected)),device='cuda',dtype=torch.float32)
                    outputs={};gradients={};losses={}
                    for v,backend in (('control',backends['control']),('control_repeat',backends['control']),('precise1',backends['precise1']),('precise2',backends['precise2'])):
                        clear()
                        with activate(backend):
                            sampled=sample_depth(points,near,gs,pipe,ds.kernel_size)
                            loss=(sampled['sampled_depth'].reshape(-1,3)[ids,2]*signs).mean()
                            loss.backward();torch.cuda.synchronize()
                        outputs[v]={k:value.detach().cpu() for k,value in sampled.items()}
                        gradients[v]={f:getattr(gs,f).grad.detach().cpu().double() for f in fields}
                        gradients[v]['query_points']=points.grad.detach().cpu().double()
                        losses[v]=float(loss.detach())
                        del sampled,loss
                    comparisons={}
                    for v in ('precise1','precise2'):
                        a,b=outputs['control'],outputs[v]
                        inside_equal=torch.equal(a['inside'],b['inside'])
                        output_equal=torch.equal(a['sampled_depth'],b['sampled_depth'])
                        per_field={}
                        for f in (*fields,'query_points'):
                            x,y=gradients['control'][f],gradients[v][f]
                            norm=max(float(torch.linalg.vector_norm(x)),1e-12)
                            per_field[f]={'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
                                'relative_l2_drift':float(torch.linalg.vector_norm(y-x))/norm,
                                'control_repeat_relative_l2_drift':float(torch.linalg.vector_norm(gradients['control_repeat'][f]-x))/norm}
                        comparisons[v]={'inside_bitwise_equal':inside_equal,'sampled_xyz_bitwise_equal':output_equal,
                            'sampled_xyz_max_abs_difference':float((a['sampled_depth']-b['sampled_depth']).abs().max()),
                            'gradient_comparison':per_field}
                    del outputs,gradients
                    (out/f'parity_{model.name}_{name}.json').write_text(json.dumps(
                        {'model':str(model),'camera':name,'nearest_camera':near.image_name,
                         'sample_losses':losses,'comparisons':comparisons},indent=2,allow_nan=False)+'\n')
                    def sample_step(backend):
                        with activate(backend):
                            sampled=sample_depth(points,near,gs,pipe,ds.kernel_size)
                            loss=sampled['sampled_depth'].mean();loss.backward()
                    gt=cam.original_image.cuda()
                    def full_step(backend):
                        with activate(backend):
                            pkg=render(cam,gs,pipe,bg,ds.kernel_size,require_depth=True)
                            rgb=.8*l1_loss(pkg['render'],gt)+.2*(1-ssim(pkg['render'].unsqueeze(0),gt.unsqueeze(0)))
                            ncc,geo=pm(gs,pkg,cam,near,iteration=7001)
                            loss=rgb+.6*ncc+.02*geo
                            loss.backward()
                            return loss
                    full_losses={}
                    for v,backend in backends.items():
                        clear();loss=full_step(backend);torch.cuda.synchronize()
                        full_losses[v]=float(loss.detach());del loss
                    timings={kind:{v:[] for v in backends} for kind in ('sample_forward_backward','render_patchmatch_loss_backward')}
                    orders=[]
                    for kind,operation in (('sample_forward_backward',sample_step),('render_patchmatch_loss_backward',full_step)):
                        for backend in backends.values():clear();operation(backend)
                        torch.cuda.synchronize()
                        for _ in range(args.rounds):
                            order=list(backends);rng.shuffle(order);orders.append({'kind':kind,'order':order})
                            for v in order:
                                events=[]
                                for _ in range(args.repeats):
                                    clear();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                                    start.record();operation(backends[v]);end.record();events.append((start,end))
                                torch.cuda.synchronize()
                                timings[kind][v].append(statistics.mean(a.elapsed_time(b) for a,b in events))
                    rows.append({'model':str(model),'camera':name,'nearest_camera':near.image_name,'query_points':count,
                        'sample_losses':losses,'full_frozen_losses':full_losses,'comparisons':comparisons,'gpu_ms_rounds':timings,'orders':orders})
                    print(f'{model.name}/{name}: '+', '.join(f'{v} sample={statistics.median(timings["sample_forward_backward"][v]):.3f}ms full={statistics.median(timings["render_patchmatch_loss_backward"][v]):.3f}ms' for v in backends),flush=True)
                    state['steps'].append({'model':str(model),'camera':name,'status':'complete'});write()
                    clear();del points,comparisons
                del gs,scene,cameras
                torch.cuda.empty_cache()
        for backend in backends.values():
            p=Path(backend.__file__);state['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        all_forward=all(c['inside_bitwise_equal'] and c['sampled_xyz_bitwise_equal'] for r in rows for c in r['comparisons'].values())
        all_grads=all(c['finite'] and c['relative_l2_drift']<=.001 for r in rows for v in r['comparisons'].values() for c in v['gradient_comparison'].values())
        report={'rows':rows,'rounds':args.rounds,'repeats':args.repeats,'all_sample_outputs_bitwise_equal':all_forward,
            'all_gradient_groups_finite_and_drift_le_0_001':all_grads,'source_sha256':state['source_sha256'],
            'limitations':['Six frozen development camera pairs, no optimizer/densification/appearance update or trajectory quality.',
                'Full-path timing uses actual render, PatchMatch geometry/NCC, RGB loss and backward, but no optimizer step.',
                'FP64 and grouping can change gradients relative to float32 control; this drift is descriptive and is not an accuracy oracle.',
                'All timing phases are serial on one GPU; correlated views/rounds are not independent training replications.']}
        (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        state['status']='complete';state['forward_gate_pass']=all_forward;state['gradient_gate_pass']=all_grads
    except BaseException as error:
        state['status']='failed';state['error']=repr(error);raise
    finally:state['wall_seconds']=time.perf_counter()-started;write()


if __name__=='__main__':main()
