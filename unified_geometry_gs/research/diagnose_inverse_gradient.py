"""Check inverse-factor VJP stability with sampling batches 2 and 1.

Two cameras selected from the failed expanded audit: scan55/0013 (largest
variant drift) and scan63/0048 (large baseline repeat drift). This diagnostic
does not turn an unsuccessful engineering gate into a passing one.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.benchmark_median_cuda import load_config
from research.stable_inverse_backend import activate,load_backend


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    ref=json.loads((args.audit/'results.json').read_text())
    selected=[r for r in ref['rows'] if (Path(r['model']).name,r['camera']) in (('scan55','0013'),('scan63','0048'))]
    if len(selected)!=2:raise ValueError('Fixed diagnostic cameras missing.')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'research/stable_inverse_backend.py',ROOT/'research/median_cuda_backend.py',
             ROOT/'gaussian_renderer/__init__.py',ROOT/'utils/loss_utils.py']
    raster=ROOT/'submodules/diff-gaussian-rasterization'
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        sources.extend(raster.glob(pattern))
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
           'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
           'selection':'Two worst-case cameras chosen after expanded full-loss audit; exploratory mechanism check.',
           'baseline':'stable2: direct inverse-factor VJP, original sampling batch 2',
           'candidate':'stable1: same direct VJP, sampling batch 1',
           'source_audit_sha256':sha(args.audit/'results.json')}
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT).as_posix())
    state['source_archive_sha256']=sha(out/'sources.zip')
    def write():(out/'manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    write()
    import utils.loss_utils as loss_utils
    from scene import GaussianModel,Scene
    from gaussian_renderer import render
    original_sample=loss_utils.sample_depth
    capture={}
    def sample(*a,**kw):
        value=original_sample(*a,**kw)
        capture['points']=a[0];capture['sampled']=value['sampled_depth']
        return value
    loss_utils.sample_depth=sample
    try:
        backends={v:load_backend(v) for v in ('stable2','stable1')}
        fields=('_xyz','_scaling','_rotation','_opacity')
        rows=[]
        for refrow in selected:
            model=Path(refrow['model']);ds,pipe=load_config(model)
            with tempfile.TemporaryDirectory(prefix='sampling-diagnostic-',dir=out) as temp:
                scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True)
                ds.model_path=str(scratch);gs=GaussianModel(ds.sh_degree,ds.sg_degree)
                scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
                cameras=scene.getTrainCameras();cam=next(c for c in cameras if c.image_name==refrow['camera'])
                near=cameras[cam.nearest_id[0]];gt=cam.original_image.cuda()
                bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
                pm=loss_utils.PatchMatch(3,1.,ds.kernel_size,pipe,debug=False,model_path=str(scratch))
                data={};upstreams={};sampled_outputs={}
                for v,backend in backends.items():
                    with activate(backend):
                        pkg=render(cam,gs,pipe,bg,ds.kernel_size,require_depth=True)
                        rgb=.8*loss_utils.l1_loss(pkg['render'],gt)+.2*(1-loss_utils.ssim(pkg['render'].unsqueeze(0),gt.unsqueeze(0)))
                        ncc,geo=pm(gs,pkg,cam,near,iteration=7001)
                        incoming=torch.autograd.grad(geo,capture['sampled'],retain_graph=True)[0]
                        upstreams[v]=incoming.detach().cpu()
                        sampled_outputs[v]=capture['sampled'].detach().cpu()
                        components={'rgb':rgb,'ncc':.6*ncc,'geometry':.02*geo,'total':rgb+.6*ncc+.02*geo}
                        per={}
                        for label,loss in components.items():
                            trials=[]
                            for _ in range(5):
                                grads=torch.autograd.grad(loss,[getattr(gs,f) for f in fields],retain_graph=True,allow_unused=True)
                                torch.cuda.synchronize()
                                trials.append({f:None if g is None else g.detach().cpu().double() for f,g in zip(fields,grads)})
                            per[label]=trials
                        # Detach the source-render query construction to isolate
                        # target sampling from the source renderer's backward.
                        isolated_points=capture['points'].detach().requires_grad_(True)
                        isolated_sample=original_sample(isolated_points,near,gs,pipe,ds.kernel_size)['sampled_depth']
                        if not torch.equal(isolated_sample.detach(),capture['sampled'].detach()):
                            raise ValueError('Isolated replay sampling output differs.')
                        direct=[]
                        for _ in range(5):
                            grads=torch.autograd.grad(isolated_sample,[getattr(gs,f) for f in fields]+[isolated_points],grad_outputs=incoming,retain_graph=True,allow_unused=True)
                            torch.cuda.synchronize()
                            direct.append({f:None if g is None else g.detach().cpu().double() for f,g in zip((*fields,'query_points'),grads)})
                        per['direct_sampling_geometry_upstream']=direct
                        data[v]=per
                        del pkg,rgb,ncc,geo,components,incoming,isolated_sample,isolated_points
                    capture.clear()
                comparisons={}
                for label in data['stable2']:
                    per={}
                    for f in data['stable2'][label][0]:
                        control=[g[f] for g in data['stable2'][label]];cand=[g[f] for g in data['stable1'][label]]
                        if control[0] is None:
                            if any(x is not None for x in (*control,*cand)):raise ValueError('Gradient participation mismatch.')
                            per[f]={'participates':False};continue
                        norm=max(float(torch.linalg.vector_norm(control[0])),1e-12)
                        def drift(a,b):return float(torch.linalg.vector_norm(a-b))/norm
                        pairwise_baseline=[drift(a,b) for i,a in enumerate(control) for b in control[i+1:]]
                        cross=[drift(a,b) for a in cand for b in control]
                        mean_control=torch.stack(control).mean(0);mean_cand=torch.stack(cand).mean(0)
                        per[f]={'participates':True,'control_gradient_l2':norm,
                            'baseline_repeat_pairwise_max_relative_l2':max(pairwise_baseline),
                            'candidate_control_cross_max_relative_l2':max(cross),
                            'candidate_mean_minus_control_mean_relative_l2':drift(mean_cand,mean_control),
                            'all_finite':all(bool(torch.isfinite(x).all()) for x in (*control,*cand))}
                    comparisons[label]=per
                row={'model':str(model),'camera':cam.image_name,'trials_per_backend_component':5,
                     'sampled_xyz_bitwise_equal':torch.equal(sampled_outputs['stable2'],sampled_outputs['stable1']),
                     'geometry_upstream_bitwise_equal':torch.equal(upstreams['stable2'],upstreams['stable1']),
                     'component_gradient_comparisons':comparisons}
                rows.append(row);(out/f'check_{model.name}_{cam.image_name}.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
                print(f'{model.name}/{cam.image_name} diagnostic complete',flush=True)
                del data,gs,scene,cameras;torch.cuda.empty_cache()
        (out/'results.json').write_text(json.dumps({'rows':rows,'limitations':[
            'Five backward replays of each fixed forward graph, not five independent training runs.',
            'Relative L2 uses first control gradient norm; means are exploratory sensitivity summaries.',
            'No revised quality or drift thresholds; original expanded audit remains failed.']},indent=2,allow_nan=False)+'\n')
        state['status']='complete'
    except BaseException as e:
        state['status']='failed';state['error']=repr(e);raise
    finally:
        loss_utils.sample_depth=original_sample;write()


if __name__=='__main__':main()
