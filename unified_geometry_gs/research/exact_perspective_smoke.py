"""One complete frozen image; no GT quality, training, timing benchmark or sample paths."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,tempfile,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from research.exact_perspective_backend import ROOT,RASTER,VARIANTS,load_backend,activate,metadata,payloads
from research.benchmark_median_cuda import load_config

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def compare(a,b):
    delta=abs(a-b);valid=np.isfinite(delta)
    return {'bitwise_equal':bool(np.array_equal(a,b)),'max_absolute_finite':float(np.max(delta[valid])) if np.any(valid) else None,
        'nonfinite_difference_components':int(sum((~valid).ravel()))}

def execute(out,state):
    torch.set_num_threads(1)
    models={}
    for variant in VARIANTS:
        print(f'building {variant}',flush=True);models[variant]=load_backend(variant)
        rec=metadata(models[variant]);save(out/f'backend_{variant}.json',rec)
        state.setdefault('completed_builds',[]).append(variant);save(out/'manifest.json',state)
    from scene import GaussianModel,Scene
    from gaussian_renderer import render
    import diff_gaussian_rasterization as rasterizer
    installed=rasterizer._C
    state['installed_binary_sha256']=sha(Path(installed.__file__));state['installed_binary_path']=str(installed.__file__)
    state['status']='rendering';save(out/'manifest.json',state)
    model=ROOT/'output_compare/ggs_pipeline/scan37';ds,pipe=load_config(model);images={};outputs={}
    with tempfile.TemporaryDirectory(prefix='scene-',dir=out) as temp:
        scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch)
        gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
        cam=next(c for c in scene.getTrainCameras() if c.image_name=='0000')
        bg=torch.tensor([1,1,1] if ds.white_background else [0,0,0],dtype=torch.float32,device='cuda')
        fields=('render','mask','normal','median_depth','interface_slope','radii')
        guards={}
        try:
            with activate(models['exact_packet']):pass
        except RuntimeError:guards['grad_context_rejected']=True
        for method in ('rasterize_gaussians_backward','sample_rasterized_depth','sample_rasterized_depth_backward','integrate_gaussians_to_points','evaluate_sdf_from_signle_view'):
            try:getattr(models['exact_packet'],method)
            except RuntimeError:guards[method+'_rejected']=True
        state['guards']=guards;assert len(guards)==6 and all(guards.values())
        with torch.no_grad():
            for name in ('installed',)+VARIANTS:
                if name=='installed':result=render(cam,gs,pipe,bg,ds.kernel_size,require_depth=True)
                else:
                    with activate(models[name]):result=render(cam,gs,pipe,bg,ds.kernel_size,require_depth=True)
                torch.cuda.synchronize();arrays={key:result[key].detach().cpu().numpy() for key in fields}
                np.savez_compressed(out/f'image_{name}.npz',**arrays);images[name]=arrays
                outputs[name]={'nonfinite_components':{key:int(sum((~np.isfinite(value)).ravel())) for key,value in arrays.items()},
                    'positive_depth_pixels':int(sum((arrays['median_depth']>0).ravel()))}
                print(f'rendered {name}, depth valid={outputs[name]["positive_depth_pixels"]}',flush=True)
                save(out/'progress.json',{'utc':now(),'completed_images':list(images),'outputs':outputs})
    assert rasterizer._C is installed
    comparisons={}
    for a,b in (('installed','affine'),('affine','exact_packet'),('affine','exact_factor'),('exact_packet','exact_factor')):
        pair={key:compare(images[a][key],images[b][key]) for key in fields}
        da=images[a]['median_depth'];db=images[b]['median_depth'];ma=da>0;mb=db>0;common=ma&mb
        pair['depth_mask_disagreement_pixels']=int(sum((ma!=mb).ravel()));pair['depth_common_positive_pixels']=int(sum(common.ravel()))
        pair['max_relative_depth_error_common_positive']=float(np.max(abs(da[common]-db[common])/np.maximum(1e-4,abs(db[common])))) if np.any(common) else None
        comparisons[a+'_vs_'+b]=pair
    preserved=all(comparisons['affine_vs_'+variant][key]['bitwise_equal'] for variant in VARIANTS[1:] for key in ('render','mask','normal','radii'))
    control=all(comparisons['installed_vs_affine'][key]['bitwise_equal'] for key in fields)
    p= comparisons['exact_packet_vs_exact_factor'];finite=not any(v for row in outputs.values() for v in row['nonfinite_components'].values())
    agreement=finite and p['depth_mask_disagreement_pixels']==0 and p['max_relative_depth_error_common_positive'] is not None and p['max_relative_depth_error_common_positive']<=1e-3
    summary={'scan':37,'camera':'0000','images':4,'outputs':outputs,'comparisons':comparisons,'legacy_rgb_alpha_normal_radii_bitwise_preserved':preserved,
        'installed_control_bitwise_pass':control,'all_outputs_finite':finite,'packet_factor_development_gate_pass':agreement,
        'overall_gate_pass':preserved and control and agreement,'scope':'Complete image depth functional only under legacy affine selection; no sampled NCC, normal consistency, backward, timing or GT quality claim.'}
    save(out/'summary.json',summary)
    for name,digest in state['original_source_sha256'].items():assert sha(ROOT/name)==digest,name
    state.update(status='complete',completed_utc=now(),overall_gate_pass=summary['overall_gate_pass'],
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',state);print(json.dumps({k:v for k,v in summary.items() if k not in ('outputs','comparisons')},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if out.exists():raise ValueError('Fresh output required; preserve failures and observation handles.')
    out.mkdir(parents=True);sources=[Path(__file__),ROOT/'research/exact_perspective_backend.py',ROOT/'research/exact_perspective_ray.inc',
        ROOT/'research/exact_perspective_factor.inc',ROOT/'research/EXACT_PERSPECTIVE_INTEGRATION.md',ROOT/'research/benchmark_median_cuda.py',
        ROOT/'gaussian_renderer/__init__.py',ROOT/'scene/cameras.py',ROOT/'scene/gaussian_model.py']
    sources.extend(p for p in RASTER.rglob('*') if p.is_file() and p.suffix in ('.cu','.cpp','.h','.hpp','.inl','.py'))
    sources.extend([ROOT/'output_compare/ggs_pipeline/scan37/cfg_args',ROOT/'output_compare/ggs_pipeline/scan37/cameras.json'])
    state={'created_utc':now(),'status':'building','pid':os.getpid(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'CUDA_HOME':os.environ.get('CUDA_HOME'),'CXX':os.environ.get('CXX'),'CC':os.environ.get('CC'),'MAX_JOBS':os.environ.get('MAX_JOBS'),
        'original_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources}}
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for source in sources:z.write(source,source.relative_to(ROOT).as_posix())
        for variant in VARIANTS:
            for n,v in payloads(variant).items():z.writestr('generated/'+variant+'/'+n,v)
    try:execute(out,state)
    except BaseException as exc:
        state.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc));save(out/'manifest.json',state)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
