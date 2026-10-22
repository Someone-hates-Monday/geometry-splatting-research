"""Full-image frozen affine/packet/factor comparison with serial paired timing."""
from __future__ import annotations
import argparse,hashlib,json,os,random,tempfile,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from research.benchmark_median_cuda import load_config

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('affine','exact_packet','exact_factor')
FIELDS=('render','mask','normal','median_depth','interface_slope')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def quantiles(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return {k:float(v) for k,v in zip(('min','p50','p90','p99','max'),np.quantile(x,(0,.5,.9,.99,1)))} if x.size else None

def execute(out,state,views):
    from research.exact_perspective_backend import load_backend,activate,metadata
    from scene import GaussianModel,Scene
    from gaussian_renderer import render
    torch.set_num_threads(1);build_start=time.perf_counter();backends={};binaries={};generated={}
    for variant in VARIANTS:
        print('Building '+variant,flush=True);backends[variant]=load_backend(variant)
        binary=Path(backends[variant].__file__);binaries[variant]={'path':str(binary),'sha256':sha(binary),'metadata':metadata(backends[variant])}
        files=[p for p in binary.parent.rglob('*') if p.is_file() and (p.suffix in ('.cu','.cpp','.h','.inc') or p.name=='build.ninja')]
        generated[variant]={str(p):sha(p) for p in files}
    state.update(status='rendering',binaries=binaries,generated_sha256=generated,build_seconds=time.perf_counter()-build_start)
    save(out/'manifest.json',state);rows=[];rng=random.Random(13018)
    for scan in (37,55,63):
        ref=read(ROOT/f'research/results/median_cpu_scan{scan}.json');model=ROOT/f'output_compare/ggs_pipeline/scan{scan}'
        ply=model/'point_cloud/iteration_30000/point_cloud.ply';expected=next(v for k,v in ref['source_sha256'].items() if k.endswith('point_cloud.ply'))
        assert sha(ply)==expected
        ds,pipe=load_config(model)
        names=list(dict.fromkeys(r['camera'] for r in ref['rows']))
        assert len(names)==8
        if views:names=names[:views]
        with tempfile.TemporaryDirectory(prefix='frozen-exact-scene-',dir=out) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch)
            gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
            cameras={c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()}
            bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
            for name in names:
                cam=cameras[name];frames={};memory={};nonfinite={};timing={v:[] for v in VARIANTS};orders=[]
                with torch.no_grad():
                    for variant in VARIANTS:
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();allocated_before=torch.cuda.memory_allocated()
                        with activate(backends[variant]):pkg=render(cam,gs,pipe,bg,ds.kernel_size,require_depth=True)
                        torch.cuda.synchronize()
                        memory[variant]={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                                         'allocated_before_bytes':allocated_before,'peak_added_allocated_bytes':torch.cuda.max_memory_allocated()-allocated_before}
                        frames[variant]={k:pkg[k].detach().cpu().numpy() for k in FIELDS}
                        nonfinite[variant]=sum(int(np.sum(~np.isfinite(a))) for a in frames[variant].values())
                        del pkg
                    for variant in VARIANTS:
                        with activate(backends[variant]):
                            for _ in range(5):render(cam,gs,pipe,bg,ds.kernel_size,require_depth=True)
                    torch.cuda.synchronize()
                    for round_id in range(3):
                        order=list(VARIANTS);rng.shuffle(order);orders.append(order)
                        for variant in order:
                            torch.cuda.synchronize();start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2));wall=time.perf_counter()
                            with activate(backends[variant]):
                                start.record()
                                for _ in range(10):render(cam,gs,pipe,bg,ds.kernel_size,require_depth=True)
                                end.record();end.synchronize()
                            timing[variant].append({'round':round_id,'gpu_ms_per_render':start.elapsed_time(end)/10,
                                                    'wall_ms_per_render':(time.perf_counter()-wall)*1000/10})
                base=frames['affine'];stats={};arrays={}
                for variant in VARIANTS:
                    frame=frames[variant];depth=frame['median_depth'].squeeze();valid=depth>0;basevalid=base['median_depth'].squeeze()>0
                    common=valid&basevalid
                    stats[variant]={'valid_pixels':int(sum(valid.ravel())),'valid_fraction':float(np.mean(valid)),
                        'negative_depth_pixels':int(sum((depth<0).ravel())),'negative_alpha_pixels':int(sum((frame['mask']<0).ravel())),
                        'nonfinite_output_components':nonfinite[variant],
                        'depth_abs_difference_vs_affine':quantiles(abs(depth[common]-base['median_depth'].squeeze()[common])),
                        'validity_mismatches_vs_affine':int(sum((valid!=basevalid).ravel())),
                        'rgb_max_abs_difference_vs_affine':float(np.max(abs(frame['render']-base['render']))),
                        'alpha_max_abs_difference_vs_affine':float(np.max(abs(frame['mask']-base['mask']))),
                        'normal_max_abs_difference_vs_affine':float(np.max(abs(frame['normal']-base['normal']))),
                        'gpu_ms_median':float(np.median([r['gpu_ms_per_render'] for r in timing[variant]])),
                        'wall_ms_median':float(np.median([r['wall_ms_per_render'] for r in timing[variant]])),
                        'timing_rounds':timing[variant],'memory':memory[variant]}
                    arrays[variant+'_depth']=frame['median_depth'];arrays[variant+'_slope']=frame['interface_slope']
                    for k in ('render','mask','normal'):
                        arrays[variant+'_'+k+('_delta' if variant!='affine' else '')]=frame[k]-base[k] if variant!='affine' else frame[k]
                b=frames['exact_packet']['median_depth'].squeeze();c=frames['exact_factor']['median_depth'].squeeze();common=(b>0)&(c>0)
                error=quantiles(abs(b[common]-c[common]));maskdiff=float(np.mean((b>0)!=(c>0)))
                gate=(all(s['nonfinite_output_components']==s['negative_depth_pixels']==s['negative_alpha_pixels']==0 for s in stats.values())
                    and all(s[k]<=1e-5 for s in stats.values() for k in ('rgb_max_abs_difference_vs_affine','alpha_max_abs_difference_vs_affine','normal_max_abs_difference_vs_affine'))
                    and maskdiff<=1e-4 and error is not None and error['p99']<=1e-4 and error['max']<=1e-3)
                row={'scan':scan,'camera':name,'width':cam.image_width,'height':cam.image_height,'variants':stats,'packet_factor_depth_abs_error':error,
                     'packet_factor_validity_mismatch_fraction':maskdiff,'numerical_gate_pass':bool(gate),'timing_orders':orders,
                     'packet_factor_slope_abs_difference_descriptive':quantiles(abs(frames['exact_packet']['interface_slope']-frames['exact_factor']['interface_slope'])),
                     'slope_metric_scope':'Added descriptively after smoke observed slope discrepancy; no slope equivalence gate or gradient claim.'}
                np.savez_compressed(out/f'scan{scan}_{name}.npz',**arrays)
                rows.append(row);save(out/f'scan{scan}_{name}.json',row);save(out/'progress.json',{'utc':now(),'completed_views':len(rows),'last_scan':scan,'last_camera':name})
                print(f'scan{scan}/{name}: numerical={gate}, packet/factor ms={stats["exact_packet"]["gpu_ms_median"]:.3f}/{stats["exact_factor"]["gpu_ms_median"]:.3f}',flush=True)
            del scene,gs,cameras;torch.cuda.empty_cache()
        assert sha(ply)==expected
    ratios=[r['variants']['exact_packet']['gpu_ms_median']/r['variants']['exact_factor']['gpu_ms_median'] for r in rows]
    gm=float(np.exp(np.mean(np.log(ratios))));numerical=all(r['numerical_gate_pass'] for r in rows)
    summary={'status':'complete','views':len(rows),'all_prespecified_24_views':len(rows)==24,'numerical_gate_pass':numerical,
             'packet_factor_render_gpu_ratio_geomean':gm,'packet_factor_render_gpu_ratio_max':max(ratios),
             'efficiency_gate_pass':len(rows)==24 and numerical and gm<=.90 and max(ratios)<=1.05,
             'rows':rows,'limits':['Depth functional only under legacy affine selection, normal, initialization, bracket and cap.',
                'NCC/sample/integrate and backward are explicitly unsupported; no training throughput claim.',
                'Existing development scenes/views, no independent GT or novel-view improvement evidence.',
                'GPU event batches may contain host launch gaps; synchronous wall also reported.']}
    save(out/'summary.json',summary)
    for name,digest in state['source_sha256'].items():assert sha(ROOT/name)==digest,name
    state.update(status='complete',completed_utc=now(),numerical_gate_pass=numerical,efficiency_gate_pass=summary['efficiency_gate_pass'],
                 artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',state);print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--views-per-scene',type=int,default=0);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise ValueError('Fresh output required; poll existing handle after timeouts')
    if args.views_per_scene not in (0,1):raise ValueError('Only full24 or separately labelled3-view smoke')
    sources=[Path(__file__),ROOT/'research/FROZEN_EXACT_RENDER_PROTOCOL.md',ROOT/'research/benchmark_median_cuda.py',ROOT/'gaussian_renderer/__init__.py']
    sources+=sorted(p for p in (ROOT/'research').glob('exact_perspective_*') if p.suffix in ('.py','.inc','.cu','.cuh'))
    sources+=sorted((ROOT/'scene').glob('*.py'))
    sources+=[ROOT/f'research/results/median_cpu_scan{scan}.json' for scan in (37,55,63)]
    out.mkdir(parents=True);state={'created_utc':now(),'status':'building','pid':os.getpid(),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},'phase':'A_full24' if not args.views_per_scene else 'A_smoke3'}
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.writestr(p.relative_to(ROOT).as_posix(),p.read_bytes())
    try:execute(out,state,args.views_per_scene)
    except BaseException as exc:
        state.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc));save(out/'manifest.json',state)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
