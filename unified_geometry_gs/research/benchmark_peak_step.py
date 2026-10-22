"""Counterbalanced frozen forward/backward latency; no optimizer or NCC step."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.benchmark_median_cuda import load_config
from research.median_peak_adaptive_backend import activate,load_backend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',nargs='+',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=10)
    args=parser.parse_args()
    if args.out.exists():
        parser.error('Fresh output required.')
    gate=json.loads(args.after.read_text())
    if not gate.get('all_groups_drift_below_threshold') or not gate.get('all_gradients_finite'):
        raise ValueError('Fixed-camera gradient gate not passed.')
    from scene import GaussianModel,Scene
    from gaussian_renderer import render
    backends={v:load_backend(v) for v in ('control','adjquad4','peakadj')}
    fields=('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity',
            '_sg_axis','_sg_sharpness','_sg_color')
    rng=np.random.default_rng(0)
    rows=[]
    sources={}
    for cpu in args.cpu:
        ref=json.loads(cpu.read_text())
        model=Path(ref['model'])
        ply=model/'point_cloud/iteration_30000/point_cloud.ply'
        digest=hashlib.sha256(ply.read_bytes()).hexdigest()
        if [v for k,v in ref['source_sha256'].items() if k.endswith('point_cloud.ply')]!=[digest]:
            raise ValueError('Frozen model changed.')
        sources[str(ply)]=digest
        ds,pipe=load_config(model)
        with tempfile.TemporaryDirectory(prefix='step-benchmark-',dir=args.out.parent.resolve()) as temp:
            scratch=Path(temp)
            (scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True)
            ds.model_path=str(scratch)
            gs=GaussianModel(ds.sh_degree,ds.sg_degree)
            scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
            cams={c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()}
            bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
            names=list(dict.fromkeys(r['camera'] for r in ref['rows']))[::4]
            for name in names:
                cam=cams[name]
                def clear():
                    for field in fields:
                        getattr(gs,field).grad=None
                def step(backend):
                    with activate(backend):
                        out=render(cam,gs,pipe,bg,ds.kernel_size,require_depth=True)
                        # All image pixels: measures full depth-adjoint workload.
                        loss=out['render'].mean()+.01*out['median_depth'].mean()
                        loss.backward()
                for backend in backends.values():
                    clear(); step(backend)
                torch.cuda.synchronize()
                measurements={v:[] for v in backends}
                orders=[]
                for _ in range(args.rounds):
                    order=list(backends)
                    rng.shuffle(order)
                    orders.append(order)
                    for variant in order:
                        events=[]
                        for _ in range(args.repeats):
                            clear()
                            start=torch.cuda.Event(enable_timing=True)
                            end=torch.cuda.Event(enable_timing=True)
                            start.record(); step(backends[variant]); end.record()
                            events.append((start,end))
                        torch.cuda.synchronize()
                        measurements[variant].append(statistics.mean(a.elapsed_time(b) for a,b in events))
                baseline=statistics.median(measurements['control'])
                for v,values in measurements.items():
                    rows.append({'model':str(model),'camera':name,'variant':v,'gpu_ms_rounds':values,
                        'gpu_ms_median':statistics.median(values),'latency_ratio_to_control':statistics.median(values)/baseline,
                        'round_orders':orders})
                print(f'{model.name}/{name}: '+', '.join(f'{v}={statistics.median(x):.3f}ms' for v,x in measurements.items()),flush=True)
            clear()
            del gs,scene,cams
            torch.cuda.empty_cache()
    for p in (Path(__file__),args.after,ROOT/'research/median_adjquad_backend.py',ROOT/'research/median_peak_adaptive_backend.py',
              ROOT/'research/median_adjquad_accumulate.inc',ROOT/'research/median_adjquad_refine.inc'):
        sources[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    for backend in backends.values():
        p=Path(backend.__file__)
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'rounds':args.rounds,'repeats':args.repeats,
        'rows':rows,'source_sha256':sources,
        'limitations':['Frozen full-image RGB-plus-depth forward/backward, no optimizer, densification, PatchMatch, or NCC.',
        'Six fixed development cameras, one GPU and one measurement session; not full-training speed or quality.',
        'Gradient clearing is outside each timed interval; GPU events include intervening host launch gaps.',
        'Five rounds and cameras are not independent training-seed replications.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
