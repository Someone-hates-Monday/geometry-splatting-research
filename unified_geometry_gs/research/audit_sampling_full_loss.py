"""Check complete RGB/PatchMatch gradients on the remaining reference cameras.

The six development cameras 0000/0027 are explicitly excluded. These are
additional frozen-camera checks, not independent scenes or test-set quality.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.benchmark_median_cuda import load_config
from research.sampling_batch_backend import activate, load_backend


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    state = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'building',
             'excluded_development_cameras': ['0000', '0027'], 'steps': [],
             'gradient_relative_l2_gate': .001,
             'limitations': ['Additional frozen cameras of the same three development scenes.',
                 'No optimizer, densification, appearance update or trajectory quality.',
                 'Finite differences are not a stable oracle for the hard PatchMatch masks.',
                 'A matching complete loss and small gradients drift are engineering checks, not exact floating-point equivalence.']}
    def write(): (out/'manifest.json').write_text(json.dumps(state, indent=2)+'\n')
    sources = [Path(__file__), ROOT/'research/sampling_batch_backend.py',
               ROOT/'research/median_cuda_backend.py', ROOT/'research/benchmark_median_cuda.py',
               ROOT/'gaussian_renderer/__init__.py', ROOT/'utils/loss_utils.py']
    raster = ROOT/'submodules/diff-gaussian-rasterization'
    for pattern in ('*.cu', '*.cpp', '*.h', 'cuda_rasterizer/*.cu', 'cuda_rasterizer/*.h'):
        sources.extend(raster.glob(pattern))
    state['source_sha256'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    with zipfile.ZipFile(out/'sources.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sources: z.write(p, p.relative_to(ROOT).as_posix())
    state['source_archive_sha256'] = sha(out/'sources.zip')
    write()
    try:
        backends = {v: load_backend(v) for v in ('control', 'batch1')}
        state['backend_sha256'] = {v: sha(Path(b.__file__)) for v, b in backends.items()}
        state['status'] = 'running'; write()
        from scene import GaussianModel, Scene
        from gaussian_renderer import render
        from utils.loss_utils import PatchMatch, l1_loss, ssim
        fields = ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc',
                  '_features_rest', '_sg_axis', '_sg_sharpness', '_sg_color')
        rows = []
        for cpu in args.cpu:
            ref = json.loads(cpu.read_text()); model = Path(ref['model'])
            ply = model/'point_cloud/iteration_30000/point_cloud.ply'
            if [v for k,v in ref['source_sha256'].items() if k.endswith('point_cloud.ply')] != [sha(ply)]:
                raise ValueError('Frozen model changed.')
            state.setdefault('input_sha256', {})[str(ply)] = sha(ply)
            ds, pipe = load_config(model)
            with tempfile.TemporaryDirectory(prefix='sampling-loss-', dir=out) as temp:
                scratch = Path(temp)
                (scratch/'point_cloud').symlink_to(model/'point_cloud', target_is_directory=True)
                ds.model_path = str(scratch)
                gs = GaussianModel(ds.sh_degree, ds.sg_degree)
                scene = Scene(ds, gs, load_iteration=30000, shuffle=False)
                cameras = scene.getTrainCameras(); lookup = {c.image_name:c for c in cameras}
                bg = torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.], device='cuda')
                pm = PatchMatch(3, 1., ds.kernel_size, pipe, debug=False, model_path=str(scratch))
                names = [n for n in dict.fromkeys(r['camera'] for r in ref['rows']) if n not in ('0000','0027')]
                for name in names:
                    cam = lookup[name]; near = cameras[cam.nearest_id[0]]
                    gt = cam.original_image.cuda()
                    gradients = {}; losses = {}
                    for v, backend in (('control',backends['control']), ('control_repeat',backends['control']), ('batch1',backends['batch1'])):
                        for f in fields: getattr(gs,f).grad = None
                        with activate(backend):
                            pkg = render(cam, gs, pipe, bg, ds.kernel_size, require_depth=True)
                            rgb = .8*l1_loss(pkg['render'],gt)+.2*(1-ssim(pkg['render'].unsqueeze(0),gt.unsqueeze(0)))
                            ncc, geo = pm(gs, pkg, cam, near, iteration=7001)
                            loss = rgb+.6*ncc+.02*geo
                            loss.backward(); torch.cuda.synchronize()
                        losses[v] = [float(x.detach()) for x in (rgb,ncc,geo,loss)]
                        gradients[v] = {f: None if getattr(gs,f).grad is None else getattr(gs,f).grad.detach().cpu().double() for f in fields}
                        del pkg, rgb, ncc, geo, loss
                    comparisons = {}
                    for f in fields:
                        x = gradients['control'][f]; y = gradients['batch1'][f]; z = gradients['control_repeat'][f]
                        if x is None:
                            if y is not None or z is not None: raise ValueError('Gradient participation differs.')
                            comparisons[f] = {'participates':False}; continue
                        if y is None or z is None: raise ValueError('Missing variant gradient.')
                        denom = max(float(torch.linalg.vector_norm(x)),1e-12)
                        comparisons[f] = {'participates': True,
                            'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(z).all()),
                            'relative_l2_drift':float(torch.linalg.vector_norm(y-x))/denom,
                            'control_repeat_relative_l2_drift':float(torch.linalg.vector_norm(z-x))/denom}
                    row = {'model':str(model), 'camera':name, 'nearest_camera':near.image_name,
                           'loss_order':['rgb','ncc','geometry','total'], 'losses':losses,
                           'loss_components_bitwise_equal':losses['control']==losses['batch1'],
                           'gradient_comparison':comparisons}
                    rows.append(row)
                    (out/f'check_{model.name}_{name}.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
                    max_drift = max(c['relative_l2_drift'] for c in comparisons.values() if c['participates'])
                    print(f'{model.name}/{name} losses_equal={row["loss_components_bitwise_equal"]} max_grad_drift={max_drift:.8g}',flush=True)
                    state['steps'].append({'model':model.name,'camera':name,'status':'complete'}); write()
                    del gradients
                del gs, scene, cameras
                torch.cuda.empty_cache()
        active = [c for r in rows for c in r['gradient_comparison'].values() if c['participates']]
        result = {'rows':rows,'participating_groups':len(active),
                  'all_loss_components_bitwise_equal':all(r['loss_components_bitwise_equal'] for r in rows),
                  'all_gradients_finite_and_drift_le_0_001':all(c['finite'] and c['relative_l2_drift']<=.001 for c in active),
                  'maximum_gradient_relative_l2_drift':max(c['relative_l2_drift'] for c in active),
                  'maximum_control_repeat_relative_l2_drift':max(c['control_repeat_relative_l2_drift'] for c in active),
                  'limitations':state['limitations']}
        (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        state['status']='complete'; state['gate_pass']=result['all_loss_components_bitwise_equal'] and result['all_gradients_finite_and_drift_le_0_001']
    except BaseException as e:
        state['status']='failed'; state['error']=repr(e); raise
    finally: write()


if __name__=='__main__': main()
