"""Fixed-input RGB VJP audit, with independent float64 and native finite differences.

No optimizer. Retains every parameter/step, including failed comparisons. The
detached-inradius reference is an explanatory control, not the correct gradient.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from tgs_native_diagnostic_backend import load_backend, ROOT

OUT = ROOT / 'experiments/tgs_backward_v1'
STEPS = [1e-2, 3e-3, 1e-3, 3e-4]
PARAMS = [f'p{i}_{j}' for i in range(3) for j in 'xyz'] + ['sigma', 'opacity']


def fixtures():
    for name, w, h, reverse, tilted in [
        ('front', 256, 256, False, False),
        ('reverse', 256, 256, True, False),
        ('rectangular', 320, 192, False, False),
        ('camera_tilt', 320, 192, False, True),
    ]:
        xy = np.array([[40.25, 40.25], [200.25, 40.25], [40.25, 160.25]])
        xy *= [w / 256, h / 256]
        z = np.array([1.7, 2.4, 2.1]) if tilted else np.full(3, 2.)
        cam_points = np.c_[((2 * xy + 1) / [w, h] - 1) * z[:, None], z]
        view = np.eye(4)
        if tilted:
            c, s = np.cos(.23), np.sin(.23)
            view[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
            view[:3, 3] = [-.31, .17, .26]
        world = (np.c_[cam_points, np.ones(3)] @ np.linalg.inv(view).T)[:, :3]
        if reverse:
            world = world[[0, 2, 1]]
        projection = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0], [0, 0, 1, 0]]) @ view
        yield dict(name=name, w=w, h=h, view=view.astype(np.float32),
                   projection=projection.astype(np.float32),
                   theta=np.r_[world.ravel(), 1.3, .65].astype(np.float32),
                   pixels=np.rint(np.array([[72, 65], [135, 55], [55, 105], [104, 91]])
                                  * [w / 256, h / 256]).astype(int),
                   weights=np.array([1., .7, -.4, 1.1]),
                   color=np.array([.7, .4, .2], dtype=np.float32),
                   channel_weights=np.array([1., -.3, .5], dtype=np.float32))


def reference(theta, f, detach_radius=False, ndc_gradient_control=False):
    p = theta[:9].reshape(3, 3)
    projection = torch.tensor(f['projection'], dtype=torch.float64)
    hom = torch.cat([p, torch.ones((3, 1), dtype=torch.float64)], dim=1) @ projection.T
    ndc = hom[:, :2] / (hom[:, 3:] + 1e-7)
    scale = torch.tensor([f['w'] / 2, f['h'] / 2], dtype=torch.float64)
    xy = (ndc + 1) * scale - .5
    if ndc_gradient_control:
        # Forward unchanged; remove pixel Jacobian to reproduce the suspected
        # native approximation, only as a cause-isolation control.
        xy = xy.detach() + ndc - ndc.detach()
    edge = xy.roll(-1, 0) - xy
    length = torch.linalg.vector_norm(edge, dim=1)
    opposite = length.roll(-1, 0)
    center = (opposite[:, None] * xy).sum(0) / length.sum()
    normals = torch.stack([edge[:, 1], -edge[:, 0]], dim=1) / length[:, None]
    offsets = -(normals * xy).sum(1)
    sign = torch.where(normals @ center + offsets > 0, -1., 1.)
    normals = normals * sign[:, None]
    offsets = offsets * sign
    radius = -(normals[-1] @ center + offsets[-1])
    if detach_radius:
        radius = radius.detach()
    pixels = torch.tensor(f['pixels'], dtype=torch.float64)
    phi = pixels @ normals.T + offsets
    vals, active = phi.max(dim=1)
    u = -vals / radius
    alpha = theta[-1] * u.pow(theta[-2])
    # All selected values must be on this smooth, unclipped branch.
    coeff = float(np.dot(f['color'].astype(float), f['channel_weights'].astype(float)))
    loss = (alpha * torch.tensor(f['weights']) * coeff).sum()
    sorted_phi = phi.sort(dim=1, descending=True).values
    margins = dict(active_edges=active.tolist(), phi=phi.detach().tolist(),
                   active_edge_gap_min=float((sorted_phi[:, 0] - sorted_phi[:, 1]).min().detach()),
                   distance_to_boundary_min=float((-vals).min().detach()),
                   inradius=float(radius.detach()),
                   alpha_min=float(alpha.min().detach()), alpha_max=float(alpha.max().detach()),
                   projected=xy.detach().tolist(), u=u.detach().tolist())
    return loss, alpha, margins


class Native:
    def __init__(self, mod, f):
        self.mod, self.f = mod, f
        self.t = lambda x, dtype=torch.float32: torch.tensor(x, dtype=dtype, device='cuda').contiguous()
        self.bg = self.t([0, 0, 0])
        self.view = self.t(f['view'].T.copy())
        self.proj = self.t(f['projection'].T.copy())
        self.count = self.t([3], torch.int32)
        self.offset = self.t([0], torch.int32)
        self.color = self.t(f['color'][None])
        self.empty = self.t([])
        self.zero = self.t([0])
        grad = np.zeros((3, f['h'], f['w']), dtype=np.float32)
        for (x, y), weight in zip(f['pixels'], f['weights']):
            grad[:, y, x] = weight * f['channel_weights']
        self.grad = self.t(grad)
        self.grad_others = self.t(np.zeros((7, f['h'], f['w'])))

    def forward(self, theta):
        t, f = self.t, self.f
        p, sigma, opacity = t(theta[:9].reshape(1, 3, 3)), t(theta[9:10]), t(theta[10:11])
        res = self.mod.rasterize_triangles(
            self.bg, p, sigma, self.count, self.offset, self.color, opacity,
            self.zero, self.zero, self.view, self.proj, 1, 1., 1., f['h'], f['w'],
            self.empty, 0, t([0, 0, 0]), False, True)
        torch.cuda.synchronize()
        # Double reduction isolates rasterizer float32 errors from loss summation.
        loss = float((res[1].double() * self.grad.double()).sum().cpu())
        samples = np.array([res[1][:, y, x].cpu().numpy() for x, y in f['pixels']])
        return loss, res, p, sigma, samples

    def backward(self, data):
        loss, res, p, sigma, samples = data
        g = self.mod.rasterize_triangles_backward(
            self.bg, p, sigma, self.count, self.offset, res[3], self.color,
            self.view, self.proj, 1, 1., 1., self.grad, self.grad_others,
            self.empty, 0, self.t([0, 0, 0]), res[4], res[0], res[5], res[6], True)
        torch.cuda.synchronize()
        return np.r_[g[0].cpu().numpy().ravel(), g[1].cpu().numpy().ravel(),
                     g[3].cpu().numpy().ravel()], [x.cpu().numpy() for x in g]


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), 1e-12))


def run(mod, variant):
    target = OUT / variant
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for f in fixtures():
        theta = f['theta']
        grad_refs = []
        for detach, ndc_control in [(False, False), (True, False), (True, True)]:
            x = torch.tensor(theta.astype(float), requires_grad=True)
            loss, alpha, margins = reference(x, f, detach, ndc_control)
            grad_refs.append(torch.autograd.grad(loss, x)[0].numpy())
        true, stopped, suspected = grad_refs
        x = torch.tensor(theta.astype(float), requires_grad=True)
        cpu_loss, cpu_alpha, margins = reference(x, f)
        assert margins['active_edge_gap_min'] > 2 and margins['distance_to_boundary_min'] > 4
        assert margins['alpha_min'] > .02 and margins['alpha_max'] < .9
        native = Native(mod, f)
        data = native.forward(theta)
        gpu, all_grads = native.backward(data)
        repeat = [native.backward(native.forward(theta))[0] for _ in range(3)]
        repeat_max = max(float(np.max(np.abs(g - gpu))) for g in repeat)
        fd = []
        for h in STEPS:
            rows = []
            for j, label in enumerate(PARAMS):
                plus, minus = theta.copy(), theta.copy()
                plus[j] += h
                minus[j] -= h
                delta = float(plus[j]) - float(minus[j])
                lp, rp, _, _, sp = native.forward(plus)
                lm, rm, _, _, sm = native.forward(minus)
                cp, _, mp = reference(torch.tensor(plus.astype(float)), f)
                cm, _, mm = reference(torch.tensor(minus.astype(float)), f)
                smooth = (mp['active_edges'] == mm['active_edges'] == margins['active_edges']
                          and min(mp['active_edge_gap_min'], mm['active_edge_gap_min']) > 1
                          and min(mp['alpha_min'], mm['alpha_min']) > .01
                          and max(mp['alpha_max'], mm['alpha_max']) < .95
                          and int(rp[3][0]) > 0 and int(rm[3][0]) > 0)
                rows.append(dict(parameter=label, step=h, actual_delta=delta,
                                 native_plus_loss=lp, native_minus_loss=lm,
                                 cpu_plus_loss=float(cp), cpu_minus_loss=float(cm),
                                 native_fd=(lp - lm) / delta,
                                 cpu_fd=float(cp - cm) / delta, smooth=smooth,
                                 plus_samples=sp.tolist(), minus_samples=sm.tolist(),
                                 plus_margins=mp, minus_margins=mm))
            fd.append(dict(step=h, rows=rows,
                           native_vs_true_relative=relative([r['native_fd'] for r in rows], true),
                           cpu_vs_true_relative=relative([r['cpu_fd'] for r in rows], true)))
        row = dict(name=f['name'], native_loss=data[0], cpu_loss=float(cpu_loss.detach()),
                   forward_sample_max_abs=float(np.max(np.abs(data[-1] - cpu_alpha.detach().numpy()[:, None] * f['color']))),
                   repeat_gradient_max_abs=repeat_max, parameters=PARAMS,
                   native_gradient=gpu.tolist(), exact_gradient=true.tolist(),
                   detached_radius_gradient=stopped.tolist(), suspected_native_gradient=suspected.tolist(),
                   geometry_relative_error=relative(gpu[:9], true[:9]),
                   sigma_opacity_relative_error=relative(gpu[9:], true[9:]),
                   geometry_vs_suspected_relative=relative(gpu[:9], suspected[:9]),
                   geometry_vs_detached_radius_relative=relative(gpu[:9], stopped[:9]),
                   margins=margins, finite_differences=fd,
                   all_perturbations_smooth=all(r['smooth'] for s in fd for r in s['rows']),
                   native_radius=int(data[1][3][0]), native_inradius=float(data[1][8][0]))
        records.append(row)
        np.savez_compressed(target / (f['name'] + '.npz'),
                            **f, native_rgb=data[1][1].cpu().numpy(),
                            native_aux=data[1][2].cpu().numpy(),
                            **{f'native_grad_{i}': g for i, g in enumerate(all_grads)})
        (target / 'progress.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps({k: row[k] for k in ['name', 'forward_sample_max_abs', 'geometry_relative_error',
                     'sigma_opacity_relative_error', 'geometry_vs_suspected_relative', 'all_perturbations_smooth']}), flush=True)
    summary = dict(variant=variant, module_file=mod.__file__,
                   module_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   scope='single triangle smooth RGB branch, precomputed color; no optimizer or training',
                   steps=STEPS, fixtures=records,
                   geometry_all_below_1e_3=all(r['geometry_relative_error'] < 1e-3 for r in records),
                   sigma_opacity_all_below_1e_3=all(r['sigma_opacity_relative_error'] < 1e-3 for r in records))
    (target / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--scale-only', action='store_true')
    args = parser.parse_args()
    if args.fixed or args.scale_only:
        from tgs_backward_fixed_backend import load_fixed
        run(load_fixed(radius=not args.scale_only), 'scale_only' if args.scale_only else 'geometry_fixed')
    else:
        run(load_backend(shfix=True), 'shfix_control')
