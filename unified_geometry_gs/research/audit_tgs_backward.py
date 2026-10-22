"""Independent NumPy closed-form audit of the retained TGS RGB VJP evidence.

Runs without Torch/CUDA. Checks the area/perimeter derivative rather than
reusing the probe's autograd incenter derivative.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/tgs_backward_v1'
VARIANTS = ['shfix_control', 'scale_only', 'geometry_fixed']


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def formula(f, theta=None, radius_derivative=True, pixel_jacobian=True):
    theta = f['theta'].astype(float) if theta is None else theta.astype(float)
    points, sigma, opacity = theta[:9].reshape(3, 3), theta[9], theta[10]
    projection = f['projection'].astype(float)
    h = np.c_[points, np.ones(3)] @ projection.T
    v = h[:, 3] + 1e-7
    scales = np.array([float(f['w']) / 2, float(f['h']) / 2])
    xy = (h[:, :2] / v[:, None] + 1) * scales - .5
    edges = np.roll(xy, -1, 0) - xy
    length = np.linalg.norm(edges, axis=1)
    perimeter = length.sum()
    D = np.linalg.det(np.stack([xy[1] - xy[0], xy[2] - xy[0]]))
    radius = abs(D) / perimeter
    R = np.array([[0., 1.], [-1., 0.]])
    n = edges @ R.T / length[:, None]
    tau = np.sign(D)  # CCW: R(edge) is outward.
    normals = tau * n
    offsets = -(normals * xy).sum(1)
    phi_all = f['pixels'] @ normals.T + offsets
    active = phi_all.argmax(1)
    phi = phi_all[np.arange(len(active)), active]
    u = -phi / radius
    alpha = opacity * u ** sigma
    coeff = f['color'].astype(float) @ f['channel_weights'].astype(float)
    g_alpha = f['weights'].astype(float) * coeff
    screen_grad = np.zeros((3, 2))
    for pixel, k, gp in zip(f['pixels'], active, g_alpha * sigma * alpha / phi):
        # phi = tau * (R e / |e|) dot (pixel-a). Differentiate e and a.
        de = tau * R.T @ (np.eye(2) - np.outer(n[k], n[k])) @ (pixel - xy[k]) / length[k]
        screen_grad[k] += gp * (-de - tau * n[k])
        screen_grad[(k + 1) % 3] += gp * de
    dlogr = np.zeros((3, 2))
    for i in range(3):
        nxt, prev = (i + 1) % 3, (i + 2) % 3
        dD = np.array([xy[nxt, 1] - xy[prev, 1], xy[prev, 0] - xy[nxt, 0]])
        dP = (xy[i] - xy[nxt]) / length[i] + (xy[i] - xy[prev]) / length[prev]
        dlogr[i] = dD / D - dP / perimeter
    if radius_derivative:
        screen_grad -= np.sum(g_alpha * sigma * alpha) * dlogr
    world_grad = []
    for i in range(3):
        J = projection[:2, :3] / v[i] - np.outer(h[i, :2], projection[3, :3]) / v[i] ** 2
        if pixel_jacobian:
            J *= scales[:, None]
        world_grad.append(J.T @ screen_grad[i])
    scalar_grad = [np.sum(g_alpha * alpha * np.log(u)), np.sum(g_alpha * u ** sigma)]
    return np.r_[np.ravel(world_grad), scalar_grad], float(g_alpha @ alpha), active, radius


def source_path(s):
    prefix = '/mnt/c/Users/Someone/Desktop/SuGaR-main/'
    return ROOT.parents[1] / s[len(prefix):] if s.startswith(prefix) else Path(s)


def main():
    source = ROOT / 'sources/tgs_native_diagnostic_v1'
    manifest = json.loads((source / 'manifest.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((source / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    results = {v: json.loads((OUT / v / 'results.json').read_text()) for v in VARIANTS}
    current_script = (ROOT / 'probe_tgs_backward.py').read_text()
    # The first baseline ran before adding a reporting-only metric and the
    # scale-only CLI switch. Reconstruct and retain its exact executed source.
    baseline_script = current_script.replace(
        '                   geometry_vs_detached_radius_relative=relative(gpu[:9], stopped[:9]),\n', '').replace(
        "    parser.add_argument('--scale-only', action='store_true')\n", '').replace(
        '    if args.fixed or args.scale_only:', '    if args.fixed:').replace(
        "        run(load_fixed(radius=not args.scale_only), 'scale_only' if args.scale_only else 'geometry_fixed')",
        "        run(load_fixed(), 'geometry_fixed')")
    assert hashlib.sha256(baseline_script.encode()).hexdigest() == results['shfix_control']['script_sha256']
    (OUT / 'shfix_control' / 'executed_probe_snapshot.py').write_bytes(baseline_script.encode())
    for v in ['scale_only', 'geometry_fixed']:
        assert hashlib.sha256(current_script.encode()).hexdigest() == results[v]['script_sha256']
        receipt = json.loads((OUT / v / 'build_receipt.json').read_text())
        for entry in receipt['sources']:
            assert hashlib.sha256(source_path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
        assert hashlib.sha256(source_path(receipt['module_file']).read_bytes()).hexdigest() == receipt['module_sha256']
        assert receipt['module_sha256'] == results[v]['module_sha256']
    assert hashlib.sha256(source_path(results['shfix_control']['module_file']).read_bytes()).hexdigest() == results['shfix_control']['module_sha256']
    rows = []
    for index in range(4):
        base_row = results['shfix_control']['fixtures'][index]
        name = base_row['name']
        with np.load(OUT / 'shfix_control' / (name + '.npz')) as z:
            f = {k: z[k].copy() for k in z.files}
        exact, cpu_loss, active, radius = formula(f)
        assert rel(exact, np.array(base_row['exact_gradient'])) < 2e-12
        assert abs(cpu_loss - base_row['cpu_loss']) < 2e-13
        assert abs(radius - base_row['margins']['inradius']) < 2e-12
        errors, explain, fds, forward = {}, {}, {}, {}
        for variant in VARIANTS:
            row = results[variant]['fixtures'][index]
            assert row['name'] == name and row['all_perturbations_smooth']
            g = np.array(row['native_gradient'])
            predicted = formula(f, radius_derivative=variant == 'geometry_fixed',
                                pixel_jacobian=variant != 'shfix_control')[0]
            errors[variant] = rel(g[:9], exact[:9])
            explain[variant] = rel(g[:9], predicted[:9])
            assert explain[variant] < 3e-6
            assert rel(g[9:], exact[9:]) < 3e-6
            assert row['forward_sample_max_abs'] < 1e-6
            with np.load(OUT / variant / (name + '.npz')) as z:
                forward[variant] = float(np.max(np.abs(z['native_rgb'] - f['native_rgb'])))
                assert forward[variant] == 0
                assert np.array_equal(z['theta'], f['theta'])
                assert np.array_equal(z['pixels'], f['pixels'])
                assert np.allclose(z['native_grad_0'].ravel(), g[:9], rtol=0, atol=0)
            best_fd = 1e9
            for step in row['finite_differences']:
                gpu_fd = []
                for j, item in enumerate(step['rows']):
                    plus, minus = f['theta'].copy(), f['theta'].copy()
                    plus[j] += item['step']
                    minus[j] -= item['step']
                    delta = float(plus[j]) - float(minus[j])
                    gp, lp, ap, _ = formula(f, plus)
                    gm, lm, am, _ = formula(f, minus)
                    assert ap.tolist() == am.tolist() == active.tolist()
                    assert abs(lp - item['cpu_plus_loss']) < 2e-13
                    assert abs(lm - item['cpu_minus_loss']) < 2e-13
                    cpu_fd = (lp - lm) / delta
                    assert abs(cpu_fd - item['cpu_fd']) < 2e-9
                    native_fd = (item['native_plus_loss'] - item['native_minus_loss']) / delta
                    assert abs(native_fd - item['native_fd']) < 1e-15
                    gpu_fd.append(native_fd)
                best_fd = min(best_fd, rel(np.array(gpu_fd), exact))
            assert best_fd < 1e-4
            fds[variant] = best_fd
        fixed_g = np.array(results['geometry_fixed']['fixtures'][index]['native_gradient'])[:9]
        stopped_g = np.array(results['scale_only']['fixtures'][index]['native_gradient'])[:9]
        cosine = float((stopped_g @ exact[:9]) / (np.linalg.norm(stopped_g) * np.linalg.norm(exact[:9])))
        best_rescale = float((stopped_g @ exact[:9]) / (stopped_g @ stopped_g))
        rows.append(dict(name=name, closed_form_autograd_relative=rel(exact, np.array(base_row['exact_gradient'])),
                         geometry_relative_errors=errors, matched_formula_relative_errors=explain,
                         best_native_fd_relative_errors=fds, forward_image_max_differences=forward,
                         scale_only_cosine_with_exact=cosine,
                         optimal_scalar_rescale=best_rescale,
                         residual_after_best_scalar_rescale=rel(best_rescale * stopped_g, exact[:9])))
    summary = dict(passed=True, archive_files_checked=len(manifest['files']), fixtures=rows,
                   total_parameter_step_pairs=3 * 4 * 11 * 4,
                   independent_cpu='NumPy explicit area/perimeter and projection derivatives; no Torch',
                   maximum_fixed_geometry_relative_error=max(r['geometry_relative_errors']['geometry_fixed'] for r in rows),
                   all_forward_images_identical=True,
                   training_performed=False, scholarly_novelty_established=False)
    (OUT / 'summary_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['passed', 'archive_files_checked',
        'total_parameter_step_pairs', 'maximum_fixed_geometry_relative_error',
        'all_forward_images_identical', 'training_performed']}))


if __name__ == '__main__':
    main()
