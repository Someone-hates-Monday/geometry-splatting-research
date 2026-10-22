#!/usr/bin/env python3
"""Compare DTU data→STL (d2s) tails for two aligned meshes (e.g. baseline vs no_dn).

Classifies mean-d2s excess into:
  near   : d < 2
  mid    : 2 <= d < 5   (surface expansion / bias)
  far    : 5 <= d < 20  (floaters / sheets)
  clip   : d >= 20      (excluded from official mean, counted separately)

Usage:
  python scripts/diagnose_d2s.py \\
    --a ./output_compare/ggs_baseline/scan122/recon_aligned.ply \\
    --b ./output_compare/ggs_ablate_no_dn/scan122/recon_aligned.ply \\
    --scan 122 --labels baseline no_dn
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from scipy.io import loadmat


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[: n1 + 1, : n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def mesh_to_obs_points(mesh_path: Path, dataset_dir: Path, scan: int, thresh: float = 0.2, patch: float = 60.0):
    data_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(data_mesh.vertices)
    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [a[non_zero] for a in (l1, l2, area2, v1, v2, tri_vert)]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)
    with mp.Pool() as pool:
        new_pts = pool.map(
            sample_single_tri,
            ((n1[i, 0], n2[i, 0], v1[i : i + 1], v2[i : i + 1], tri_vert[i : i + 1, 0]) for i in range(len(n1))),
            chunksize=1024,
        )
    data_pcd = np.concatenate([vertices, *new_pts], axis=0) if new_pts else vertices
    rng = np.random.default_rng(0)
    rng.shuffle(data_pcd, axis=0)

    nn = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm="kd_tree", n_jobs=-1)
    nn.fit(data_pcd)
    rnn = nn.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for i, idxs in enumerate(rnn):
        if mask[i]:
            mask[idxs] = False
            mask[i] = True
    data_down = data_pcd[mask]

    obs = loadmat(str(dataset_dir / "ObsMask" / f"ObsMask{scan}_10.mat"))
    ObsMask, BB, Res = obs["ObsMask"], obs["BB"].astype(np.float32), obs["Res"]
    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(-1) == 3
    data_in = data_down[inbound]
    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    gin = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(-1) == 3
    data_grid_in = data_grid[gin]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(bool)
    return data_in[gin][in_obs], data_in, vertices.shape[0]


def summarize(name: str, dist: np.ndarray, max_dist: float = 20.0) -> dict:
    d = dist.reshape(-1)
    in_mean = d[d < max_dist]
    bins = {
        "near_<2": float(((d >= 0) & (d < 2)).mean()),
        "mid_2_5": float(((d >= 2) & (d < 5)).mean()),
        "far_5_20": float(((d >= 5) & (d < 20)).mean()),
        "clip_ge20": float((d >= 20).mean()),
    }
    # Contribution to official mean from each band (among d < max_dist)
    contrib = {}
    if in_mean.size:
        mean = float(in_mean.mean())
        for lo, hi, key in ((0, 2, "near_<2"), (2, 5, "mid_2_5"), (5, 20, "far_5_20")):
            m = (in_mean >= lo) & (in_mean < hi)
            contrib[key] = float(in_mean[m].sum() / in_mean.size) if m.any() else 0.0
    else:
        mean = float("nan")
    pct = {
        "p50": float(np.percentile(in_mean, 50)) if in_mean.size else None,
        "p90": float(np.percentile(in_mean, 90)) if in_mean.size else None,
        "p95": float(np.percentile(in_mean, 95)) if in_mean.size else None,
        "p99": float(np.percentile(in_mean, 99)) if in_mean.size else None,
    }
    return {
        "name": name,
        "n_obs": int(d.size),
        "mean_d2s": mean,
        "frac": bins,
        "contrib_to_mean": contrib,
        "percentiles": pct,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--scan", required=True, type=int)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--dataset_dir", type=Path, default=Path("dtu_eval/Offical_DTU_Dataset"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stl = np.asarray(
        o3d.io.read_point_cloud(str(args.dataset_dir / "Points" / "stl" / f"stl{args.scan:03d}_total.ply")).points
    )
    nn = skln.NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    nn.fit(stl)

    reports = []
    for path, label in zip((args.a, args.b), args.labels):
        print(f"[diagnose-d2s] sampling {label}: {path}")
        pts_obs, pts_in, n_vert = mesh_to_obs_points(path, args.dataset_dir, args.scan)
        dist, _ = nn.kneighbors(pts_obs, n_neighbors=1, return_distance=True)
        r = summarize(label, dist)
        r["n_vertices"] = int(n_vert)
        r["n_data_in"] = int(pts_in.shape[0])
        # bbox of obs points
        r["obs_bbox_diag"] = float(np.linalg.norm(pts_obs.max(0) - pts_obs.min(0)))
        reports.append(r)
        print(
            f"  mean_d2s={r['mean_d2s']:.6f} n_obs={r['n_obs']} verts={r['n_vertices']} "
            f"frac near/mid/far/clip="
            f"{r['frac']['near_<2']:.3f}/{r['frac']['mid_2_5']:.3f}/"
            f"{r['frac']['far_5_20']:.3f}/{r['frac']['clip_ge20']:.3f}"
        )
        print(f"  contrib_to_mean near/mid/far={r['contrib_to_mean']}")
        print(f"  percentiles {r['percentiles']}")

    a, b = reports
    delta = {
        "mean_d2s": b["mean_d2s"] - a["mean_d2s"],
        "frac_mid": b["frac"]["mid_2_5"] - a["frac"]["mid_2_5"],
        "frac_far": b["frac"]["far_5_20"] - a["frac"]["far_5_20"],
        "frac_clip": b["frac"]["clip_ge20"] - a["frac"]["clip_ge20"],
        "contrib_mid": b["contrib_to_mean"]["mid_2_5"] - a["contrib_to_mean"]["mid_2_5"],
        "contrib_far": b["contrib_to_mean"]["far_5_20"] - a["contrib_to_mean"]["far_5_20"],
        "n_obs": b["n_obs"] - a["n_obs"],
        "n_vertices": b["n_vertices"] - a["n_vertices"],
    }
    # Heuristic label for the dominant excess source
    if delta["mean_d2s"] <= 0:
        verdict = "B not worse on d2s"
    elif delta["contrib_far"] >= abs(delta["contrib_mid"]) and delta["contrib_far"] > 0:
        verdict = "floaters/sheets (far band dominates mean excess)"
    elif delta["contrib_mid"] > 0:
        verdict = "surface expansion/bias (mid band dominates mean excess)"
    elif delta["frac_clip"] > 0.01 and delta["n_obs"] > 0:
        verdict = "extra out-of-band / clipped geometry (not in official mean)"
    else:
        verdict = "mild / mixed d2s shift"

    out = {
        "scan": args.scan,
        "a": str(args.a),
        "b": str(args.b),
        "reports": reports,
        "delta_b_minus_a": delta,
        "verdict": verdict,
    }
    print(f"\nΔ(B−A) mean_d2s={delta['mean_d2s']:+.6f} "
          f"contrib_mid={delta['contrib_mid']:+.6f} contrib_far={delta['contrib_far']:+.6f}")
    print(f"VERDICT: {verdict}")

    out_path = args.out or Path(f"./output_compare/metrics/scan{args.scan}_d2s_diagnose.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
