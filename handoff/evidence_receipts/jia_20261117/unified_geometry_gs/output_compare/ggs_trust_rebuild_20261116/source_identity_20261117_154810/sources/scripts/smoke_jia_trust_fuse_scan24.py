#!/usr/bin/env python3
"""scan24 fixed-first-image trust-mask smoke (mask consistency + boundary stats).

Loss/grad default-path compatibility lives in compat_jia_patchmatch_grad.py
(must run under torch.enable_grad, not this no_grad wrapper).

Does not build full-scene trust mesh. Refuses overwrite of out_dir artifacts.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.jia_io_guard import assert_fresh_output_dir  # noqa: E402
from utils.loss_utils import PatchMatch  # noqa: E402
from mesh_extract import _trust_mask_depth  # noqa: E402


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tensor_digest(t: torch.Tensor) -> str:
    arr = t.detach().cpu().numpy()
    return sha256_bytes(arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode())


def main() -> None:
    ap = ArgumentParser()
    model = ModelParams(ap, sentinel=True)
    pipe = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--ncc_mask_thresh", type=float, default=0.9)
    ap.add_argument("--quiet", action="store_true")
    args = get_combined_args(ap)
    safe_state(args.quiet)

    out_dir = Path(
        assert_fresh_output_dir(
            args.out_dir,
            artifacts=["smoke_receipt.json", "smoke_arrays.npz", "smoke_redmask.png"],
            forbidden_roots=[args.model_path],
            label="smoke_jia_trust_fuse_scan24",
        )
    )

    dataset = model.extract(args)
    pipeline = pipe.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTrainCameras()
    preview = min(cams, key=lambda c: c.image_name)
    preview_idx = next(i for i, c in enumerate(cams) if c.image_name == preview.image_name)

    pm = PatchMatch(3, 1.0, dataset.kernel_size, pipeline, debug=False, model_path=dataset.model_path)
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    with torch.no_grad():
        render_pkg = render(preview, gaussians, pipeline, bg, dataset.kernel_size)
        out0 = pm(gaussians, render_pkg, preview, None, iteration=0)
        assert isinstance(out0, tuple) and len(out0) == 2
        nearest = cams[preview.nearest_id[0]] if preview.nearest_id else None
        out1 = pm(gaussians, render_pkg, preview, nearest, iteration=0)
        assert len(out1) == 2
        out2 = pm(gaussians, render_pkg, preview, nearest, iteration=0, return_trust_maps=False)
        assert len(out2) == 2

        # Boundary: no nearest trust maps full-reject
        out_tm = pm(gaussians, render_pkg, preview, None, iteration=0, return_trust_maps=True)
        assert len(out_tm) == 3
        tm0 = out_tm[-1]
        assert not bool(tm0["keep"].any().item())

        # Boundary: empty d_mask
        pkg0 = dict(render_pkg)
        pkg0["median_depth"] = torch.zeros_like(render_pkg["median_depth"])
        out_empty = pm(gaussians, pkg0, preview, nearest, iteration=0, return_trust_maps=True)
        assert len(out_empty) == 3
        assert not bool(out_empty[-1]["keep"].any().item())

        digests = []
        for run in (1, 2):
            color, depth, stats, maps, depth_hw, K = _trust_mask_depth(
                gaussians=gaussians,
                pipe=pipeline,
                kernel_size=dataset.kernel_size,
                patchmatch=pm,
                viewpoint_cam=preview,
                train_cams=cams,
                background=bg,
                ncc_mask_thresh=float(args.ncc_mask_thresh),
            )
            assert stats["counts_sum_eq_B"], stats
            assert stats["depth_nonzero_eq_keep"], stats
            assert stats.get("other_fail_subdiv_sum_eq", True), stats
            # Formula check: K == B & geom & eval & warp & isfinite(err) & err<tau
            B = torch.isfinite(depth_hw) & (depth_hw > 0)
            err = maps["error"]
            K_re = (
                B
                & maps["geometry_valid"]
                & maps["evaluated"]
                & maps["warp_valid"]
                & torch.isfinite(err)
                & (err < float(args.ncc_mask_thresh))
            )
            assert torch.equal(K, K_re), "K formula mismatch"
            depth_t = torch.from_numpy(depth)
            assert torch.equal(depth_t[K], depth_hw[K])
            assert torch.all(depth_t[~K] == 0)
            digests.append(
                {
                    "stats": stats,
                    "K": tensor_digest(K),
                    "error": tensor_digest(maps["error"]),
                    "evaluated": tensor_digest(maps["evaluated"]),
                    "warp_valid": tensor_digest(maps["warp_valid"]),
                    "depth": sha256_bytes(np.ascontiguousarray(depth).tobytes()),
                }
            )
            if run == 1:
                np.savez_compressed(
                    out_dir / "smoke_arrays.npz",
                    depth_hw=depth_hw.numpy(),
                    K=K.numpy(),
                    error=maps["error"].numpy(),
                    evaluated=maps["evaluated"].numpy(),
                    warp_valid=maps["warp_valid"].numpy(),
                    geometry_valid=maps["geometry_valid"].numpy(),
                )
                try:
                    import cv2

                    img = (np.clip(color, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
                    depth_np = depth_hw.numpy()
                    K_np = K.numpy()
                    # Outside B: non-finite OR <=0 (NaN must not fall through as "rejected")
                    outside_B = (~np.isfinite(depth_np)) | (depth_np <= 0)
                    rejected = (~K_np) & (~outside_B)
                    img[rejected] = (0, 0, 255)
                    img[outside_B] = (img[outside_B] * 0.3).astype(np.uint8)
                    cv2.imwrite(str(out_dir / "smoke_redmask.png"), img)
                except Exception as e:  # noqa: BLE001
                    (out_dir / "smoke_redmask_error.txt").write_text(str(e), encoding="utf-8")

    consistent = digests[0] == digests[1]
    receipt = {
        "preview_image_name": preview.image_name,
        "preview_camera_list_index": preview_idx,
        "nearest_id0": int(preview.nearest_id[0]) if preview.nearest_id else None,
        "nearest_image_name": nearest.image_name if nearest is not None else None,
        "default_return_arity_ok": True,
        "two_run_consistent": consistent,
        "grad_compat": "see compat_jia_patchmatch_grad.py (not covered here; this script uses no_grad)",
        "run1": digests[0],
        "run2": digests[1],
        "model_path": dataset.model_path,
        "iteration": args.iteration,
        "note_keep_frac_is_diagnostic_only": True,
    }
    (out_dir / "smoke_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"ok": consistent, "receipt": str(out_dir / "smoke_receipt.json")}, indent=2))
    if not consistent:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
