#!/usr/bin/env python3
"""Compare HEAD PatchMatch vs working-tree PatchMatch on default training path.

Requires CUDA. Does not run optimizer. Writes a receipt JSON; refuses overwrite.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.jia_io_guard import assert_fresh_output_dir  # noqa: E402
from utils.loss_utils import PatchMatch as CurrentPatchMatch  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(t: torch.Tensor | None) -> str | None:
    if t is None:
        return None
    arr = t.detach().float().cpu().numpy()
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()).hexdigest()


def load_head_patchmatch(repo_git: Path, artifact_dir: Path) -> tuple[type, dict]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_git), text=True).strip()
    src = subprocess.check_output(
        ["git", "show", f"HEAD:unified_geometry_gs/utils/loss_utils.py"],
        cwd=str(repo_git),
    )
    out = artifact_dir / f"loss_utils_HEAD_{head[:12]}.py"
    if out.exists():
        # Refuse silent overwrite of a different blob
        if hashlib.sha256(out.read_bytes()).hexdigest() != hashlib.sha256(src).hexdigest():
            raise RuntimeError(f"baseline artifact conflict: {out}")
    else:
        out.write_bytes(src)
    spec = importlib.util.spec_from_file_location("jia_loss_utils_head_baseline", out)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Ensure sibling imports resolve against uggs root
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    meta = {
        "git_head": head,
        "baseline_path": str(out),
        "baseline_sha256": sha256_file(out),
        "current_loss_utils_sha256": sha256_file(ROOT / "utils" / "loss_utils.py"),
    }
    return mod.PatchMatch, meta


def clone_pkg_for_grad(render_pkg: dict) -> dict:
    pkg = dict(render_pkg)
    pkg["median_depth"] = render_pkg["median_depth"].detach().clone().requires_grad_(True)
    pkg["normal"] = render_pkg["normal"].detach().clone().requires_grad_(True)
    return pkg


def run_forward(pm, gaussians, pkg, cam, nearest, *, return_trust_maps: bool | None):
    kwargs = {"iteration": 0}
    if return_trust_maps is not None:
        kwargs["return_trust_maps"] = return_trust_maps
    out = pm(gaussians, pkg, cam, nearest, **kwargs)
    return out


def loss_pair(out) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(out, tuple) and len(out) >= 2
    return out[0], out[1]


def compare_scalar(a: torch.Tensor, b: torch.Tensor, name: str, atol: float, rtol: float) -> dict:
    a_f = a.detach().float().reshape(-1)
    b_f = b.detach().float().reshape(-1)
    if a_f.numel() != b_f.numel():
        return {"ok": False, "name": name, "reason": f"numel {a_f.numel()} vs {b_f.numel()}"}
    diff = (a_f - b_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    # relative where |b| large
    denom = torch.maximum(b_f.abs(), torch.tensor(1e-8, device=b_f.device))
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    bit_eq = bool(torch.equal(a_f, b_f))
    ok = bit_eq or (max_abs <= atol and max_rel <= rtol)
    return {
        "ok": ok,
        "name": name,
        "bit_equal": bit_eq,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "atol": atol,
        "rtol": rtol,
        "a": float(a_f.sum().item()),
        "b": float(b_f.sum().item()),
    }


def compare_grad(ga: torch.Tensor | None, gb: torch.Tensor | None, name: str, atol: float, rtol: float) -> dict:
    if ga is None and gb is None:
        return {"ok": True, "name": name, "both_none": True}
    if (ga is None) != (gb is None):
        return {
            "ok": False,
            "name": name,
            "reason": f"None mismatch head={ga is None} cur={gb is None}",
        }
    assert ga is not None and gb is not None
    return compare_scalar(ga, gb, name, atol, rtol)


def main() -> None:
    ap = ArgumentParser()
    model = ModelParams(ap, sentinel=True)
    pipe = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--repo_git", type=str, default=str(ROOT.parent))
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--quiet", action="store_true")
    args = get_combined_args(ap)
    safe_state(args.quiet)

    out_dir = Path(
        assert_fresh_output_dir(
            args.out_dir,
            artifacts=["compat_receipt.json"],
            forbidden_roots=[args.model_path],
            label="compat_jia_patchmatch_grad",
        )
    )
    BaselinePM, baseline_meta = load_head_patchmatch(Path(args.repo_git), out_dir)

    dataset = model.extract(args)
    pipeline = pipe.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTrainCameras()
    preview = min(cams, key=lambda c: c.image_name)
    nearest = cams[preview.nearest_id[0]] if preview.nearest_id else None
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # Deterministic CUDA for this check; declare if disabled.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass

    pm_h = BaselinePM(3, 1.0, dataset.kernel_size, pipeline, debug=False, model_path=dataset.model_path)
    pm_c = CurrentPatchMatch(3, 1.0, dataset.kernel_size, pipeline, debug=False, model_path=dataset.model_path)

    atol = float(args.atol)
    rtol = float(args.rtol)
    # Pre-declare: default demand bit-equality (atol=rtol=0). Do not loosen after seeing diffs.
    tolerance_policy = {
        "declared_before_run": True,
        "atol": atol,
        "rtol": rtol,
        "note": "CUDA nondeterminism would require pre-declared nonzero tol + HEAD self-repeat; default is bit-eq.",
    }

    cases = []

    def add_case(name: str, ok: bool, detail: dict):
        cases.append({"case": name, "ok": ok, **detail})

    # --- Case A: nearest None (HEAD has no return_trust_maps kw) ---
    with torch.no_grad():
        render_pkg = render(preview, gaussians, pipeline, bg, dataset.kernel_size)
        out_h = run_forward(pm_h, gaussians, render_pkg, preview, None, return_trust_maps=None)
        out_c_default = run_forward(pm_c, gaussians, render_pkg, preview, None, return_trust_maps=None)
        out_c_false = run_forward(pm_c, gaussians, render_pkg, preview, None, return_trust_maps=False)
        for label, out_c in (("default", out_c_default), ("return_trust_maps_False", out_c_false)):
            nh, gh = loss_pair(out_h)
            nc, gc = loss_pair(out_c)
            c1 = compare_scalar(nh, nc, "ncc", atol, rtol)
            c2 = compare_scalar(gh, gc, "geo", atol, rtol)
            add_case(
                f"no_nearest/{label}",
                c1["ok"] and c2["ok"] and len(out_h) == 2 and len(out_c) == 2,
                {"ncc": c1, "geo": c2, "arity_h": len(out_h), "arity_c": len(out_c)},
            )

    # --- Case B: empty d_mask via zero depth ---
    with torch.no_grad():
        pkg0 = dict(render_pkg)
        pkg0["median_depth"] = torch.zeros_like(render_pkg["median_depth"])
        out_h = run_forward(pm_h, gaussians, pkg0, preview, nearest, return_trust_maps=None)
        out_c = run_forward(pm_c, gaussians, pkg0, preview, nearest, return_trust_maps=False)
        c1 = compare_scalar(loss_pair(out_h)[0], loss_pair(out_c)[0], "ncc", atol, rtol)
        c2 = compare_scalar(loss_pair(out_h)[1], loss_pair(out_c)[1], "geo", atol, rtol)
        add_case("empty_d_mask", c1["ok"] and c2["ok"], {"ncc": c1, "geo": c2})

    # --- Case C: normal valid path WITH grads ---
    torch.cuda.synchronize()
    pkg_h = clone_pkg_for_grad(render_pkg)
    pkg_c = clone_pkg_for_grad(render_pkg)
    # identical starting values
    pkg_c["median_depth"].data.copy_(pkg_h["median_depth"].data)
    pkg_c["normal"].data.copy_(pkg_h["normal"].data)

    with torch.enable_grad():
        out_h = run_forward(pm_h, gaussians, pkg_h, preview, nearest, return_trust_maps=None)
        out_c = run_forward(pm_c, gaussians, pkg_c, preview, nearest, return_trust_maps=False)
        ncc_h, geo_h = loss_pair(out_h)
        ncc_c, geo_c = loss_pair(out_c)
        loss_h = ncc_h.reshape(-1).sum() + geo_h.reshape(-1).sum()
        loss_c = ncc_c.reshape(-1).sum() + geo_c.reshape(-1).sum()
        # Skip backward only if both are disconnected zeros with no graph
        can_bwd_h = loss_h.requires_grad
        can_bwd_c = loss_c.requires_grad
        if can_bwd_h:
            loss_h.backward()
        if can_bwd_c:
            loss_c.backward()

    comps = [
        compare_scalar(ncc_h, ncc_c, "ncc_loss", atol, rtol),
        compare_scalar(geo_h, geo_c, "geo_loss", atol, rtol),
        compare_grad(pkg_h["median_depth"].grad, pkg_c["median_depth"].grad, "grad_median_depth", atol, rtol),
        compare_grad(pkg_h["normal"].grad, pkg_c["normal"].grad, "grad_normal", atol, rtol),
    ]
    add_case(
        "normal_enable_grad",
        all(c["ok"] for c in comps) and can_bwd_h and can_bwd_c,
        {
            "comps": comps,
            "requires_grad_h": can_bwd_h,
            "requires_grad_c": can_bwd_c,
            "grad_requested": True,
            "ran_under_no_grad": False,
        },
    )

    # --- Case D: single valid depth pixel (stress reshape path); compare default losses ---
    with torch.no_grad():
        depth = render_pkg["median_depth"].clone()
        flat = depth.view(-1)
        # keep one strictly positive finite pixel; zero others
        pos = torch.nonzero(torch.isfinite(flat) & (flat > 0), as_tuple=False)
        if pos.numel() == 0:
            add_case("single_positive_depth", False, {"reason": "no positive depth"})
        else:
            keep_i = int(pos[0].item())
            flat.zero_()
            flat[keep_i] = render_pkg["median_depth"].view(-1)[keep_i]
            pkg1 = dict(render_pkg)
            pkg1["median_depth"] = depth
            out_h = run_forward(pm_h, gaussians, pkg1, preview, nearest, return_trust_maps=None)
            out_c = run_forward(pm_c, gaussians, pkg1, preview, nearest, return_trust_maps=False)
            c1 = compare_scalar(loss_pair(out_h)[0], loss_pair(out_c)[0], "ncc", atol, rtol)
            c2 = compare_scalar(loss_pair(out_h)[1], loss_pair(out_c)[1], "geo", atol, rtol)
            add_case("single_positive_depth", c1["ok"] and c2["ok"], {"ncc": c1, "geo": c2, "keep_i": keep_i})

    # --- Case E: current-only trust maps arity on empty / none (HEAD has no API) ---
    with torch.no_grad():
        out_t = pm_c(gaussians, render_pkg, preview, None, iteration=0, return_trust_maps=True)
        assert len(out_t) == 3
        tm = out_t[-1]
        ok_maps = all(k in tm for k in ("evaluated", "geometry_valid", "warp_valid", "error", "keep"))
        add_case(
            "current_trust_maps_no_nearest",
            ok_maps and (not bool(tm["keep"].any().item())),
            {"keys": sorted(tm.keys()), "keep_any": bool(tm["keep"].any().item())},
        )

    # Extension identity
    import warp_patch_ncc

    ext_meta = {
        "warp_patch_ncc_file": getattr(warp_patch_ncc, "__file__", None),
    }

    ok_all = all(c["ok"] for c in cases)
    receipt = {
        "ok": ok_all,
        "baseline": baseline_meta,
        "tolerance_policy": tolerance_policy,
        "preview_image_name": preview.image_name,
        "nearest_image_name": None if nearest is None else nearest.image_name,
        "extension": ext_meta,
        "cases": cases,
        "model_path": dataset.model_path,
        "iteration": args.iteration,
        "render_depth_digest": tensor_digest(render_pkg["median_depth"]),
        "render_normal_digest": tensor_digest(render_pkg["normal"]),
    }
    receipt_path = out_dir / "compat_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"ok": ok_all, "receipt": str(receipt_path)}, indent=2))
    if not ok_all:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
