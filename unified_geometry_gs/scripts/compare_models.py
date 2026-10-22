#!/usr/bin/env python3
"""
Compare two trained model folders with quantitative metrics.

Metrics:
1) Image-space on train cameras: L1, PSNR, SSIM
2) Mesh-space (important): DTU eval mean_d2s / mean_s2d / overall (Chamfer to GT)

Mesh comparison runs by default using dtu_eval/Offical_DTU_Dataset. Put official
DTU data there (SampleSet + Points.zip, see INSTALL.md) to get mesh metrics.

Example:
  python scripts/compare_models.py \
    --baseline ./output_compare/ggs_baseline/scan105 \
    --optimized ./output_compare/m1_unified/scan105 \
    --out ./output_compare/metrics/scan105_baseline_vs_m1.json

Mesh only (no CUDA image render; folder names may be scan105_edge_dn):
  python scripts/compare_models.py --mesh_only \
    --baseline ./output/dtu/scan105 --optimized ./output/dtu/scan105_edge_dn \
    --out ./output_compare/metrics/scan105_baseline_vs_edge_dn.json

SuGaR / external mesh folder (recon_post.ply + no cfg_args): pass COLMAP scan root:
  python scripts/compare_models.py --mesh_only \
    --baseline ./output_compare/ggs_baseline/scan24 \
    --optimized ./output_compare/sugar/scan24 \
    --external_source_path \"$UGGS_DTU_ROOT/scan24\" \
    --out ./output_compare/metrics/scan24_baseline_vs_sugar.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

# Ensure project root is on path when run as: python scripts/compare_models.py ...
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def load_cfg(model_path: str, role: str = "model") -> Namespace:
    cfg_path = Path(model_path) / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing cfg_args: {cfg_path}\n"
            f"  ({role}) This folder is not a finished training run (or path is wrong).\n"
            "  Train this scan first with the same pipeline, e.g.\n"
            f"    python train.py -s \"$UGGS_DTU_ROOT/scanNNN\" -m \"{model_path}\" ...\n"
            "  then mesh_extract.py; or set --baseline/--optimized to an existing output dir "
            "(e.g. ./output_compare/ggs_baseline/scanNNN)."
        )
    text = cfg_path.read_text(encoding="utf-8")
    cfg = eval(text, {"Namespace": Namespace})  # noqa: S307 - trusted local artifact
    cfg.model_path = model_path
    return cfg


def compute_image_metrics(model_path: str, num_views: int) -> dict[str, float]:
    cfg = load_cfg(model_path, role="image metrics")
    gaussians = GaussianModel(cfg.sh_degree, cfg.sg_degree)
    scene = Scene(cfg, gaussians, load_iteration=-1, shuffle=False)
    cams = scene.getTrainCameras()
    if len(cams) == 0:
        raise RuntimeError(f"No train cameras for {model_path}")

    selected = cams[: min(num_views, len(cams))]
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

    l1_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []

    with torch.no_grad():
        for cam in selected:
            out = render(cam, gaussians, pipe, background, kernel_size=cfg.kernel_size)
            pred = torch.clamp(out["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
            l1_values.append(float(l1_loss(pred, gt).mean().item()))
            psnr_values.append(float(psnr(pred, gt).mean().item()))
            ssim_values.append(float(ssim(pred.unsqueeze(0), gt.unsqueeze(0)).mean().item()))

    return {
        "num_views": float(len(selected)),
        "l1": float(sum(l1_values) / len(l1_values)),
        "psnr": float(sum(psnr_values) / len(psnr_values)),
        "ssim": float(sum(ssim_values) / len(ssim_values)),
    }


_SCAN_ID_RE = re.compile(r"(?i)^scan(\d+)")


def _scan_id_from_model_path(model_path: str) -> int | None:
    """Infer DTU scan id from path segment scanNNN or scanNNN_suffix (e.g. scan105_edge_dn)."""
    for p in reversed(Path(model_path).parts):
        m = _SCAN_ID_RE.match(p)
        if m:
            return int(m.group(1))
    return None


def check_dtu_eval_root(dtu_eval_root: str, scan: int) -> tuple[bool, list[str]]:
    """Verify DTU eval dataset has required files for this scan. Returns (ok, missing_list)."""
    root = Path(dtu_eval_root)
    missing: list[str] = []
    # Calibration: at least one cal*/pos_001.txt
    cal = root / "Calibration"
    if not cal.is_dir():
        missing.append(f"{cal} (folder)")
    else:
        found = False
        for sub in ("cal18", "cal49"):
            if (cal / sub / "pos_001.txt").is_file():
                found = True
                break
        if not found:
            for name in (cal.iterdir() if cal.exists() else []):
                if name.is_dir() and (name / "pos_001.txt").is_file():
                    found = True
                    break
        if not found:
            missing.append(f"Calibration/cal18/pos_001.txt (or cal49/pos_001.txt) under {root}")
    # ObsMask
    obs1 = root / "ObsMask" / f"ObsMask{scan}_10.mat"
    obs2 = root / "ObsMask" / f"Plane{scan}.mat"
    if not obs1.is_file():
        missing.append(str(obs1))
    if not obs2.is_file():
        missing.append(str(obs2))
    # Points/stl
    stl = root / "Points" / "stl" / f"stl{scan:03d}_total.ply"
    if not stl.is_file():
        missing.append(str(stl))
    return (len(missing) == 0, missing)


def _iter_dirs_limited(root: Path, max_depth: int = 4) -> list[Path]:
    dirs: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        cur, depth = queue.pop(0)
        if depth > max_depth:
            continue
        dirs.append(cur)
        try:
            for child in cur.iterdir():
                if child.is_dir():
                    queue.append((child, depth + 1))
        except (PermissionError, FileNotFoundError):
            continue
    return dirs


def resolve_dtu_eval_root(dtu_eval_root: str, scan: int) -> tuple[str | None, list[str]]:
    """
    Resolve DTU eval root from flexible layouts.
    Supports passing:
    - exact Offical_DTU_Dataset root
    - parent folder containing Offical_DTU_Dataset/Official_DTU_Dataset
    - parent folder containing Calibration/ObsMask/Points in descendants
    """
    root = Path(dtu_eval_root).expanduser().resolve()
    candidates: list[Path] = [
        root,
        root / "Offical_DTU_Dataset",
        root / "Official_DTU_Dataset",
        root / "SampleSet",
        root / "MVS",
    ]
    candidates.extend(_iter_dirs_limited(root, max_depth=4))

    seen: set[str] = set()
    uniq_candidates: list[Path] = []
    for c in candidates:
        cs = str(c)
        if cs in seen:
            continue
        seen.add(cs)
        uniq_candidates.append(c)

    last_missing: list[str] = []
    for c in uniq_candidates:
        ok, missing = check_dtu_eval_root(str(c), scan)
        if ok:
            return str(c), []
        if len(missing) < len(last_missing) or not last_missing:
            last_missing = missing
    return None, last_missing


def compute_mesh_metrics(
    model_path: str,
    dtu_eval_root: str,
    cwd: str | Path,
    scan_id: int,
    external_source_path: str | None = None,
    external_images: str = "images",
) -> dict[str, Any] | None:
    """Run DTU mesh eval (UGGS via evaluate_dtu_mesh.py, or external via evaluate_external_mesh_dtu.py)."""
    mp = Path(model_path).resolve()
    cwd_p = Path(cwd).resolve()
    cfg_args = mp / "cfg_args"
    recon_post = mp / "recon_post.ply"

    if cfg_args.is_file():
        cmd = [
            sys.executable,
            str(cwd_p / "evaluate_dtu_mesh.py"),
            "-m",
            str(mp),
            "--iteration",
            "-1",
            "--DTU",
            dtu_eval_root,
        ]
    elif recon_post.is_file():
        if not external_source_path:
            print(
                f"No cfg_args in {mp}; need --external_source_path (COLMAP scan dir) to align recon_post.ply.",
                file=sys.stderr,
            )
            return None
        cmd = [
            sys.executable,
            str(cwd_p / "evaluate_external_mesh_dtu.py"),
            "--source_path",
            os.path.abspath(external_source_path),
            "--mesh_in",
            str(recon_post),
            "--out_dir",
            str(mp),
            "--scan_id",
            str(int(scan_id)),
            "--DTU",
            os.path.abspath(dtu_eval_root),
            "--images",
            external_images,
        ]
    else:
        print(
            f"Mesh eval skipped for {mp}: need cfg_args or recon_post.ply.",
            file=sys.stderr,
        )
        return None

    try:
        result = subprocess.run(
            cmd, cwd=str(cwd_p), capture_output=True, text=True, timeout=600
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Mesh eval subprocess error: {e}", file=sys.stderr)
        return None
    if result.returncode != 0:
        print(f"Mesh eval failed for {model_path}:", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        print("UGGS: ensure recon_post.ply exists. External: check --external_source_path and COLMAP.", file=sys.stderr)
        return None
    result_path = Path(model_path) / "vis" / "results.json"
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "mean_d2s": data.get("mean_d2s"),
        "mean_s2d": data.get("mean_s2d"),
        "overall": data.get("overall"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline model folder")
    parser.add_argument("--optimized", required=True, help="Optimized model folder (e.g. M1)")
    parser.add_argument("--num_views", type=int, default=16, help="Number of train views for image metrics")
    parser.add_argument(
        "--mesh_only",
        action="store_true",
        help="Skip image metrics; only run DTU mesh eval (UGGS: cfg_args + recon_post.ply; external: recon_post.ply + --external_source_path).",
    )
    parser.add_argument(
        "--scan_id",
        type=int,
        default=None,
        metavar="N",
        help="DTU scan id (e.g. 105). Overrides auto-detect from folder names.",
    )
    default_dtu = str(_PROJECT_ROOT / "dtu_eval" / "Offical_DTU_Dataset")
    parser.add_argument(
        "--dtu_eval_root",
        type=str,
        default=default_dtu,
        help="DTU eval root for mesh quality comparison (default: dtu_eval/Offical_DTU_Dataset). Set to '' to skip.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./output_compare/metrics/compare_baseline_vs_optimized.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--external_source_path",
        type=str,
        default="",
        metavar="DIR",
        help="COLMAP DTU scan root (sparse/0, images/) when a compared folder has recon_post.ply but no cfg_args, e.g. SuGaR exports.",
    )
    parser.add_argument(
        "--external_images",
        type=str,
        default="images",
        help="Images subdirectory under --external_source_path (default: images).",
    )
    args = parser.parse_args()
    ext_src = args.external_source_path.strip() or None

    report: dict[str, Any] = {
        "baseline": {"path": args.baseline},
        "optimized": {"path": args.optimized},
        "delta_optimized_minus_baseline": {},
    }

    if args.mesh_only:
        report["image_metrics_note"] = "Skipped (--mesh_only)."
    else:
        baseline_img = compute_image_metrics(args.baseline, args.num_views)
        optimized_img = compute_image_metrics(args.optimized, args.num_views)
        report["baseline"]["image_metrics"] = baseline_img
        report["optimized"]["image_metrics"] = optimized_img
        report["delta_optimized_minus_baseline"].update({
            "l1": optimized_img["l1"] - baseline_img["l1"],
            "psnr": optimized_img["psnr"] - baseline_img["psnr"],
            "ssim": optimized_img["ssim"] - baseline_img["ssim"],
        })

    cwd = _PROJECT_ROOT
    dtu_eval_root = args.dtu_eval_root.strip() or None
    if dtu_eval_root:
        scan = args.scan_id
        if scan is None:
            scan = _scan_id_from_model_path(args.baseline) or _scan_id_from_model_path(args.optimized)
        if scan is None:
            print("Warning: Cannot infer DTU scan id; use --scan_id N. Skipping mesh metrics.", file=sys.stderr)
            report["mesh_metrics_note"] = (
                "Scan id could not be inferred (expect .../scan105 or .../scan105_edge_dn); pass --scan_id."
            )
        else:
            resolved_root, missing = resolve_dtu_eval_root(dtu_eval_root, scan)
            if resolved_root is None:
                report["mesh_metrics_note"] = (
                    "网格质量对比未执行 (Mesh metrics skipped): 缺少 DTU 评估数据。\n"
                    "Required files:\n  " + "\n  ".join(missing)
                    + "\n  Download: SampleSet (Calibration, ObsMask) and Points.zip from https://roboimagedata.compute.dtu.dk/?page_id=36 ."
                    + f"\n  Put them under: {dtu_eval_root} (see INSTALL.md)."
                )
                print("\n[网格质量对比 未运行] " + report["mesh_metrics_note"], file=sys.stderr)
            else:
                baseline_mesh = compute_mesh_metrics(
                    args.baseline,
                    resolved_root,
                    cwd,
                    scan,
                    external_source_path=ext_src,
                    external_images=args.external_images,
                )
                optimized_mesh = compute_mesh_metrics(
                    args.optimized,
                    resolved_root,
                    cwd,
                    scan,
                    external_source_path=ext_src,
                    external_images=args.external_images,
                )
                report["baseline"]["mesh_metrics"] = baseline_mesh
                report["optimized"]["mesh_metrics"] = optimized_mesh
                if baseline_mesh and optimized_mesh:
                    report["delta_optimized_minus_baseline"].update({
                        "mean_d2s": optimized_mesh["mean_d2s"] - baseline_mesh["mean_d2s"],
                        "mean_s2d": optimized_mesh["mean_s2d"] - baseline_mesh["mean_s2d"],
                        "overall": optimized_mesh["overall"] - baseline_mesh["overall"],
                    })
                else:
                    report["mesh_metrics_note"] = (
                        "网格质量对比不完整: 至少一个模型 mesh 评估失败。请先为两个模型导出网格 (recon_post.ply)。"
                    )
                    print(report["mesh_metrics_note"], file=sys.stderr)
    report["mesh_quality_compared"] = bool(
        report.get("baseline", {}).get("mesh_metrics") and report.get("optimized", {}).get("mesh_metrics")
    )
    report["metrics_description"] = {
        "image_metrics": "L1 (↓), PSNR (↑), SSIM (↑) on training views.",
        "mesh_metrics": "mean_d2s, mean_s2d, overall (Chamfer to GT, ↓ = better). 网格质量对比需 DTU 官方数据.",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Comparison Summary ===")
    delta = report["delta_optimized_minus_baseline"]
    img_delta = {k: v for k, v in delta.items() if k in ("l1", "psnr", "ssim")}
    if img_delta:
        print("Image: ", img_delta)
    elif args.mesh_only:
        print("Image: (skipped --mesh_only)")
    if "mean_d2s" in delta:
        print(
            "Mesh (网格质量): ",
            {k: delta[k] for k in delta if k in ("mean_d2s", "mean_s2d", "overall")},
        )
    if report.get("mesh_metrics_note"):
        print("Note:", report["mesh_metrics_note"].split("\n")[0])
    print(f"Saved full report to: {out_path}")


if __name__ == "__main__":
    main()
