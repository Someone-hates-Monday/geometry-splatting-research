"""Save a local source-and-diagnostics snapshot without models or secrets.

This workspace has no Git history. Hashes alone cannot recover old source, so
the archive stores the actual selected files and a per-file digest manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    manifest_out = args.out.with_suffix(".manifest.json")
    if args.out.exists() or manifest_out.exists():
        parser.error("Archive and manifest paths must be new.")
    pair = json.loads((args.pair / "manifest.json").read_text())
    paths = {ROOT / name for name in pair["source_sha256"]}
    paths.update(ROOT / name for name in ("gaussian_renderer/__init__.py", "scene/cameras.py",
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu",
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/render_backward.cu",
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h", "dtu_eval/eval.py"))
    for pattern in ("*.py", "*.cjs", "*.cpp", "*.cuh", "*.md", "*.tex", "*.inc", "canvas_coherent_stage*.tsx", "canvas_guarded_stage*.tsx", "canvas_strict_stage*.tsx", "canvas_device_root_stage*.tsx", "canvas_reciprocal_stage*.tsx", "canvas_short_certificate_stage*.tsx", "canvas_full_topology_stage*.tsx", "canvas_csr_capture_stage*.tsx", "canvas_shared_cache_stage*.tsx", "canvas_csr_decode_stage*.tsx", "results/*.json", "literature/*.json",
        "results/factor_holdout_launch_failure*.zip", "results/precision_scale_vjp_failure*.zip",
        "results/thin_quantile_prototype*.zip", "results/quantile_band_failure*.zip",
        "results/quantile_band_oracle_failure*.zip",
        "results/quantile_band*.log", "results/thin_quantile*.log",
        "results/directional_field*.log",
        "results/world_affine_rays*.log", "results/canvas_directional_staging_failure*.zip",
        "results/matched_ray_factors*.log", "results/perspective_reparam*.log",
        "results/clipped_packet*.log", "results/clipped_packet_native_prototype*.zip",
        "results/packet_input*.log", "results/packet_conditioning*.log", "results/production_packet*.log",
        "results/exact_perspective*.log", "results/frozen_exact*.log",
        "results/exact_outlier*.log", "results/exact_outlier*.zip",
        "results/coherent_factor*.log",
        "figures/*.png", "figures/*.svg", "figures/*.json",
        "experiments/*/manifest.json", "experiments/*/extraction_manifest.json", "experiments/*/paired_mesh_metrics.json",
        "experiments/exact37_v1/postprocess_v2/*.json", "experiments/exact37_v1/seeded_eval_v2/paired_seeded_metrics.json",
        "experiments/exact37_v1/*/training_integration.json",
        "experiments/adjquad*/sources.zip", "experiments/adjquad*/*/solver_training_integration.json",
        "experiments/adjquad*/*/alignment.json", "experiments/adjquad*/images_*.json",
        "experiments/adjquad*/eval_*/manifest.json", "experiments/adjquad*/eval_*/results.json",
        "experiments/adjquad*/eval_*/eval_seeded_snapshot.py",
        "experiments/peak_development*/*.json", "experiments/peak_development*/sources.zip",
        "experiments/sampling*/*.json", "experiments/sampling*/sources.zip",
        "experiments/sampling_batch_v1/benchmark_sampling_batch_before_ssim_fix.py",
        "experiments/inverse_gradient*/*.json", "experiments/inverse_gradient*/sources.zip",
        "experiments/precise_gradient*/*.json", "experiments/precise_gradient*/sources.zip",
        "experiments/precise_development*/*.json", "experiments/precise_development*/sources.zip",
        "experiments/precise_development*/*/*.json", "experiments/precise_development*/*/sources.zip",
        "experiments/mixed_development*/*.json", "experiments/mixed_development*/sources.zip",
        "experiments/mixed_development*/*/*.json", "experiments/mixed_development*/*/sources.zip",
        "experiments/mixed_training_timing*/*.json", "experiments/mixed_training_timing*/sources.zip",
        "experiments/mixed_training_timing*/*/*integration.json",
        "experiments/world_preprocess*/*.json", "experiments/world_preprocess*/sources.zip",
        "experiments/corrected_smoke*/*.json", "experiments/corrected_smoke*/sources.zip",
        "experiments/corrected_smoke*/*/*integration.json",
        "experiments/corrected_quality*/*.json", "experiments/corrected_quality*/sources.zip",
        "experiments/corrected_quality*/*/*integration.json",
        "experiments/corrected_quality*/eval_*/*.json", "experiments/corrected_quality*/*/alignment.json",
        "experiments/factor*/*.json", "experiments/factor*/sources.zip",
        "experiments/factor*/*/*integration.json", "experiments/factor*/*/alignment.json",
        "experiments/factor*/eval_*/*.json", "experiments/factor*/eval_*/eval_seeded_snapshot.py",
        "experiments/gram*/*.json", "experiments/gram*/sources.zip", "experiments/gram*/*/*integration.json",
        "experiments/quantile_band*/*.json", "experiments/quantile_band*/sources.zip", "experiments/quantile_band*/*.npz",
        "experiments/directional_field*/*.json", "experiments/directional_field*/sources.zip",
        "experiments/world_affine_rays*/*.json", "experiments/world_affine_rays*/sources.zip",
        "experiments/matched_ray_factors*/*.json", "experiments/matched_ray_factors*/sources.zip", "experiments/matched_ray_factors*/*.npz",
        "experiments/perspective_reparam*/*.json", "experiments/perspective_reparam*/sources.zip", "experiments/perspective_reparam*/*.npz",
        "experiments/clipped_packet*/*.json", "experiments/clipped_packet*/sources.zip", "experiments/clipped_packet*/*.npz",
        "experiments/packet_input*/*.json", "experiments/packet_input*/sources.zip", "experiments/packet_input*/*.npz",
        "experiments/production_packet*/*.json", "experiments/production_packet*/sources.zip", "experiments/production_packet*/*.npz",
        "experiments/production_packet*/failure.txt", "experiments/production_packet*/generated/*.cu",
        "experiments/production_packet*/generated/*.cpp", "experiments/production_packet*/generated/build.ninja",
        "experiments/exact_perspective*/*.json", "experiments/exact_perspective*/sources.zip", "experiments/exact_perspective*/*.npz",
        "experiments/exact_perspective*/failure.txt",
        "experiments/frozen_exact*/*.json", "experiments/frozen_exact*/sources.zip", "experiments/frozen_exact*/*.npz",
        "experiments/frozen_exact*/failure.txt",
        "experiments/exact_outlier*/*.json", "experiments/exact_outlier*/sources.zip", "experiments/exact_outlier*/*.npz",
        "experiments/exact_outlier*/failure.txt", "experiments/exact_outlier*/generated/*.cu",
        "experiments/exact_outlier*/generated/*.cpp", "experiments/exact_outlier*/generated/*.h",
        "experiments/exact_outlier*/generated/build.ninja",
        "experiments/exact_outlier*/*.png", "experiments/exact_outlier*/*.svg",
        "experiments/coherent_factor*/*.json", "experiments/coherent_factor*/sources.zip", "experiments/coherent_factor*/*.npz",
        "experiments/coherent_factor*/failure.txt",
        "experiments/guarded_precision*/*.json", "experiments/guarded_precision*/sources.zip", "experiments/guarded_precision*/failure.txt",
        "results/strict_interval*.log", "experiments/strict_interval*/*.json", "experiments/strict_interval*/sources.zip",
        "experiments/strict_interval*/*.bin", "experiments/strict_interval*/*.npy", "experiments/strict_interval*/*.npz",
        "experiments/strict_interval*/failure.txt", "experiments/strict_interval*/generated/*.cu",
        "experiments/strict_interval*/generated/*.cpp", "experiments/strict_interval*/generated/*.cuh",
        "experiments/strict_interval*/generated/build.ninja",
        "experiments/device_budget*/*.json", "experiments/device_budget*/*.npz", "experiments/device_budget*/sources.zip",
        "experiments/device_budget*/failure.txt", "experiments/device_budget*/generated/*.cu", "experiments/device_budget*/generated/*.cpp",
        "experiments/device_budget*/generated/*.cuh", "experiments/device_budget*/generated/build.ninja",
        "experiments/device_root*/*.json", "experiments/device_root*/*.npz", "experiments/device_root*/sources.zip",
        "experiments/device_root*/failure.txt", "experiments/device_root*/generated/*.cu", "experiments/device_root*/generated/*.cpp",
        "experiments/device_root*/generated/*.cuh", "experiments/device_root*/generated/build.ninja",
        "experiments/reciprocal_root*/*.json", "experiments/reciprocal_root*/*.npz", "experiments/reciprocal_root*/sources.zip",
        "experiments/reciprocal_root*/failure.txt", "experiments/reciprocal_root*/*_generated/*.cu",
        "experiments/reciprocal_root*/*_generated/*.cpp", "experiments/reciprocal_root*/*_generated/*.cuh",
        "experiments/reciprocal_root*/*_generated/build.ninja",
        "experiments/short_certificate*/*.json", "experiments/short_certificate*/*.npz", "experiments/short_certificate*/sources.zip",
        "experiments/short_certificate*/failure.txt", "experiments/short_certificate*/*_generated/*.cu",
        "experiments/short_certificate*/*_generated/*.cpp", "experiments/short_certificate*/*_generated/*.cuh",
        "experiments/short_certificate*/*_generated/build.ninja",
        "experiments/short_fallback*/*.json", "experiments/short_fallback*/*.npz", "experiments/short_fallback*/sources.zip",
        "experiments/short_fallback*/failure.txt",
        "experiments/full_topology*/*.json", "experiments/full_topology*/*.npz", "experiments/full_topology*/sources.zip",
        "experiments/full_topology*/failure.txt", "experiments/full_topology*/*_generated/src/*.cu",
        "experiments/full_topology*/*_generated/src/*.cpp", "experiments/full_topology*/*_generated/src/*.h",
        "experiments/full_topology*/*_generated/src/cuda_rasterizer/*.cu", "experiments/full_topology*/*_generated/src/cuda_rasterizer/*.h",
        "experiments/full_topology*/*_generated/build.ninja",
        "experiments/csr_capture*/*.json", "experiments/csr_capture*/*.npz", "experiments/csr_capture*/sources.zip", "experiments/csr_capture*/failure.txt",
        "experiments/csr_capture*/csr_generated/src/*.cu", "experiments/csr_capture*/csr_generated/src/*.cpp", "experiments/csr_capture*/csr_generated/src/*.h",
        "experiments/csr_capture*/csr_generated/src/cuda_rasterizer/*.cu", "experiments/csr_capture*/csr_generated/src/cuda_rasterizer/*.h",
        "experiments/csr_capture*/csr_generated/build.ninja",
        "experiments/shared_cache*/*.json", "experiments/shared_cache*/*.npz", "experiments/shared_cache*/sources.zip", "experiments/shared_cache*/failure.txt",
        "experiments/shared_cache*/generated/*.cu", "experiments/shared_cache*/generated/*.cpp", "experiments/shared_cache*/generated/*.cuh", "experiments/shared_cache*/generated/build.ninja",
        "experiments/csr_decode*/*.json", "experiments/csr_decode*/*.npz", "experiments/csr_decode*/sources.zip", "experiments/csr_decode*/failure.txt",
        "experiments/csr_decode*/query_generated/*.cu", "experiments/csr_decode*/query_generated/*.cpp", "experiments/csr_decode*/query_generated/*.h",
        "experiments/csr_decode*/query_generated/cuda_rasterizer/*.cu", "experiments/csr_decode*/query_generated/cuda_rasterizer/*.h", "experiments/csr_decode*/query_generated/build.ninja",
        "experiments/csr_decode*/stream_generated/*.cu", "experiments/csr_decode*/stream_generated/*.cpp", "experiments/csr_decode*/stream_generated/*.cuh", "experiments/csr_decode*/stream_generated/build.ninja",
        "canvas_frozen_certified_stage*.tsx", "experiments/frozen_certified_mesh*/*.json", "experiments/frozen_certified_mesh*/sources.zip", "experiments/frozen_certified_mesh*/failure.txt",
        "experiments/frozen_certified_mesh*/scan*/*.json", "experiments/frozen_certified_mesh*/scan*/0000_images.npz",
        "experiments/frozen_certified_mesh*/scan*/eval*/*.json", "experiments/frozen_certified_mesh*/scan*/eval*/eval_seeded_snapshot.py",
        "experiments/frozen_certified_mesh*/scan*/*.log", "experiments/frozen_certified_mesh*/scan*/eval*/*.log",
        "canvas_world_root_vjp_stage*.tsx", "experiments/world_root_vjp*/*.json", "experiments/world_root_vjp*/*.npz", "experiments/world_root_vjp*/sources.zip", "experiments/world_root_vjp*/failure.txt",
        "canvas_dynamic_root_stage*.tsx", "experiments/dynamic_root*/*.json", "experiments/dynamic_root*/*.npz", "experiments/dynamic_root*/sources.zip", "experiments/dynamic_root*/failure.txt",
        "experiments/dynamic_root*/generated/*.cu", "experiments/dynamic_root*/generated/*.cpp", "experiments/dynamic_root*/generated/*.cuh", "experiments/dynamic_root*/generated/build.ninja",
        "canvas_float_product_stage*.tsx", "experiments/float_product*/*.json", "experiments/float_product*/*.npz", "experiments/float_product*/sources.zip", "experiments/float_product*/failure.txt",
        "experiments/float_product*/generated/*.cu", "experiments/float_product*/generated/*.cpp", "experiments/float_product*/generated/*.cuh", "experiments/float_product*/generated/build.ninja",
        "experiments/profile*/*summary.json", "experiments/profile*/*/*integration.json",
        "results/*cuda_resources*.txt",
        "experiments/exact37_v1/seeded_eval_v2/manifest.json", "experiments/exact37_v1/seeded_eval_v2/*/eval_seeded_snapshot.py",
        "experiments/solver_mesh_v1/scan*/*/seed0/manifest.json", "experiments/solver_mesh_v1/scan*/*/seed0/eval_seeded_snapshot.py"):
        paths.update((ROOT / "research").glob(pattern))
    rasterizer = ROOT / "submodules/diff-gaussian-rasterization"
    for pattern in ("*.cu", "*.cpp", "*.h", "cuda_rasterizer/*.cu", "cuda_rasterizer/*.h"):
        paths.update(rasterizer.glob(pattern))
    files = {p.relative_to(ROOT).as_posix(): p for p in sorted(paths)}
    canvas = Path("C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx")
    if canvas.is_file():
        files["research/canvas_result_snapshot.tsx"] = canvas
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    changed = [name for name, digest in pair["source_sha256"].items() if hashes[name] != digest]
    if changed:
        raise ValueError(f"Paired-run sources changed: {changed}")
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "pair": str(args.pair.resolve()),
        "pair_stage_at_snapshot": pair["steps"][-1]["stage"], "files_sha256": hashes,
        "scope": "Selected research code, relevant production sources, diagnostic JSON, frozen ray coefficient NPZ and scientific figure exports; excludes models, input photographs, credentials and compiled binaries.",
        "limitations": ["Does not recover source versions modified before this snapshot.",
            "CUDA binaries, third-party vendor headers, input datasets and model PLYs are not embedded; recorded hashes are references.",
            "A selected Windows host audit executable is embedded inside strict_interval_host source archive for that local test only."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != hashes[name]:
                raise RuntimeError(f"Source changed during snapshot: {name}")
            archive.writestr(name, payload)
        archive.writestr("SNAPSHOT_MANIFEST.json", json.dumps(metadata, indent=2) + "\n")
    with zipfile.ZipFile(args.out) as archive:
        for name, digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ArithmeticError(f"Archive verification failed: {name}")
    metadata["archive_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    manifest_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "bytes": args.out.stat().st_size,
        "archive_sha256": metadata["archive_sha256"], "pair_sources_match": True}, indent=2))


if __name__ == "__main__":
    main()
