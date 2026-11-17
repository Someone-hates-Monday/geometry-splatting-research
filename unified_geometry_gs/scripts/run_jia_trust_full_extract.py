#!/usr/bin/env python3
"""Full trust-rebuild extract: scan24 → scan105 → scan122, seed=0, no overwrite.

Uses frozen models under ggs_trust_rebuild_20261116. Writes verdict after all three.
Does not train. Does not retune tau.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCANS = ("scan24", "scan105", "scan122")
EVAL_SEED = 0
TAU = 0.9


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("CMD", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("CMD " + " ".join(cmd) + "\n")
        log.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\nEXIT {p.returncode}\n")
    if p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {' '.join(cmd)} (see {log_path})")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_seed0(results: dict, path: Path) -> None:
    if results.get("eval_seed") != 0:
        raise RuntimeError(f"{path} eval_seed must be explicit 0, got {results.get('eval_seed')!r}")


def extract_mesh(*, model: Path, write_dir: Path, trust: bool, dtu_scan: Path, log: Path) -> None:
    cmd = [
        sys.executable,
        "mesh_extract.py",
        "-m",
        str(model),
        "-s",
        str(dtu_scan),
        "-r",
        "2",
        "--iteration",
        "30000",
        "--trunc_voxel_multiplier",
        "4",
        "--voxel_size",
        "0.002",
        "--depth_max",
        "8.0",
        "--num_cluster",
        "1",
        "--write_dir",
        str(write_dir),
    ]
    if trust:
        cmd += ["--trust_fuse", "--ncc_mask_thresh", str(TAU)]
    run(cmd, log)


def eval_mesh(
    *,
    model: Path,
    mesh: Path,
    out_dir: Path,
    dtu_scan: Path,
    dtu_eval_root: Path,
    forbid: list[Path],
    log: Path,
) -> dict:
    cmd = [
        sys.executable,
        "evaluate_dtu_mesh.py",
        "-m",
        str(model),
        "-s",
        str(dtu_scan),
        "-r",
        "2",
        "--iteration",
        "30000",
        "--mesh_path",
        str(mesh),
        "--out_dir",
        str(out_dir),
        "--eval_seed",
        str(EVAL_SEED),
        "--DTU",
        str(dtu_eval_root),
    ]
    for root in forbid:
        cmd += ["--forbid_root", str(root)]
    run(cmd, log)
    results_path = out_dir / "vis" / "results.json"
    results = load_json(results_path)
    require_seed0(results, results_path)
    receipt = load_json(out_dir / "eval_receipt.json")
    if receipt.get("returncode") != 0:
        raise RuntimeError(f"eval receipt returncode != 0: {receipt}")
    return results


def write_preview_redmask(*, model: Path, dtu_scan: Path, out_dir: Path, log: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/smoke_jia_trust_fuse_scan24.py",
        "-m",
        str(model),
        "-s",
        str(dtu_scan),
        "-r",
        "2",
        "--iteration",
        "30000",
        "--out_dir",
        str(out_dir),
    ]
    run(cmd, log)


def main() -> None:
    base = ROOT / "output_compare" / "ggs_trust_rebuild_20261116"
    stamp = os.environ.get("JIA_RUN_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    dtu_root = Path(os.environ.get("UGGS_DTU_ROOT", "/mnt/c/Users/Someone/Desktop/SuGaR-main/dtu/DTU"))
    dtu_eval = Path(os.environ.get("DTU_EVAL_ROOT", str(ROOT / "dtu_eval" / "Offical_DTU_Dataset")))
    if not dtu_eval.is_dir():
        raise SystemExit(f"missing DTU eval root: {dtu_eval}")

    # Prefer preflight identity from driver; otherwise create+selfcheck here.
    if os.environ.get("JIA_SOURCE_IDENTITY_DIR"):
        identity_dir = Path(os.environ["JIA_SOURCE_IDENTITY_DIR"])
    else:
        identity_dir = base / f"source_identity_{stamp}"
        run(
            [sys.executable, "scripts/snapshot_jia_trust_source_identity.py", str(identity_dir)],
            base / "scan24" / "logs" / f"snapshot_{stamp}.log",
        )
        compat = base / "scan24" / "compat_grad_20261117_145006" / "compat_receipt.json"
        run(
            [
                sys.executable,
                "scripts/selfcheck_jia_eval_seed_and_snapshot.py",
                str(identity_dir / "source_identity_manifest.json"),
                str(compat),
            ],
            base / "scan24" / "logs" / f"selfcheck_{stamp}.log",
        )
    if not (identity_dir / "source_identity_manifest.json").is_file():
        raise SystemExit(f"missing source identity manifest under {identity_dir}")

    identity = load_json(identity_dir / "source_identity_manifest.json")
    scene_rows = []
    run_root_meta = {
        "stamp": stamp,
        "eval_seed": EVAL_SEED,
        "tau": TAU,
        "source_identity_dir": str(identity_dir),
        "loss_utils_sha256": identity["loss_utils_sha256"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scenes": {},
    }

    for scan in SCANS:
        t0 = time.time()
        sdir = base / scan
        model = sdir / "model"
        ply = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        if not ply.is_file():
            raise RuntimeError(f"missing frozen model {ply}")
        model_sha = sha256_file(ply)
        dtu_scan = dtu_root / scan
        logs = sdir / "logs"
        logs.mkdir(parents=True, exist_ok=True)

        # --- control ---
        if scan == "scan24":
            # Reuse existing seed0 control eval; keep mesh from control_trunc4
            ctrl_mesh_dir = sdir / "control_trunc4"
            ctrl_mesh = ctrl_mesh_dir / "recon_post.ply"
            ctrl_eval_dir = sdir / "control_trunc4_eval_seed0_run1_20261117_145006"
            ctrl_results_path = ctrl_eval_dir / "vis" / "results.json"
            if not ctrl_results_path.is_file():
                raise RuntimeError(f"scan24 missing seed0 control results: {ctrl_results_path}")
            ctrl_results = load_json(ctrl_results_path)
            require_seed0(ctrl_results, ctrl_results_path)
            ctrl_eval_reused = True
        else:
            ctrl_mesh_dir = sdir / f"control_trunc4_{stamp}"
            ctrl_mesh = ctrl_mesh_dir / "recon_post.ply"
            extract_mesh(
                model=model,
                write_dir=ctrl_mesh_dir,
                trust=False,
                dtu_scan=dtu_scan,
                log=logs / f"control_extract_{stamp}.log",
            )
            if not ctrl_mesh.is_file() or ctrl_mesh.stat().st_size <= 0:
                raise RuntimeError(f"empty control mesh {ctrl_mesh}")
            ctrl_eval_dir = sdir / f"control_trunc4_eval_seed0_{stamp}"
            ctrl_results = eval_mesh(
                model=model,
                mesh=ctrl_mesh,
                out_dir=ctrl_eval_dir,
                dtu_scan=dtu_scan,
                dtu_eval_root=dtu_eval,
                forbid=[model, ctrl_mesh_dir],
                log=logs / f"control_eval_seed0_{stamp}.log",
            )
            ctrl_eval_reused = False

        # Verify model unchanged after control work
        if sha256_file(ply) != model_sha:
            raise RuntimeError(f"model hash changed during {scan} control phase")

        # --- trust ---
        trust_dir = sdir / f"trust_trunc4_{stamp}"
        extract_mesh(
            model=model,
            write_dir=trust_dir,
            trust=True,
            dtu_scan=dtu_scan,
            log=logs / f"trust_extract_{stamp}.log",
        )
        trust_mesh = trust_dir / "recon_post.ply"
        if not trust_mesh.is_file() or trust_mesh.stat().st_size <= 0:
            raise RuntimeError(f"empty trust mesh {trust_mesh}")
        metrics = trust_dir / "per_view_metrics.json"
        if metrics.is_file():
            pv = load_json(metrics)
            for v in pv.get("views", []):
                if not v.get("counts_sum_eq_B", False):
                    raise RuntimeError(f"{scan} per-view counts not complete: {v.get('image_name')}")

        trust_eval_dir = sdir / f"trust_trunc4_eval_seed0_{stamp}"
        trust_results = eval_mesh(
            model=model,
            mesh=trust_mesh,
            out_dir=trust_eval_dir,
            dtu_scan=dtu_scan,
            dtu_eval_root=dtu_eval,
            forbid=[model, ctrl_mesh_dir, trust_dir, ctrl_eval_dir if scan == "scan24" else ctrl_eval_dir],
            log=logs / f"trust_eval_seed0_{stamp}.log",
        )

        if sha256_file(ply) != model_sha:
            raise RuntimeError(f"model hash changed during {scan} trust phase")

        # Fixed-view redmask (new dir)
        preview_dir = sdir / f"trust_preview_redmask_{stamp}"
        write_preview_redmask(
            model=model,
            dtu_scan=dtu_scan,
            out_dir=preview_dir,
            log=logs / f"trust_preview_{stamp}.log",
        )

        row = {
            "scan": scan,
            "model_ply_sha256": model_sha,
            "control_mesh": str(ctrl_mesh),
            "control_mesh_sha256": sha256_file(ctrl_mesh),
            "control_eval_dir": str(ctrl_eval_dir),
            "control_results": ctrl_results,
            "control_results_sha256": sha256_file(ctrl_eval_dir / "vis" / "results.json"),
            "control_eval_reused": ctrl_eval_reused,
            "trust_mesh": str(trust_mesh),
            "trust_mesh_sha256": sha256_file(trust_mesh),
            "trust_eval_dir": str(trust_eval_dir),
            "trust_results": trust_results,
            "trust_results_sha256": sha256_file(trust_eval_dir / "vis" / "results.json"),
            "preview_dir": str(preview_dir),
            "delta_overall": float(trust_results["overall"]) - float(ctrl_results["overall"]),
            "delta_s2d": float(trust_results["mean_s2d"]) - float(ctrl_results["mean_s2d"]),
            "delta_d2s": float(trust_results["mean_d2s"]) - float(ctrl_results["mean_d2s"]),
            "elapsed_sec": time.time() - t0,
        }
        scene_rows.append(row)
        run_root_meta["scenes"][scan] = row
        # Persist incremental progress
        (base / f"full_extract_progress_{stamp}.json").write_text(
            json.dumps(run_root_meta, indent=2), encoding="utf-8"
        )
        print(json.dumps({"scene_done": scan, "delta_overall": row["delta_overall"]}, indent=2), flush=True)

    # --- gates ---
    mean_d_overall = sum(r["delta_overall"] for r in scene_rows) / 3.0
    mean_d_s2d = sum(r["delta_s2d"] for r in scene_rows) / 3.0
    d122 = next(r["delta_overall"] for r in scene_rows if r["scan"] == "scan122")
    gate1 = mean_d_overall <= -5e-4
    gate2 = mean_d_s2d <= 0.0
    gate3 = d122 <= 5e-4
    passed = gate1 and gate2 and gate3
    verdict = {
        "stamp": stamp,
        "eval_seed": EVAL_SEED,
        "tau": TAU,
        "source_identity_dir": str(identity_dir),
        "loss_utils_sha256": identity["loss_utils_sha256"],
        "scenes": scene_rows,
        "mean_delta_overall": mean_d_overall,
        "mean_delta_s2d": mean_d_s2d,
        "delta_overall_scan122": d122,
        "gates": {
            "mean_delta_overall_le_-5e-4": gate1,
            "mean_delta_s2d_le_0": gate2,
            "scan122_delta_overall_le_+5e-4": gate3,
        },
        "pass": passed,
        "note_keep_frac_not_geometry_evidence": True,
        "untested_boundaries": [
            "CUDA exactly-one-valid-candidate not proven by single-positive-depth case alone",
            "gt_mask boundary pixel exhaustive cases not separately instrumented",
            "full CUDA-all-invalid injection path not separately forced",
        ],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    # CSV
    csv_lines = [
        "scan,control_overall,trust_overall,delta_overall,control_s2d,trust_s2d,delta_s2d,control_d2s,trust_d2s,delta_d2s"
    ]
    for r in scene_rows:
        c, t = r["control_results"], r["trust_results"]
        csv_lines.append(
            f"{r['scan']},{c['overall']},{t['overall']},{r['delta_overall']},"
            f"{c['mean_s2d']},{t['mean_s2d']},{r['delta_s2d']},"
            f"{c['mean_d2s']},{t['mean_d2s']},{r['delta_d2s']}"
        )
    out_verdict = base / f"verdict_{stamp}.json"
    out_csv = base / f"paired_results_{stamp}.csv"
    out_verdict.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    out_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": str(out_verdict), "pass": passed, "mean_delta_overall": mean_d_overall}, indent=2))
    # Scientific FAIL is still a completed run; exit 0 if technically valid.
    # Use exit 2 only for incomplete/invalid pipeline (raised above).


if __name__ == "__main__":
    main()
