#!/usr/bin/env python3
"""Build read-only seal manifest for Jia trust-rebuild (no GPU, no overwrite of sources)."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/mnt/c/Users/Someone/Desktop/SuGaR-main")
UGGS = REPO / "unified_geometry_gs"
BASE = UGGS / "output_compare" / "ggs_trust_rebuild_20261116"
STAMP = "20261117_154810"
HANDOFF = REPO / "handoff"
OUT_DIR = BASE / f"seal_{STAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add(entries: list, rel: str, path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries.append(
        {
            "role": role,
            "relpath": rel.replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "abspath": str(path),
        }
    )


def main() -> None:
    entries: list[dict] = []
    # Handoff docs
    for name, role in [
        ("JIA_TRUST_REBUILD_PROTOCOL_20261116.md", "protocol"),
        ("JIA_TRUST_REBUILD_ORDER_DEVIATION.md", "order_deviation"),
        ("JIA_LOCAL_RECOVERY_AUDIT.md", "asset_missing_audit"),
        ("JIA_TRUST_REBUILD_P1_RESUBMIT_20261117.md", "p1_resubmit"),
        ("JIA_TRUST_REBUILD_P1_REVIEW_DECISION_20261117.md", "p1_decision"),
        ("JIA_TRUST_REBUILD_FULL_RESULT_20261117.md", "full_result_note"),
        ("JIA_TRUST_REBUILD_FINAL_REVIEW_20261117.md", "final_review"),
        ("JIA_TRUST_REBUILD_FINAL_AUDIT_20261117.json", "final_audit"),
        ("JIA_TRUST_REBUILD_CLOSEOUT_20261117.md", "closeout"),
        ("JIA_TRUST_REBUILD_STATUS.md", "status"),
        ("JIA_TRUST_FUSE_PHASE1_BLOCKED.md", "phase1_blocked"),
        ("PLAN_JIA_TRUST_FUSE.md", "plan"),
        ("TEAM_PLAN_2026_11_16.md", "team_plan"),
    ]:
        add(entries, f"handoff/{name}", HANDOFF / name, role)

    # Source identity snapshot
    id_dir = BASE / f"source_identity_{STAMP}"
    man = id_dir / "source_identity_manifest.json"
    add(entries, f"output_compare/ggs_trust_rebuild_20261116/source_identity_{STAMP}/source_identity_manifest.json", man, "source_identity_manifest")
    id_man = json.loads(man.read_text(encoding="utf-8"))
    for e in id_man["entries"]:
        p = id_dir / e["snapshot_relpath"]
        add(
            entries,
            f"output_compare/ggs_trust_rebuild_20261116/source_identity_{STAMP}/{e['snapshot_relpath']}",
            p,
            f"source_snapshot:{e['relpath']}",
        )

    # Models
    for scan in ("scan24", "scan105", "scan122"):
        ply = BASE / scan / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        add(entries, f"output_compare/ggs_trust_rebuild_20261116/{scan}/model/point_cloud/iteration_30000/point_cloud.ply", ply, f"model_ply:{scan}")
        cfg = BASE / scan / "model" / "cfg_args"
        if cfg.is_file():
            add(entries, f"output_compare/ggs_trust_rebuild_20261116/{scan}/model/cfg_args", cfg, f"model_cfg:{scan}")

    # Control / trust meshes + eval JSON + receipts + preview
    # scan24 control mesh is unstamped dir; seed0 eval reused
    pairs = {
        "scan24": {
            "control_mesh": BASE / "scan24" / "control_trunc4" / "recon_post.ply",
            "control_results": BASE / "scan24" / "control_trunc4_eval_seed0_run1_20261117_145006" / "vis" / "results.json",
            "control_receipt": BASE / "scan24" / "control_trunc4_eval_seed0_run1_20261117_145006" / "eval_receipt.json",
            "trust_mesh": BASE / "scan24" / f"trust_trunc4_{STAMP}" / "recon_post.ply",
            "trust_results": BASE / "scan24" / f"trust_trunc4_eval_seed0_{STAMP}" / "vis" / "results.json",
            "trust_receipt": BASE / "scan24" / f"trust_trunc4_eval_seed0_{STAMP}" / "eval_receipt.json",
            "per_view": BASE / "scan24" / f"trust_trunc4_{STAMP}" / "per_view_metrics.json",
            "redmask": BASE / "scan24" / f"trust_preview_redmask_{STAMP}" / "smoke_redmask.png",
            "preview_receipt": BASE / "scan24" / f"trust_preview_redmask_{STAMP}" / "smoke_receipt.json",
        },
        "scan105": {
            "control_mesh": BASE / "scan105" / f"control_trunc4_{STAMP}" / "recon_post.ply",
            "control_results": BASE / "scan105" / f"control_trunc4_eval_seed0_{STAMP}" / "vis" / "results.json",
            "control_receipt": BASE / "scan105" / f"control_trunc4_eval_seed0_{STAMP}" / "eval_receipt.json",
            "trust_mesh": BASE / "scan105" / f"trust_trunc4_{STAMP}" / "recon_post.ply",
            "trust_results": BASE / "scan105" / f"trust_trunc4_eval_seed0_{STAMP}" / "vis" / "results.json",
            "trust_receipt": BASE / "scan105" / f"trust_trunc4_eval_seed0_{STAMP}" / "eval_receipt.json",
            "per_view": BASE / "scan105" / f"trust_trunc4_{STAMP}" / "per_view_metrics.json",
            "redmask": BASE / "scan105" / f"trust_preview_redmask_{STAMP}" / "smoke_redmask.png",
            "preview_receipt": BASE / "scan105" / f"trust_preview_redmask_{STAMP}" / "smoke_receipt.json",
        },
        "scan122": {
            "control_mesh": BASE / "scan122" / f"control_trunc4_{STAMP}" / "recon_post.ply",
            "control_results": BASE / "scan122" / f"control_trunc4_eval_seed0_{STAMP}" / "vis" / "results.json",
            "control_receipt": BASE / "scan122" / f"control_trunc4_eval_seed0_{STAMP}" / "eval_receipt.json",
            "trust_mesh": BASE / "scan122" / f"trust_trunc4_{STAMP}" / "recon_post.ply",
            "trust_results": BASE / "scan122" / f"trust_trunc4_eval_seed0_{STAMP}" / "vis" / "results.json",
            "trust_receipt": BASE / "scan122" / f"trust_trunc4_eval_seed0_{STAMP}" / "eval_receipt.json",
            "per_view": BASE / "scan122" / f"trust_trunc4_{STAMP}" / "per_view_metrics.json",
            "redmask": BASE / "scan122" / f"trust_preview_redmask_{STAMP}" / "smoke_redmask.png",
            "preview_receipt": BASE / "scan122" / f"trust_preview_redmask_{STAMP}" / "smoke_receipt.json",
        },
    }
    for scan, paths in pairs.items():
        for key, p in paths.items():
            rel = str(p.relative_to(REPO)).replace("\\", "/")
            add(entries, rel, p, f"{scan}:{key}")

    # Verdict / CSV / logs / compat
    for name, role in [
        (f"verdict_{STAMP}.json", "verdict"),
        (f"paired_results_{STAMP}.csv", "paired_csv"),
        (f"full_extract_{STAMP}.log", "full_extract_log"),
        (f"full_extract_progress_{STAMP}.json", "progress"),
    ]:
        p = BASE / name
        if p.is_file():
            add(entries, f"output_compare/ggs_trust_rebuild_20261116/{name}", p, role)

    compat = BASE / "scan24" / "compat_grad_20261117_145006" / "compat_receipt.json"
    add(entries, "output_compare/ggs_trust_rebuild_20261116/scan24/compat_grad_20261117_145006/compat_receipt.json", compat, "compat_receipt")

    # Verify CSV vs JSON deltas
    verdict = json.loads((BASE / f"verdict_{STAMP}.json").read_text(encoding="utf-8"))
    csv_path = BASE / f"paired_results_{STAMP}.csv"
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    csv_checks = []
    for r, s in zip(rows, verdict["scenes"]):
        assert r["scan"] == s["scan"]
        for k_csv, k_v in [
            ("delta_overall", "delta_overall"),
            ("delta_s2d", "delta_s2d"),
            ("delta_d2s", "delta_d2s"),
        ]:
            a = float(r[k_csv])
            b = float(s[k_v])
            ok = a == b
            csv_checks.append({"scan": s["scan"], "field": k_csv, "csv": a, "verdict": b, "equal": ok})
            if not ok:
                raise SystemExit(f"CSV/verdict mismatch {s['scan']} {k_csv}: {a} vs {b}")

    # Aggregate keep from per-view (weighted by n_B)
    keep_stats = {}
    for scan, paths in pairs.items():
        pv = json.loads(paths["per_view"].read_text(encoding="utf-8"))
        views = pv["views"]
        n_B = sum(int(v["n_B"]) for v in views)
        n_keep = sum(int(v["n_keep"]) for v in views)
        keep_stats[scan] = {
            "n_views": len(views),
            "n_B_sum": n_B,
            "n_keep_sum": n_keep,
            "aggregate_keep_fraction": (n_keep / n_B) if n_B else None,
            "note": "Weighted by positive-depth B pixels across all train views; not single-preview keep.",
        }

    # Repo dirty summary (paths only)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(REPO), text=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    diff_stat = subprocess.check_output(
        ["git", "diff", "--stat", "--", "unified_geometry_gs/evaluate_dtu_mesh.py", "unified_geometry_gs/mesh_extract.py", "unified_geometry_gs/utils/loss_utils.py"],
        cwd=str(REPO),
        text=True,
    )

    seal = {
        "date": datetime.now(timezone.utc).isoformat(),
        "classification": "VALID_SCIENTIFIC_FAIL",
        "jia_task_status": "完成验证，未过线",
        "research_goal_complete": False,
        "stamp": STAMP,
        "git_head": head,
        "eval_seed": 0,
        "tau": 0.9,
        "views_per_scan": {"scan24": 49, "scan105": 64, "scan122": 64},
        "aggregate_keep_fraction": keep_stats,
        "preview_keep_fraction_scan24_not_scene_rate": 0.7476172565840542,
        "mean_delta_overall": verdict["mean_delta_overall"],
        "mean_delta_s2d": verdict["mean_delta_s2d"],
        "delta_overall_scan122": verdict["delta_overall_scan122"],
        "gates": verdict["gates"],
        "pass": False,
        "csv_verdict_delta_checks": csv_checks,
        "n_entries": len(entries),
        "entries": entries,
        "git_status_porcelain": status.splitlines(),
        "tracked_code_diff_stat": diff_stat.strip(),
        "limitations": [
            "CUDA exactly-one-valid-candidate not proven by single-positive-depth case alone",
            "gt_mask boundary pixel exhaustive cases not separately instrumented",
            "full CUDA-all-invalid injection path not separately forced",
            "No GPU/Chamfer recompute in seal step; relies on prior run evidence + independent audit JSON",
        ],
        "allowed_conclusion_zh": (
            "在重新构建并冻结的scan24、105、122模型上，我们将固定最近候选近邻、tau=0.9的强可信掩码用于trunc4融合。"
            "相对同模型无掩码对照，平均overall增加0.004532 mm，平均s2d增加0.027515 mm，未达到三条预设门槛，因此不采用此改动。"
        ),
        "not_authorized": [
            "no local truncation second cut",
            "no tau/neighbor retune",
            "no scene expansion",
            "no retrain",
            "no auto-upload",
            "do not modify original 90-volume Release COMPLETE receipt",
        ],
    }
    out = OUT_DIR / "SEAL_MANIFEST.json"
    out.write_text(json.dumps(seal, indent=2, ensure_ascii=False), encoding="utf-8")
    # compact index without embedding every large path twice
    index = {
        "seal_manifest": str(out),
        "seal_manifest_sha256": sha256(out),
        "verdict": str(BASE / f"verdict_{STAMP}.json"),
        "csv": str(csv_path),
        "source_identity": str(id_dir),
        "n_entries": len(entries),
        "jia_task_status": seal["jia_task_status"],
        "pass": False,
    }
    idx_path = HANDOFF / "JIA_TRUST_REBUILD_SEAL_INDEX_20261117.json"
    idx_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
