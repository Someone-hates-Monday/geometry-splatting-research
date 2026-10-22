#!/usr/bin/env bash
# Freeze no_dn checkpoints; sweep TSDF mesh extraction only.
#
# Sparse grid (defaults, one factor at a time vs default v002/dmax8/trunc8/c1):
#   v001, v004, dmax6, dmax10, trunc4, trunc12, c2
#
# Pass vs ./output_compare/ggs_ablate_no_dn/<scan> mesh (PSNR unchanged — same Gaussians):
#   mean overall Δ <= -5e-4
#   mean s2d Δ <= 0
#
# Usage:
#   ./scripts/run_ggs_extract_sweep.sh scan24 scan105 scan122
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

PYTHON="${PYTHON:-python}"
NO_DN_ROOT="${NO_DN_ROOT:-./output_compare/ggs_ablate_no_dn}"
OUT_ROOT="${OUT_ROOT:-./output_compare}"
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
RUN_D2S122="${RUN_D2S122:-1}"

if [ "$#" -gt 0 ]; then
  SCANS=("$@")
else
  SCANS=(scan24 scan105 scan122)
fi

mkdir -p "$METRIC_ROOT"

# tag|voxel|depth_max|trunc|clusters
CONFIGS=(
  "ex_v001|0.001|8.0|8.0|1"
  "ex_v004|0.004|8.0|8.0|1"
  "ex_dmax6|0.002|6.0|8.0|1"
  "ex_dmax10|0.002|10.0|8.0|1"
  "ex_trunc4|0.002|8.0|4.0|1"
  "ex_trunc12|0.002|8.0|12.0|1"
  "ex_c2|0.002|8.0|8.0|2"
)

prepare_model_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  cp -f "$src/cfg_args" "$dst/cfg_args"
  # Point mesh_extract / evaluate at the frozen no_dn Gaussians.
  ln -sfn "$(cd "$src/point_cloud" && pwd)" "$dst/point_cloud"
  if [ -f "$src/cameras.json" ]; then
    ln -sfn "$(cd "$src" && pwd)/cameras.json" "$dst/cameras.json"
  fi
  # Rewrite model_path inside cfg_args so logs match dst.
  "$PYTHON" - <<PY
from pathlib import Path
p = Path("$dst/cfg_args")
txt = p.read_text(encoding="utf-8")
# Namespace(... model_path='...' ...)
import re
txt2, n = re.subn(r"model_path='[^']*'", "model_path='${dst}'", txt, count=1)
if n != 1:
    raise SystemExit(f"failed to rewrite model_path in {p}")
p.write_text(txt2, encoding="utf-8")
PY
}

extract_one() {
  local tag="$1" scan="$2" voxel="$3" dmax="$4" trunc="$5" clusters="$6"
  local src="$NO_DN_ROOT/$scan"
  local dst="$OUT_ROOT/ggs_${tag}/$scan"
  echo "----- extract $tag $scan voxel=$voxel dmax=$dmax trunc=$trunc c=$clusters -----"
  prepare_model_dir "$src" "$dst"
  "$PYTHON" mesh_extract.py \
    -m "$dst" \
    --iteration -1 \
    --voxel_size "$voxel" \
    --depth_max "$dmax" \
    --trunc_voxel_multiplier "$trunc" \
    --num_cluster "$clusters"
}

compare_one() {
  local tag="$1" scan="$2"
  local base="$NO_DN_ROOT/$scan"
  local opt="$OUT_ROOT/ggs_${tag}/$scan"
  local json="$METRIC_ROOT/${scan}_no_dn_vs_${tag}.json"
  if [ ! -f "$opt/recon_post.ply" ]; then
    echo "SKIP compare $tag $scan (no recon_post.ply)"
    return 0
  fi
  echo "----- compare no_dn vs $tag $scan -----"
  "$PYTHON" scripts/compare_models.py \
    --baseline "$base" --optimized "$opt" \
    --mesh_only \
    --dtu_eval_root "$DTU_EVAL_ROOT" \
    --out "$json"
}

for scan in "${SCANS[@]}"; do
  [ -d "$NO_DN_ROOT/$scan/point_cloud" ] || { echo "ERROR: missing $NO_DN_ROOT/$scan/point_cloud"; exit 1; }
  [ -f "$NO_DN_ROOT/$scan/cfg_args" ] || { echo "ERROR: missing $NO_DN_ROOT/$scan/cfg_args"; exit 1; }
done

echo "[extract-sweep] scans=${SCANS[*]} configs=${#CONFIGS[@]}"

if [ "$SKIP_EXTRACT" != "1" ]; then
  for conf in "${CONFIGS[@]}"; do
    IFS='|' read -r tag voxel dmax trunc clusters <<<"$conf"
    for scan in "${SCANS[@]}"; do
      extract_one "$tag" "$scan" "$voxel" "$dmax" "$trunc" "$clusters"
    done
  done
fi

for conf in "${CONFIGS[@]}"; do
  IFS='|' read -r tag _ <<<"$conf"
  for scan in "${SCANS[@]}"; do
    compare_one "$tag" "$scan"
  done
done

if [ "$RUN_D2S122" = "1" ]; then
  for conf in "${CONFIGS[@]}"; do
    IFS='|' read -r tag _ <<<"$conf"
    a="$NO_DN_ROOT/scan122/recon_aligned.ply"
    b="$OUT_ROOT/ggs_${tag}/scan122/recon_aligned.ply"
    if [ -f "$a" ] && [ -f "$b" ]; then
      echo "----- d2s diagnose scan122 no_dn vs $tag -----"
      "$PYTHON" scripts/diagnose_d2s.py \
        --a "$a" --b "$b" --scan 122 --labels no_dn "$tag" \
        --out "$METRIC_ROOT/scan122_d2s_no_dn_vs_${tag}.json" || true
    fi
  done
fi

echo ""
echo "=== Extract-sweep verdict vs no_dn (mesh only; PSNR unchanged) ==="
"$PYTHON" - <<'PY'
import json
from pathlib import Path

met = Path("./output_compare/metrics")
tags = ["ex_v001", "ex_v004", "ex_dmax6", "ex_dmax10", "ex_trunc4", "ex_trunc12", "ex_c2"]
scans = ("scan24", "scan105", "scan122")
print(f"{'tag':12s} {'overall':>12s} {'s2d':>12s}  verdict")
best = None
for tag in tags:
    rows = []
    for scan in scans:
        p = met / f"{scan}_no_dn_vs_{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8")).get("delta_optimized_minus_baseline") or {}
        if isinstance(d.get("overall"), (int, float)):
            rows.append(d)
    if not rows:
        print(f"{tag:12s} {'MISSING':>12s}")
        continue
    mo = sum(r["overall"] for r in rows) / len(rows)
    ms = sum(r["mean_s2d"] for r in rows) / len(rows)
    ok = mo <= -5e-4 and ms <= 0.0 and len(rows) == 3
    mark = "PASS" if ok else "fail"
    print(f"{tag:12s} {mo:+.6e} {ms:+.6e}  {mark} (n={len(rows)})")
    for scan, r in zip(
        [s for s in scans if (met / f"{s}_no_dn_vs_{tag}.json").exists()],
        rows,
    ):
        print(
            f"  {scan}: overall={r.get('overall'):+.6e} "
            f"s2d={r.get('mean_s2d'):+.6e} d2s={r.get('mean_d2s'):+.6e}"
        )
    if best is None or mo < best[0]:
        best = (mo, ms, tag, ok)

if best:
    print(
        f"\nBEST by mean overall: {best[2]} "
        f"(overall={best[0]:+.6e}, s2d={best[1]:+.6e}) "
        f"{'PASS — adopt extract params' if best[3] else 'still FAIL — keep default no_dn extract'}"
    )
else:
    print("No metrics found.")
PY
echo "Done."
