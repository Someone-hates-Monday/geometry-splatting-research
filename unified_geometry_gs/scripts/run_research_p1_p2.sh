#!/usr/bin/env bash
# RESEARCH_PROTOCOL P1 + P2 (no training).
#   P1: σ_z on frozen no_dn
#   P2: extract m=2 and m=6 (t=0.004, 0.012); reuse existing m=4,8,12
#   G2: judge t* vs σ_z
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
SCANS=(scan24 scan105 scan122)

mkdir -p "$METRIC_ROOT"

prepare_model_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  cp -f "$src/cfg_args" "$dst/cfg_args"
  ln -sfn "$(cd "$src/point_cloud" && pwd)" "$dst/point_cloud"
  if [ -f "$src/cameras.json" ]; then
    ln -sfn "$(cd "$src" && pwd)/cameras.json" "$dst/cameras.json"
  fi
  "$PYTHON" - <<PY
from pathlib import Path
import re
p = Path("$dst/cfg_args")
txt = p.read_text(encoding="utf-8")
txt2, n = re.subn(r"model_path='[^']*'", "model_path='${dst}'", txt, count=1)
if n != 1:
    raise SystemExit(f"failed to rewrite model_path in {p}")
p.write_text(txt2, encoding="utf-8")
PY
}

extract_m() {
  local m="$1" scan="$2"
  local src="$NO_DN_ROOT/$scan"
  local dst="$OUT_ROOT/ggs_ex_m${m}/$scan"
  echo "----- P2 extract m=$m $scan (t=$(python -c "print($m*0.002)")) -----"
  prepare_model_dir "$src" "$dst"
  "$PYTHON" mesh_extract.py \
    -m "$dst" \
    --iteration -1 \
    --voxel_size 0.002 \
    --depth_max 8.0 \
    --trunc_voxel_multiplier "$m" \
    --num_cluster 1
  # Absolute mesh metrics (results.json)
  "$PYTHON" evaluate_dtu_mesh.py -m "$dst" --iteration -1 --DTU "$DTU_EVAL_ROOT"
}

# Ensure legacy m=4 / m=12 tags also have results.json (may already)
ensure_eval() {
  local dir="$1"
  if [ -f "$dir/recon_post.ply" ] && [ ! -f "$dir/vis/results.json" ]; then
    echo "----- eval missing results: $dir -----"
    "$PYTHON" evaluate_dtu_mesh.py -m "$dir" --iteration -1 --DTU "$DTU_EVAL_ROOT"
  fi
}

echo "========== P1: σ_z =========="
"$PYTHON" scripts/measure_sigma_z.py \
  --models \
    "$NO_DN_ROOT/scan24" \
    "$NO_DN_ROOT/scan105" \
    "$NO_DN_ROOT/scan122" \
  --seed 0 \
  --out "$METRIC_ROOT/sigma_z_no_dn.json"

echo "========== P2: extract m=2,6 + ensure evals =========="
if [ "$SKIP_EXTRACT" != "1" ]; then
  for m in 2 6; do
    for scan in "${SCANS[@]}"; do
      extract_m "$m" "$scan"
    done
  done
fi

# Ensure absolute overalls exist for m=4 (ex_trunc4), m=8 (no_dn), m=12 (ex_trunc12)
for scan in "${SCANS[@]}"; do
  ensure_eval "$NO_DN_ROOT/$scan"
  ensure_eval "$OUT_ROOT/ggs_ex_trunc4/$scan"
  ensure_eval "$OUT_ROOT/ggs_ex_trunc12/$scan"
  ensure_eval "$OUT_ROOT/ggs_ex_m2/$scan"
  ensure_eval "$OUT_ROOT/ggs_ex_m6/$scan"
done

echo "========== G2 judgment =========="
"$PYTHON" scripts/judge_g2.py \
  --sigma_z "$METRIC_ROOT/sigma_z_no_dn.json" \
  --out_root "$OUT_ROOT" \
  --out "$METRIC_ROOT/g2_verdict.json"

echo "=== P1+P2+G2 finished ==="
cat "$METRIC_ROOT/g2_verdict.json"
