#!/usr/bin/env bash
# Run M1 on multiple DTU scans and compare against baseline.
# Goal: verify average mesh overall improvement across scans.
#
# Usage:
#   ./scripts/run_m1_vd09_multiscan.sh [scan1 scan2 scan3 ...]
#
# Examples:
#   ./scripts/run_m1_vd09_multiscan.sh scan24 scan105 scan122
#   ./scripts/run_m1_vd09_multiscan.sh   # default: scan24 scan105 scan122
#   LAMBDA_DEPTH_OCC=0.06 TAG=m1_vd06 ./scripts/run_m1_vd09_multiscan.sh scan24 scan105 scan122
#
# Assumptions:
# - Baseline outputs already exist at: ./output_compare/ggs_baseline/<scan>
# - DTU eval data exists at: ./dtu_eval/Offical_DTU_Dataset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
BASELINE_ROOT="${BASELINE_ROOT:-./output_compare/ggs_baseline}"
LAMBDA_OCC="${LAMBDA_OCC:-0.02}"
LAMBDA_DEPTH_OCC="${LAMBDA_DEPTH_OCC:-0.09}"
REG_FROM_ITER="${REG_FROM_ITER:-7000}"
TAG="${TAG:-m1_lo${LAMBDA_OCC/./}_ld${LAMBDA_DEPTH_OCC/./}_r${REG_FROM_ITER}}"
OPT_ROOT="${OPT_ROOT:-./output_compare/${TAG}}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"

if [ "$#" -gt 0 ]; then
  SCANS=("$@")
else
  SCANS=(scan24 scan105 scan122)
fi

mkdir -p "$OPT_ROOT" "$METRIC_ROOT"

echo "[m1-multiscan] scans: ${SCANS[*]}"
echo "[m1-multiscan] tag=$TAG"
echo "[m1-multiscan] lambda_occ=$LAMBDA_OCC lambda_depth_occ=$LAMBDA_DEPTH_OCC reg_from_iter=$REG_FROM_ITER"
echo "[m1-multiscan] baseline_root=$BASELINE_ROOT"
echo "[m1-multiscan] optimized_root=$OPT_ROOT"
echo "[m1-multiscan] metrics_root=$METRIC_ROOT"

for scan in "${SCANS[@]}"; do
  echo
  echo "===== [$scan] train $TAG ====="
  if [ ! -d "$BASELINE_ROOT/$scan" ]; then
    echo "ERROR: baseline missing: $BASELINE_ROOT/$scan"
    echo "Train baseline first or override BASELINE_ROOT."
    exit 1
  fi

  LAMBDA_OCC="$LAMBDA_OCC" \
  LAMBDA_DEPTH_OCC="$LAMBDA_DEPTH_OCC" \
  REG_FROM_ITER="$REG_FROM_ITER" \
    "$SCRIPT_DIR/run_m1.sh" "$UGGS_DTU_ROOT/$scan" "$OPT_ROOT/$scan"

  echo "===== [$scan] compare ====="
  python scripts/compare_models.py \
    --baseline "$BASELINE_ROOT/$scan" \
    --optimized "$OPT_ROOT/$scan" \
    --dtu_eval_root "$DTU_EVAL_ROOT" \
    --out "$METRIC_ROOT/${scan}_baseline_vs_${TAG}.json"
done

echo
echo "===== Summary ($TAG - baseline) ====="
python - <<'PY'
import json
import os
from pathlib import Path

metric_root = Path("./output_compare/metrics")
tag = os.environ.get("TAG", "m1")
suffix = f"_baseline_vs_{tag}.json"
files = sorted(metric_root.glob(f"scan*{suffix}"))
if not files:
    raise SystemExit("No metric files found.")

rows = []
for p in files:
    data = json.loads(p.read_text(encoding="utf-8"))
    d = data.get("delta_optimized_minus_baseline", {})
    rows.append(
        (
            p.name.replace(suffix, ""),
            d.get("overall"),
            d.get("mean_d2s"),
            d.get("mean_s2d"),
            d.get("l1"),
            d.get("psnr"),
            d.get("ssim"),
        )
    )

def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None

for r in rows:
    print(
        f"{r[0]} | overall={r[1]:+.6f} d2s={r[2]:+.6f} s2d={r[3]:+.6f} "
        f"| l1={r[4]:+.6f} psnr={r[5]:+.6f} ssim={r[6]:+.6f}"
    )

print("-" * 90)
print(
    "avg   | overall={:+.6f} d2s={:+.6f} s2d={:+.6f} | l1={:+.6f} psnr={:+.6f} ssim={:+.6f}".format(
        mean([r[1] for r in rows]),
        mean([r[2] for r in rows]),
        mean([r[3] for r in rows]),
        mean([r[4] for r in rows]),
        mean([r[5] for r in rows]),
        mean([r[6] for r in rows]),
    )
)
PY
