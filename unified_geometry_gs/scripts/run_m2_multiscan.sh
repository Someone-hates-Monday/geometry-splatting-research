#!/usr/bin/env bash
# Run M2(w_geo) on multiple DTU scans and compare against baseline.
#
# Usage:
#   ./scripts/run_m2_multiscan.sh [scan1 scan2 scan3 ...]
#
# Example:
#   W_GEO_BETA=0.2 W_GEO_MIN=0.8 TAG=m2_wgeo_safe \
#     ./scripts/run_m2_multiscan.sh scan24 scan105 scan122
#
# Assumptions:
# - Baseline outputs exist at ./output_compare/ggs_baseline/<scan>
# - DTU eval data exists at ./dtu_eval/Offical_DTU_Dataset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
BASELINE_ROOT="${BASELINE_ROOT:-./output_compare/ggs_baseline}"
W_GEO_BETA="${W_GEO_BETA:-0.2}"
W_GEO_MIN="${W_GEO_MIN:-0.8}"
TAG="${TAG:-m2_wgeo_b${W_GEO_BETA/./}_min${W_GEO_MIN/./}}"
OPT_ROOT="${OPT_ROOT:-./output_compare/${TAG}}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"

if [ "$#" -gt 0 ]; then
  SCANS=("$@")
else
  SCANS=(scan24 scan105 scan122)
fi

mkdir -p "$OPT_ROOT" "$METRIC_ROOT"

echo "[m2-multiscan] scans: ${SCANS[*]}"
echo "[m2-multiscan] tag=$TAG"
echo "[m2-multiscan] w_geo_beta=$W_GEO_BETA w_geo_min=$W_GEO_MIN"
echo "[m2-multiscan] baseline_root=$BASELINE_ROOT"
echo "[m2-multiscan] optimized_root=$OPT_ROOT"
echo "[m2-multiscan] metrics_root=$METRIC_ROOT"

for scan in "${SCANS[@]}"; do
  echo
  echo "===== [$scan] train $TAG ====="
  if [ ! -d "$BASELINE_ROOT/$scan" ]; then
    echo "ERROR: baseline missing: $BASELINE_ROOT/$scan"
    echo "Train baseline first or override BASELINE_ROOT."
    exit 1
  fi

  W_GEO_BETA="$W_GEO_BETA" W_GEO_MIN="$W_GEO_MIN" \
    "$SCRIPT_DIR/run_m2.sh" "$UGGS_DTU_ROOT/$scan" "$OPT_ROOT/$scan"

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
tag = os.environ["TAG"]
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

