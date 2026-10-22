#!/usr/bin/env bash
# baseline (./output/dtu/<SCAN>) vs edge_dn — orbit9 并排 + 顶点 STL 距离着色，复制到单独报告目录。
#
#   SCAN=scan105 ./scripts/visual_baseline_vs_edge_dn.sh
#   SCAN=scan65  ./scripts/visual_baseline_vs_edge_dn.sh
#
# Env: DTU, REPORT_COPY (default ./report_figures_<SCAN>_edge_dn_vs_baseline),
#      VMAX_VERTEX, SKIP_SNAP, STITCH_ONLY, PYTHON — 同 visual_scan65_edge_dn_vs_baseline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f configs/dtu_path.env ]; then
  # shellcheck source=/dev/null
  source configs/dtu_path.env
fi

SCAN="${SCAN:-scan105}"
PYTHON="${PYTHON:-python}"
DTU="${DTU:-${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}}"
REPORT_COPY="${REPORT_COPY:-./report_figures_${SCAN}_edge_dn_vs_baseline}"
VMAX="${VMAX_VERTEX:-6}"

EXTRA=(--scans "$SCAN" --variants baseline edge_dn --dtu_dir "$DTU" --vmax_vertex "$VMAX")
if [ "${SKIP_SNAP:-0}" = "1" ]; then
  EXTRA+=(--skip_snapshots)
fi
if [ "${STITCH_ONLY:-0}" = "1" ]; then
  EXTRA+=(--stitch_only)
fi

"$PYTHON" scripts/visual_compare_output_dtu.py "${EXTRA[@]}"

SRC="$ROOT/report_figures_edge_compare/$SCAN"
if [ ! -d "$SRC" ]; then
  echo "ERROR: missing $SRC"
  exit 1
fi

mkdir -p "$REPORT_COPY"
cp -rf "$SRC/." "$REPORT_COPY/"
echo ""
echo "===== 报告用汇总目录 ($SCAN baseline vs edge_dn) ====="
echo "$REPORT_COPY"
echo "  orbit9_rows/row_*.png — 左 baseline | 右 edge_dn"
echo "  orbit9_rows/_all_views_stacked.png"
echo "  vertex_err_d2s/ — 着色 PLY + colorbar + compare_vertex_d2s_summary.json"
echo ""
echo "顶点 JSON 中 delta_mean_A_minus_B 为 baseline−edge_dn 的顶点均值 proxy；"
echo "mesh 优劣以 evaluate_dtu_mesh / compare_models 为准。"
