#!/usr/bin/env bash
# 兼容入口：固定 scan65（mesh 上 edge_dn 改善最明显的展示场景）。
# 实现见 visual_baseline_vs_edge_dn.sh
#
#   SCAN=scan65 REPORT_COPY=./report_figures_scan65_edge_dn_vs_baseline \
#     ./scripts/visual_baseline_vs_edge_dn.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env SCAN=scan65 REPORT_COPY="${REPORT_COPY:-./report_figures_scan65_edge_dn_vs_baseline}" \
  "$SCRIPT_DIR/visual_baseline_vs_edge_dn.sh"
