#!/usr/bin/env bash
# Vertex-error coloring (distance mesh vertex → DTU STL) for many methods on one scan.
# Same --vmax / colormap across all exports so colors are comparable.
#
# Requires: conda uggs, DTU SampleSet (Points/stl/stlXXX_total.ply), sklearn.
#
# Usage:
#   cd unified_geometry_gs && conda activate uggs && source configs/dtu_path.env
#   chmod +x scripts/run_vertex_error_scan_suite.sh
#
#   # scan24 默认；包含 baseline + 你们的 edge/pm 实验 +（若存在）M1-3 + SuGaR
#   ./scripts/run_vertex_error_scan_suite.sh
#
#   SCAN=scan105 ./scripts/run_vertex_error_scan_suite.sh
#
# SuGaR（二选一）:
#   SUGAR_MODEL_DIR=./output_compare/sugar/scan24 ./scripts/run_vertex_error_scan_suite.sh
#   SUGAR_ALIGNED_PLY=/path/to/sugar_recon_aligned.ply ./scripts/run_vertex_error_scan_suite.sh
#
# 无 cfg_args 的目录（仅 recon_post.ply）需要对齐到 DTU，需提供 COLMAP 扫描根目录：
#   EXTERNAL_MESH_SOURCE_PATH=/path/to/scan24   # 精确路径（推荐）
#   或 EXTERNAL_MESH_SOURCE_ROOT=$UGGS_DTU_ROOT # 与 SCAN 拼接为 $ROOT/$SCAN
#
# Env:
#   DTU=./dtu_eval/Offical_DTU_Dataset
#   OUT_DIR=./report_figures_vertex_err/scan24
#   VMAX_VERTEX=6
#   SKIP_EVAL=1   # do not run evaluate_dtu_mesh.py when recon_aligned.ply missing
#   PYTHON=python
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f configs/dtu_path.env ]; then
  # shellcheck source=/dev/null
  source configs/dtu_path.env
fi

PYTHON="${PYTHON:-python}"
SCAN="${SCAN:-scan24}"
NUM="${SCAN#scan}"
DTU="${DTU:-./dtu_eval/Offical_DTU_Dataset}"
OUT_DIR="${OUT_DIR:-./report_figures_vertex_err/${SCAN}}"
VMAX="${VMAX_VERTEX:-6}"
CMAP="${COLORMAP_VERTEX:-turbo}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "$OUT_DIR"
CB="$OUT_DIR/colorbar_${CMAP}_vmax${VMAX}.png"
META="$OUT_DIR/run_manifest.txt"
: >"$META"
echo "scan=$SCAN dtu=$DTU vmax=$VMAX out=$OUT_DIR" | tee -a "$META"

ensure_aligned() {
  local dir="$1"
  local num="${SCAN#scan}"
  local src=""
  [ -d "$dir" ] || return 1
  if [ -f "$dir/recon_aligned.ply" ]; then
    echo "$dir/recon_aligned.ply"
    return 0
  fi
  [ -f "$dir/recon_post.ply" ] || return 1
  if [ "$SKIP_EVAL" = "1" ]; then
    echo "SKIP_EVAL: missing recon_aligned under $dir" | tee -a "$META" >&2
    return 1
  fi
  if [ -f "$dir/cfg_args" ]; then
    echo "evaluate_dtu_mesh -> $dir" | tee -a "$META" >&2
    "$PYTHON" evaluate_dtu_mesh.py -m "$dir" --iteration -1 --DTU "$DTU" || return 1
  else
    src="${EXTERNAL_MESH_SOURCE_PATH:-}"
    if [ -z "$src" ] && [ -n "${EXTERNAL_MESH_SOURCE_ROOT:-}" ]; then
      src="${EXTERNAL_MESH_SOURCE_ROOT%/}/${SCAN}"
    fi
    if [ ! -d "$src/sparse/0" ]; then
      echo "external mesh: missing COLMAP at $src (set EXTERNAL_MESH_SOURCE_PATH or EXTERNAL_MESH_SOURCE_ROOT)" | tee -a "$META" >&2
      return 1
    fi
    echo "evaluate_external_mesh_dtu -> $dir (source_path=$src)" | tee -a "$META" >&2
    "$PYTHON" evaluate_external_mesh_dtu.py \
      --source_path "$src" \
      --mesh_in "$dir/recon_post.ply" \
      --out_dir "$dir" \
      --scan_id "$num" \
      --DTU "$DTU" || return 1
  fi
  [ -f "$dir/recon_aligned.ply" ] && echo "$dir/recon_aligned.ply"
}

export_one() {
  local label="$1"
  local mesh="$2"
  if [ ! -f "$mesh" ]; then
    echo "SKIP $label (no file $mesh)" | tee -a "$META"
    return 0
  fi
  local out_ply="$OUT_DIR/${label}_d2s_vertex_${CMAP}_vmax${VMAX}.ply"
  echo "---- $label <- $mesh ----" | tee -a "$META"
  if [ ! -f "$CB" ]; then
    "$PYTHON" scripts/export_mesh_vertex_error_dtu.py \
      --dtu_dir "$DTU" \
      --scan "$NUM" \
      --mesh "$mesh" \
      --out "$out_ply" \
      --vmax "$VMAX" \
      --colormap "$CMAP" \
      --colorbar_out "$CB"
  else
    "$PYTHON" scripts/export_mesh_vertex_error_dtu.py \
      --dtu_dir "$DTU" \
      --scan "$NUM" \
      --mesh "$mesh" \
      --out "$out_ply" \
      --vmax "$VMAX" \
      --colormap "$CMAP"
  fi
}

# --- Built-in candidates (UGGS under ./output/dtu; historical under ./output_compare) ---
declare -a JOBS=(
  "uggs_baseline:./output/dtu/${SCAN}"
  "edge_dn:./output/dtu/${SCAN}_edge_dn"
  "pm_edge_go:./output/dtu/${SCAN}_pm_edge_go"
  "pm_edge_soft:./output/dtu/${SCAN}_pm_edge_soft"
  "pm_edge_full:./output/dtu/${SCAN}_pm_edge"
  "ggs_baseline:./output_compare/ggs_baseline/${SCAN}"
  "m1_ablate_03_full:./output_compare/m1_ablate_03_full/${SCAN}"
  "m1_b_mesh_align:./output_compare/m1_b_mesh_align/${SCAN}"
)

for entry in "${JOBS[@]}"; do
  label="${entry%%:*}"
  dir="${entry#*:}"
  mesh="$(ensure_aligned "$dir" || true)"
  if [ -n "${mesh:-}" ]; then
    export_one "$label" "$mesh"
  fi
done

# --- SuGaR: directory (same layout as UGGS model_path) ---
SUGAR_MODEL_DIR="${SUGAR_MODEL_DIR:-./output_compare/sugar/${SCAN}}"
if [ -n "${SUGAR_ALIGNED_PLY:-}" ]; then
  export_one "sugar" "$SUGAR_ALIGNED_PLY"
elif [ -d "$SUGAR_MODEL_DIR" ]; then
  sm="$(ensure_aligned "$SUGAR_MODEL_DIR" || true)"
  if [ -n "${sm:-}" ]; then
    export_one "sugar" "$sm"
  fi
else
  echo "SKIP sugar (set SUGAR_MODEL_DIR or SUGAR_ALIGNED_PLY)" | tee -a "$META"
fi

echo "" | tee -a "$META"
echo "Done. Colored PLY + shared colorbar:" | tee -a "$META"
echo "  $OUT_DIR" | tee -a "$META"
echo "Open in MeshLab / CloudCompare with $CB for scale." | tee -a "$META"
