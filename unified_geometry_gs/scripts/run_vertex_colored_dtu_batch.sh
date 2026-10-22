#!/usr/bin/env bash
# DTU 顶点着色网格（每个模型单独一个彩色 PLY），批量多 scan。
# 不做 orbit9、不做三联对比目录；每个 scan 输出到 report_figures_vertex_colored/<scan>/。
#
# 依赖: conda uggs, DTU Points/stl, sklearn, open3d
#
# Usage（已在 unified_geometry_gs 目录内，不要再 cd unified_geometry_gs）:
#   cd /path/to/SuGaR-main/unified_geometry_gs
#   source configs/dtu_path.env
#   export PYTHON=python
#   chmod +x scripts/run_vertex_colored_dtu_batch.sh
#
#   SCANS="scan37 scan55 scan65 scan105 scan110" ./scripts/run_vertex_colored_dtu_batch.sh
#
# 仅 train 输出目录下的 baseline vs 边缘模型（与 compare_dtu_output_vs_baseline / scan65_edge_dn 一致）:
#   OUT_ROOT=./report_figures_vertex_edge_dn_vs_baseline \
#   VERTEX_ONLY_OUTPUT_DTU=1 OUTPUT_DTU_SUFFIXES="edge_dn" \
#   SCANS="scan24 scan105" ./scripts/run_vertex_colored_dtu_batch.sh
#
# Env:
#   DTU=./dtu_eval/Offical_DTU_Dataset
#   OUT_ROOT=./report_figures_vertex_colored   # 每个 scan -> $OUT_ROOT/<scan>/
#   VMAX_VERTEX=6   COLORMAP_VERTEX=turbo
#   SKIP_EVAL=1     # 已有 recon_aligned.ply 时不跑 evaluate_dtu_mesh / external
#   EXTERNAL_MESH_SOURCE_ROOT=$UGGS_DTU_ROOT   # 仅 recon_post、无 cfg_args 时需要
#   SUGAR_MODEL_DIR=./output_compare/sugar/scan24  # 可选；按 scan 循环时用默认 .../sugar/<scan>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f configs/dtu_path.env ]; then
  # shellcheck source=/dev/null
  source configs/dtu_path.env
fi

PYTHON="${PYTHON:-python}"
DTU="${DTU:-./dtu_eval/Offical_DTU_Dataset}"
SCANS="${SCANS:-scan105}"
OUT_ROOT="${OUT_ROOT:-./report_figures_vertex_colored}"
VMAX="${VMAX_VERTEX:-6}"
CMAP="${COLORMAP_VERTEX:-turbo}"
SKIP_EVAL="${SKIP_EVAL:-0}"
# 若=1：只处理 ./output/dtu/<scan> 与 ./output/dtu/<scan>_<suffix>，忽略 output_compare / SuGaR
VERTEX_ONLY_OUTPUT_DTU="${VERTEX_ONLY_OUTPUT_DTU:-0}"
# 空格分隔，例如 edge_dn 或 edge_dn pm_edge_soft（需对应目录存在）
OUTPUT_DTU_SUFFIXES="${OUTPUT_DTU_SUFFIXES:-}"

ensure_aligned() {
  local scan="$1"
  local dir="$2"
  local num="${scan#scan}"
  local src=""
  [ -d "$dir" ] || return 1
  if [ -f "$dir/recon_aligned.ply" ]; then
    echo "$dir/recon_aligned.ply"
    return 0
  fi
  [ -f "$dir/recon_post.ply" ] || return 1
  if [ "$SKIP_EVAL" = "1" ]; then
    echo "SKIP_EVAL: missing recon_aligned under $dir ($scan)" >&2
    return 1
  fi
  if [ -f "$dir/cfg_args" ]; then
    echo "evaluate_dtu_mesh -> $dir ($scan)" >&2
    "$PYTHON" evaluate_dtu_mesh.py -m "$dir" --iteration -1 --DTU "$DTU" || return 1
  else
    src="${EXTERNAL_MESH_SOURCE_PATH:-}"
    if [ -z "$src" ] && [ -n "${EXTERNAL_MESH_SOURCE_ROOT:-}" ]; then
      src="${EXTERNAL_MESH_SOURCE_ROOT%/}/${scan}"
    fi
    if [ ! -d "$src/sparse/0" ]; then
      echo "external mesh: need COLMAP at $src for $dir ($scan)" >&2
      return 1
    fi
    echo "evaluate_external_mesh_dtu -> $dir (source_path=$src)" >&2
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
  local scan="$1"
  local label="$2"
  local mesh="$3"
  local out_dir="$4"
  local cb="$5"
  local num="${scan#scan}"
  if [ ! -f "$mesh" ]; then
    echo "SKIP $scan $label (no mesh)"
    return 0
  fi
  local out_ply="$out_dir/${label}_d2s_vertex_${CMAP}_vmax${VMAX}.ply"
  echo "---- $scan $label <- $mesh ----"
  if [ ! -f "$cb" ]; then
    "$PYTHON" scripts/export_mesh_vertex_error_dtu.py \
      --dtu_dir "$DTU" \
      --scan "$num" \
      --mesh "$mesh" \
      --out "$out_ply" \
      --vmax "$VMAX" \
      --colormap "$CMAP" \
      --colorbar_out "$cb"
  else
    "$PYTHON" scripts/export_mesh_vertex_error_dtu.py \
      --dtu_dir "$DTU" \
      --scan "$num" \
      --mesh "$mesh" \
      --out "$out_ply" \
      --vmax "$VMAX" \
      --colormap "$CMAP"
  fi
}

run_one_scan() {
  local scan="$1"
  local out_dir="${OUT_ROOT%/}/${scan}"
  local cb="$out_dir/colorbar_${CMAP}_vmax${VMAX}.png"
  local meta="$out_dir/run_manifest.txt"
  mkdir -p "$out_dir"
  : >"$meta"
  echo "scan=$scan dtu=$DTU vmax=$VMAX out=$out_dir" | tee -a "$meta"

  if [ "$VERTEX_ONLY_OUTPUT_DTU" = "1" ]; then
    if [ -z "${OUTPUT_DTU_SUFFIXES// /}" ]; then
      echo "WARN: VERTEX_ONLY_OUTPUT_DTU=1 but OUTPUT_DTU_SUFFIXES empty — set e.g. OUTPUT_DTU_SUFFIXES=edge_dn" | tee -a "$meta"
    fi
    local mesh_base dir_base suf
    dir_base="./output/dtu/${scan}"
    mesh_base="$(ensure_aligned "$scan" "$dir_base" || true)"
    if [ -n "${mesh_base:-}" ]; then
      export_one "$scan" "dtu_baseline" "$mesh_base" "$out_dir" "$cb" | tee -a "$meta"
    else
      echo "SKIP $scan dtu_baseline (no aligned mesh under $dir_base)" | tee -a "$meta"
    fi
    for suf in $OUTPUT_DTU_SUFFIXES; do
      local dir_var="./output/dtu/${scan}_${suf}"
      local mesh_v
      mesh_v="$(ensure_aligned "$scan" "$dir_var" || true)"
      if [ -n "${mesh_v:-}" ]; then
        export_one "$scan" "$suf" "$mesh_v" "$out_dir" "$cb" | tee -a "$meta"
      else
        echo "SKIP $scan $suf (no aligned mesh under $dir_var)" | tee -a "$meta"
      fi
    done
    echo "Done $scan -> $out_dir (colorbar: $cb)" | tee -a "$meta"
    return
  fi

  declare -a jobs=(
    "baseline:./output_compare/ggs_baseline/${scan}"
    "M1-3:./output_compare/m1_ablate_03_full/${scan}"
    "M1-B:./output_compare/m1_b_mesh_align/${scan}"
  )

  for entry in "${jobs[@]}"; do
    local label="${entry%%:*}"
    local dir="${entry#*:}"
    local mesh
    mesh="$(ensure_aligned "$scan" "$dir" || true)"
    if [ -n "${mesh:-}" ]; then
      export_one "$scan" "$label" "$mesh" "$out_dir" "$cb" | tee -a "$meta"
    else
      echo "SKIP $scan $label (no aligned mesh)" | tee -a "$meta"
    fi
  done

  if [ -n "${OUTPUT_DTU_SUFFIXES// /}" ]; then
    echo "--- extra: ./output/dtu/${scan} vs _suffix (${OUTPUT_DTU_SUFFIXES}) ---" | tee -a "$meta"
    local mesh_b dir_b suf2
    dir_b="./output/dtu/${scan}"
    mesh_b="$(ensure_aligned "$scan" "$dir_b" || true)"
    if [ -n "${mesh_b:-}" ]; then
      export_one "$scan" "dtu_baseline" "$mesh_b" "$out_dir" "$cb" | tee -a "$meta"
    else
      echo "SKIP $scan dtu_baseline ($dir_b)" | tee -a "$meta"
    fi
    for suf2 in $OUTPUT_DTU_SUFFIXES; do
      local d2="./output/dtu/${scan}_${suf2}"
      local m2
      m2="$(ensure_aligned "$scan" "$d2" || true)"
      if [ -n "${m2:-}" ]; then
        export_one "$scan" "$suf2" "$m2" "$out_dir" "$cb" | tee -a "$meta"
      else
        echo "SKIP $scan $suf2 ($d2)" | tee -a "$meta"
      fi
    done
  fi

  local sug="${SUGAR_MODEL_DIR:-./output_compare/sugar/${scan}}"
  # SuGaR: use ./output_compare/sugar/<scan>/ (recon_post + COLMAP via EXTERNAL_MESH_*).
  # Per-scan override: SUGAR_MODEL_DIR=/path/to/sugar_scan105 ./scripts/... SCANS=scan105
  if [ -d "$sug" ]; then
    local sm
    sm="$(ensure_aligned "$scan" "$sug" || true)"
    if [ -n "${sm:-}" ]; then
      export_one "$scan" "SuGaR" "$sm" "$out_dir" "$cb" | tee -a "$meta"
    else
      echo "SKIP $scan SuGaR" | tee -a "$meta"
    fi
  fi

  echo "Done $scan -> $out_dir (colorbar: $cb)" | tee -a "$meta"
}

for scan in $SCANS; do
  echo ""
  echo "########## $scan ##########"
  run_one_scan "$scan"
done

echo ""
echo "All scans finished. Root: $OUT_ROOT/<scan>/"
echo "Each folder: *_d2s_vertex_${CMAP}_vmax${VMAX}.ply + colorbar_${CMAP}_vmax${VMAX}.png"
