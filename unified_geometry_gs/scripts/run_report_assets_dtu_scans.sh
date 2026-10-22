#!/usr/bin/env bash
# One-shot report assets for DTU scans: DTU mesh eval → orbit9 mesh screenshots → vertex-error triple PLY.
#
# Default scans: scan24 (best M1-3 overall + M1-B wins) and scan105 (strong M1-3 d2s; no scan beats 24 on overall).
#
# Usage (from unified_geometry_gs, WSL):
#   source configs/dtu_path.env   # optional
#   export PYTHON=python
#   chmod +x scripts/run_report_assets_dtu_scans.sh
#   ./scripts/run_report_assets_dtu_scans.sh
#
# Env:
#   SCANS="scan24 scan105"   # override
#   DTU=/path/to/Offical_DTU_Dataset
#   SKIP_EVAL=1              # skip evaluate_dtu_mesh if recon_aligned.ply already exists
#   SKIP_SNAP=1
#   SKIP_VERTEX=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DTU="${DTU:-./dtu_eval/Offical_DTU_Dataset}"
SCANS="${SCANS:-scan24 scan105}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_SNAP="${SKIP_SNAP:-0}"
SKIP_VERTEX="${SKIP_VERTEX:-0}"
VMAX_VERTEX="${VMAX_VERTEX:-6}"
COLORMAP_VERTEX="${COLORMAP_VERTEX:-turbo}"

scan_to_num() {
  local s="$1"
  echo "${s#scan}"
}

run_eval_for_scan() {
  local scan="$1"
  local num method mp
  num="$(scan_to_num "$scan")"
  for method in ggs_baseline m1_ablate_03_full m1_b_mesh_align; do
    mp="./output_compare/$method/$scan"
    if [ ! -f "$mp/recon_post.ply" ]; then
      echo "WARN: skip eval — missing $mp/recon_post.ply"
      continue
    fi
    echo "----- evaluate_dtu_mesh $method $scan -----"
    "$PYTHON" evaluate_dtu_mesh.py -m "$mp" --iteration -1 --DTU "$DTU"
  done
}

run_snapshots_orbit9() {
  local scan="$1"
  local out_snap="./report_figures/${scan}_orbit9"
  mkdir -p "$out_snap"
  echo "----- capture_mesh_snapshots orbit9 -> $out_snap -----"
  "$PYTHON" scripts/capture_mesh_snapshots.py \
    --backend matplotlib \
    --mat_preset orbit9 \
    --out_dir "$out_snap" \
    --width 1600 \
    --height 1200 \
    "baseline:./output_compare/ggs_baseline/$scan/recon_post.ply" \
    "M1-3:./output_compare/m1_ablate_03_full/$scan/recon_post.ply" \
    "M1-B:./output_compare/m1_b_mesh_align/$scan/recon_post.ply"
}

run_vertex_triplet() {
  local scan="$1"
  local num out_err ba m13 mb
  num="$(scan_to_num "$scan")"
  out_err="./report_figures/${scan}_err_vertex_triple"
  ba="./output_compare/ggs_baseline/$scan/recon_aligned.ply"
  m13="./output_compare/m1_ablate_03_full/$scan/recon_aligned.ply"
  mb="./output_compare/m1_b_mesh_align/$scan/recon_aligned.ply"
  if [ ! -f "$ba" ] || [ ! -f "$m13" ] || [ ! -f "$mb" ]; then
    echo "WARN: skip vertex triplet — need recon_aligned.ply for all three under $scan"
    return
  fi
  mkdir -p "$out_err"
  echo "----- vertex error triplet -> $out_err -----"
  "$PYTHON" scripts/export_mesh_vertex_error_dtu.py \
    --dtu_dir "$DTU" \
    --scan "$num" \
    --vmax "$VMAX_VERTEX" \
    --colormap "$COLORMAP_VERTEX" \
    --mesh_a "$ba" \
    --mesh_b "$m13" \
    --mesh_c "$mb" \
    --out_dir "$out_err"
}

for scan in $SCANS; do
  echo ""
  echo "########## $scan ##########"
  if [ "$SKIP_EVAL" = "1" ]; then
    echo "SKIP_EVAL=1 — skipping evaluate_dtu_mesh"
  else
    run_eval_for_scan "$scan"
  fi
  if [ "$SKIP_SNAP" != "1" ]; then
    run_snapshots_orbit9 "$scan"
  fi
  if [ "$SKIP_VERTEX" != "1" ]; then
    run_vertex_triplet "$scan"
  fi
done

echo ""
echo "Done. See ./report_figures/<scan>_orbit9/ and ./report_figures/<scan>_err_vertex_triple/"
