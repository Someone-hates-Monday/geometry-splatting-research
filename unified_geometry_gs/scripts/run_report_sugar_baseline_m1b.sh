#!/usr/bin/env bash
# DTU report: SuGaR vs GGS baseline vs M1-B — mesh eval, orbit9 snapshots, vertex-error triplet + metrics JSON.
#
# Prerequisites:
#   - conda env uggs, DTU SampleSet + Points under DTU root
#   - baseline & M1-B: ./output_compare/ggs_baseline/<SCAN>, ./output_compare/m1_b_mesh_align/<SCAN> with recon_post.ply
#   - SuGaR: ./output_compare/sugar/<SCAN>/recon_post.ply (or set SUGAR_MODEL_DIR)
#   - COLMAP scan path for SuGaR alignment: SOURCE_PATH or UGGS_DTU_ROOT (see configs/dtu_path.env)
#
# Usage (from unified_geometry_gs — repo folder that contains train.py / scripts/):
#   cd /path/to/SuGaR-main/unified_geometry_gs
#   source configs/dtu_path.env   # defines UGGS_DTU_ROOT
#   export PYTHON=python
#   chmod +x scripts/run_report_sugar_baseline_m1b.sh
#   ./scripts/run_report_sugar_baseline_m1b.sh
#
# If SuGaR recon_post.ply is missing for a scan, orbit + vertex export still run for baseline + M1-B only.
#
# Env:
#   SCAN=scan24
#   SCANS="scan24 scan105"
#   SUGAR_MODEL_DIR=./output_compare/sugar/scan24
#   SOURCE_PATH=/abs/path/to/scan24   # overrides UGGS_DTU_ROOT/SCAN
#   DTU=./dtu_eval/Offical_DTU_Dataset
#   SKIP_EVAL=1  SKIP_SNAP=1  SKIP_VERTEX=1  SKIP_METRICS=1
#   FORCE_MESH_EVAL=1  — redo DTU mesh eval even if vis/results.json exists
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DTU="${DTU:-./dtu_eval/Offical_DTU_Dataset}"
SCANS="${SCANS:-${SCAN:-scan24}}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_SNAP="${SKIP_SNAP:-0}"
SKIP_VERTEX="${SKIP_VERTEX:-0}"
SKIP_METRICS="${SKIP_METRICS:-0}"
VMAX_VERTEX="${VMAX_VERTEX:-6}"
COLORMAP_VERTEX="${COLORMAP_VERTEX:-turbo}"

if [ -f configs/dtu_path.env ]; then
  # shellcheck source=/dev/null
  source configs/dtu_path.env
fi

scan_to_num() {
  local s="$1"
  echo "${s#scan}"
}

resolve_source_path() {
  local scan="$1"
  if [ -n "${SOURCE_PATH:-}" ] && [ -d "${SOURCE_PATH}/sparse/0" ]; then
    echo "$SOURCE_PATH"
    return 0
  fi
  if [ -n "${UGGS_DTU_ROOT:-}" ]; then
    local p="${UGGS_DTU_ROOT%/}/${scan}"
    if [ -d "$p/sparse/0" ]; then
      echo "$p"
      return 0
    fi
  fi
  echo ""
}

ensure_sugar_eval() {
  local scan="$1"
  local num src sug
  num="$(scan_to_num "$scan")"
  sug="${SUGAR_MODEL_DIR:-./output_compare/sugar/${scan}}"
  src="$(resolve_source_path "$scan")"
  if [ ! -f "$sug/recon_post.ply" ]; then
    echo "WARN: skip SuGaR — missing $sug/recon_post.ply"
    return 1
  fi
  if [ -f "$sug/vis/results.json" ] && [ "${FORCE_MESH_EVAL:-0}" != "1" ]; then
    echo "SuGaR: skip eval (existing $sug/vis/results.json); FORCE_MESH_EVAL=1 to redo"
    return 0
  fi
  if [ -z "$src" ]; then
    echo "WARN: SuGaR needs COLMAP path — set SOURCE_PATH or UGGS_DTU_ROOT for $scan"
    return 1
  fi
  echo "----- evaluate_external_mesh_dtu SuGaR $scan -----"
  "$PYTHON" evaluate_external_mesh_dtu.py \
    --source_path "$src" \
    --mesh_in "$sug/recon_post.ply" \
    --out_dir "$sug" \
    --scan_id "$num" \
    --DTU "$DTU" \
    --write_recon_post
}

ensure_uggs_eval() {
  local dir="$1"
  local label="$2"
  if [ ! -f "$dir/recon_post.ply" ]; then
    echo "WARN: skip $label — missing $dir/recon_post.ply"
    return 1
  fi
  if [ -f "$dir/vis/results.json" ] && [ "${FORCE_MESH_EVAL:-0}" != "1" ]; then
    echo "$label: skip eval (existing vis/results.json); FORCE_MESH_EVAL=1 to redo"
    return 0
  fi
  echo "----- evaluate_dtu_mesh $label -----"
  "$PYTHON" evaluate_dtu_mesh.py -m "$dir" --iteration -1 --DTU "$DTU"
}

run_snapshots_orbit9() {
  local scan="$1"
  local out_snap sug ba mb
  local -a snap_entries=()
  out_snap="./report_figures/${scan}_orbit9_sugar_baseline_m1b"
  sug="${SUGAR_MODEL_DIR:-./output_compare/sugar/${scan}}"
  ba="./output_compare/ggs_baseline/${scan}"
  mb="./output_compare/m1_b_mesh_align/${scan}"
  mkdir -p "$out_snap"
  if [ -f "${sug}/recon_post.ply" ]; then
    snap_entries+=("SuGaR:${sug}/recon_post.ply")
  else
    echo "INFO: orbit9 — SuGaR skipped (missing ${sug}/recon_post.ply); rendering baseline + M1-B only."
  fi
  snap_entries+=("baseline:${ba}/recon_post.ply" "M1-B:${mb}/recon_post.ply")
  echo "----- capture_mesh_snapshots orbit9 -> $out_snap -----"
  "$PYTHON" scripts/capture_mesh_snapshots.py \
    --backend matplotlib \
    --mat_preset orbit9 \
    --out_dir "$out_snap" \
    --width 1600 \
    --height 1200 \
    "${snap_entries[@]}"
}

run_vertex_triplet() {
  local scan="$1"
  local num out_err sug ba mb sg
  num="$(scan_to_num "$scan")"
  out_err="./report_figures/${scan}_err_vertex_sugar_baseline_m1b"
  sug="${SUGAR_MODEL_DIR:-./output_compare/sugar/${scan}}"
  ba="./output_compare/ggs_baseline/${scan}/recon_aligned.ply"
  mb="./output_compare/m1_b_mesh_align/${scan}/recon_aligned.ply"
  sg="${sug}/recon_aligned.ply"
  if [ -f "$sg" ] && [ -f "$ba" ] && [ -f "$mb" ]; then
    mkdir -p "$out_err"
    echo "----- vertex error triplet (SuGaR, baseline, M1-B) -> $out_err -----"
    "$PYTHON" scripts/export_mesh_vertex_error_dtu.py \
      --dtu_dir "$DTU" \
      --scan "$num" \
      --vmax "$VMAX_VERTEX" \
      --colormap "$COLORMAP_VERTEX" \
      --mesh_a "$sg" \
      --mesh_b "$ba" \
      --mesh_c "$mb" \
      --label_a "SuGaR" \
      --label_b "baseline" \
      --label_c "M1-B" \
      --out_dir "$out_err"
    return
  fi
  if [ -f "$ba" ] && [ -f "$mb" ]; then
    mkdir -p "$out_err"
    echo "----- vertex error pair (baseline vs M1-B; no SuGaR aligned mesh) -> $out_err -----"
    "$PYTHON" scripts/export_mesh_vertex_error_dtu.py \
      --dtu_dir "$DTU" \
      --scan "$num" \
      --vmax "$VMAX_VERTEX" \
      --colormap "$COLORMAP_VERTEX" \
      --mesh_a "$ba" \
      --mesh_b "$mb" \
      --out_dir "$out_err"
    return
  fi
  echo "WARN: skip vertex export — need recon_aligned.ply for baseline and M1-B ($scan)"
}

run_metrics_json() {
  local scan="$1"
  local num out sug ba mb
  num="$(scan_to_num "$scan")"
  out="./report_figures/${scan}_metrics_sugar_baseline_m1b.json"
  sug="${SUGAR_MODEL_DIR:-./output_compare/sugar/${scan}}"
  ba="./output_compare/ggs_baseline/${scan}"
  mb="./output_compare/m1_b_mesh_align/${scan}"
  "$PYTHON" - << PY
import json
import pathlib

scan = "${scan}"
num = ${num}
out = pathlib.Path("${out}")
paths = {
    "SuGaR": pathlib.Path("${sug}") / "vis" / "results.json",
    "baseline": pathlib.Path("${ba}") / "vis" / "results.json",
    "M1-B": pathlib.Path("${mb}") / "vis" / "results.json",
}
mesh_metrics = {}
missing = []
for name, p in paths.items():
    if p.is_file():
        mesh_metrics[name] = json.loads(p.read_text(encoding="utf-8"))
    else:
        missing.append(str(p))
report = {
    "scan": scan,
    "scan_id": num,
    "mesh_metrics_dtuv2": mesh_metrics,
    "missing_results_json": missing,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(out.read_text(encoding="utf-8"))
PY
}

for scan in $SCANS; do
  echo ""
  echo "########## $scan ##########"
  if [ "$SKIP_EVAL" != "1" ]; then
    ensure_sugar_eval "$scan" || true
    ensure_uggs_eval "./output_compare/ggs_baseline/${scan}" "ggs_baseline" || true
    ensure_uggs_eval "./output_compare/m1_b_mesh_align/${scan}" "m1_b_mesh_align" || true
  else
    echo "SKIP_EVAL=1 — not running mesh eval scripts"
  fi
  if [ "$SKIP_SNAP" != "1" ]; then
    run_snapshots_orbit9 "$scan"
  fi
  if [ "$SKIP_VERTEX" != "1" ]; then
    run_vertex_triplet "$scan"
  fi
  if [ "$SKIP_METRICS" != "1" ]; then
    run_metrics_json "$scan"
  fi
done

echo ""
echo "Done."
echo "  Orbit PNGs:    ./report_figures/<scan>_orbit9_sugar_baseline_m1b/"
echo "  Vertex PLYs:   ./report_figures/<scan>_err_vertex_sugar_baseline_m1b/"
echo "  Metrics JSON: ./report_figures/<scan>_metrics_sugar_baseline_m1b.json"
echo ""
echo "Pairwise mesh compare (example):"
echo "  $PYTHON scripts/compare_models.py --mesh_only --scan_id N \\"
echo "    --baseline ./output_compare/ggs_baseline/scanNN \\"
echo "    --optimized ./output_compare/sugar/scanNN \\"
echo "    --external_source_path \"\$UGGS_DTU_ROOT/scanNN\" --dtu_eval_root \"$DTU\" \\"
echo "    --out ./output_compare/metrics/scanNN_baseline_vs_sugar.json"
