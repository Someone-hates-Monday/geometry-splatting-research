#!/usr/bin/env bash
# Broader leftover cleanup (data duplicates, temp extracts, old figures, vendored junk).
# Does NOT touch: dtu/DTU (train), dtu_eval/Offical_DTU_Dataset (eval GT),
#   output_compare/{ggs_ablate_no_dn,ggs_baseline,ggs_ex_trunc4,ggs_tau_*,metrics},
#   UGGS source / submodules (installed packages stay).
set -eo pipefail
REPO=/mnt/c/Users/Someone/Desktop/SuGaR-main
UGGS="$REPO/unified_geometry_gs"
DRY="${DRY_RUN:-0}"

rm_path() {
  local p="$1"
  local why="$2"
  if [ -e "$p" ] || [ -L "$p" ]; then
    local sz
    sz=$(du -sh "$p" 2>/dev/null | awk '{print $1}')
    echo "  DELETE ($sz) $p  # $why"
    if [ "$DRY" != "1" ]; then
      rm -rf "$p"
    fi
  fi
}

echo "[cleanup-repo] dry_run=$DRY"
echo "before:"; df -h /mnt/c | tail -1; du -sh "$REPO" "$UGGS" 2>/dev/null

echo "=== UGGS temp / duplicate data ==="
rm_path "$UGGS/dtu_eval/_extract" "SampleSet extract; Official_DTU_Dataset already has Points/Calib/ObsMask"
rm_path "$UGGS/dtu_eval/_download" "empty download staging"
rm_path "$UGGS/dtu_eval/_tmp_extract" "empty tmp"
rm_path "$UGGS/data" "duplicate of \$REPO/dtu/DTU (UGGS_DTU_ROOT)"
rm_path "$UGGS/dtu_download.log" "download log"

echo "=== old report figure dumps ==="
rm_path "$UGGS/report_figures" "old M1/edge figure dump"
rm_path "$UGGS/report_figures_edge_compare" "old compare figures"
rm_path "$UGGS/report_figures_scan65_edge_dn_vs_baseline" "old edge_dn figures"
rm_path "$UGGS/report_figures_vertex_colored" "old vertex color figures"
rm_path "$UGGS/report_figures_vertex_edge_dn_vs_baseline" "old edge_dn figures"
rm_path "$UGGS/report_figures_vertex_edge_dn_vs_baseline.zip" "zip of above"
rm_path "$UGGS/report_figures_vertex_err" "old error figures"
rm_path "$UGGS/presentations/uggs-m1-guizang.zip" "duplicate of presentations/uggs-m1-guizang/"

echo "=== broken extract symlinks / empty clouds ==="
# dangling point_cloud + cameras.json symlinks after Gaussian deletion
for scan in scan24 scan105 scan122; do
  rm_path "$UGGS/output_compare/ggs_ex_trunc4/$scan/point_cloud" "dangling symlink"
  if [ -L "$UGGS/output_compare/ggs_ex_trunc4/$scan/cameras.json" ]; then
    rm_path "$UGGS/output_compare/ggs_ex_trunc4/$scan/cameras.json" "symlink to deleted tree"
  fi
done
find "$UGGS/output_compare" -type d -name point_cloud -empty -delete 2>/dev/null || true

echo "=== caches / build trees (rebuilt on pip install) ==="
find "$UGGS" -type d -name __pycache__ -prune -print 2>/dev/null | while read -r d; do
  rm_path "$d" "pycache"
done
find "$UGGS/submodules" -type d -name build -prune -print 2>/dev/null | while read -r d; do
  rm_path "$d" "extension build dir"
done
find "$UGGS/submodules" -type d -name '*.egg-info' -prune -print 2>/dev/null | while read -r d; do
  rm_path "$d" "egg-info"
done

echo "=== repo-root leftovers (not UGGS / not SuGaR source) ==="
rm_path "$REPO/pytorch3d-main" "vendored pytorch3d source tree"
if [ -d "$REPO/gaussian_splatting" ]; then
  chmod -R u+w "$REPO/gaussian_splatting" 2>/dev/null || true
fi
rm_path "$REPO/gaussian_splatting" "vendored 3DGS copy; UGGS has its own submodules"
rm_path "$REPO/dtu_results" "old packed DTU result dump"
rm_path "$REPO/media" "SuGaR demo media assets"
rm_path "$REPO/eval_dtu-20260311T144651Z-1-001" "old eval zip extract"
rm_path "$REPO/network_test_log.txt" "network test log"
rm_path "$REPO/run_network_test.cmd" "network test helper"
rm_path "$REPO/output" "legacy output if any"

# Optional heavy viewer — keep SIBR in UGGS (used by some viewers); skip unless huge

echo "after:"
df -h /mnt/c | tail -1
du -sh "$REPO" "$UGGS" "$UGGS/output_compare" "$UGGS/dtu_eval" 2>/dev/null
echo "UGGS top:"
du -sh "$UGGS"/* 2>/dev/null | sort -hr | head -20
echo "repo top:"
du -sh "$REPO"/* 2>/dev/null | sort -hr | head -15
echo "Done."
