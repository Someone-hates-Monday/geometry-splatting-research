#!/usr/bin/env bash
# Free disk: drop obsolete runs; slim kept Pipeline* dirs (drop chkpnts/debug).
# KEEP: metrics/, ggs_baseline, ggs_ablate_no_dn, ggs_ex_trunc4, ggs_tau_*
# Metrics JSON already record failed ablations — full models not needed.
set -eo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)/output_compare"
DRY="${DRY_RUN:-0}"

keep_top=(
  metrics
  ggs_baseline
  ggs_ablate_no_dn
  ggs_ex_trunc4
  ggs_tau_0.3
  ggs_tau_0.5
  ggs_tau_1.2
)

echo "[cleanup] root=$ROOT dry_run=$DRY"
echo "[cleanup] before:"
du -sh "$ROOT" 2>/dev/null || true

# --- 1) Delete entire obsolete experiment trees ---
discard=(
  ggs_ncc_fail ggs_ncc_gate_dn ggs_ncc_corr ggs_mv2 ggs_ncc_core
  ggs_ablate_ncc_only ggs_ablate_no_mv ggs_ablate_geo_only
  ggs_ex_v001 ggs_ex_v004 ggs_ex_dmax6 ggs_ex_dmax10 ggs_ex_trunc12
  ggs_ex_m2 ggs_ex_m6 ggs_ex_c2
  m1_field_occ m1_vd06_main m1_vd04_val6 m1_vd09 m1_b_mesh_align
  m1_ablate_01_occ_only m1_ablate_02_depth_occ_only m1_ablate_03_full
  m1_ablate_04_full_rank m1_vd06_multi m1_meshfocus m1_meshmid
  m1_router m1_vd08 m1_vd07 m1_vd06 m1_unified
  m2_wgeo_v2_safe m2_wgeo_safe m2_wgeo
)

for name in "${discard[@]}"; do
  p="$ROOT/$name"
  if [ -e "$p" ] || [ -L "$p" ]; then
    sz=$(du -sh "$p" 2>/dev/null | awk '{print $1}')
    echo "  DELETE $name ($sz)"
    if [ "$DRY" != "1" ]; then
      rm -rf "$p"
    fi
  fi
done

# Also drop any other m1_*/m2_* not listed
shopt -s nullglob
for p in "$ROOT"/m1_* "$ROOT"/m2_*; do
  name=$(basename "$p")
  echo "  DELETE leftover $name ($(du -sh "$p" | awk '{print $1}'))"
  if [ "$DRY" != "1" ]; then
    rm -rf "$p"
  fi
done
shopt -u nullglob

# --- 2) Slim kept model dirs: remove dense checkpoints + debug + old iterations ---
slim_model_dir() {
  local d="$1"
  [ -d "$d" ] || return 0
  # checkpoints (dominant size)
  if compgen -G "$d/chkpnt*.pth" > /dev/null; then
    local n
    n=$(ls "$d"/chkpnt*.pth 2>/dev/null | wc -l)
    echo "    strip $n chkpnt*.pth in $(basename "$(dirname "$d")")/$(basename "$d")"
    if [ "$DRY" != "1" ]; then
      rm -f "$d"/chkpnt*.pth
    fi
  fi
  if [ -d "$d/debug" ]; then
    echo "    strip debug/ in $(basename "$(dirname "$d")")/$(basename "$d")"
    if [ "$DRY" != "1" ]; then
      rm -rf "$d/debug"
    fi
  fi
  # Keep only latest point_cloud iteration (sort by numeric suffix of basename).
  if [ -d "$d/point_cloud" ] && [ ! -L "$d/point_cloud" ]; then
    local latest=""
    local max_iter=-1
    local it base num
    for it in "$d/point_cloud"/iteration_*; do
      [ -e "$it" ] || continue
      base=$(basename "$it")
      num=${base#iteration_}
      if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -gt "$max_iter" ]; then
        max_iter=$num
        latest=$it
      fi
    done
    if [ -n "$latest" ]; then
      for it in "$d/point_cloud"/iteration_*; do
        [ -e "$it" ] || continue
        [ "$it" = "$latest" ] && continue
        echo "    strip older $(basename "$it") (keep $(basename "$latest"))"
        if [ "$DRY" != "1" ]; then
          rm -rf "$it"
        fi
      done
    fi
  fi
}

for name in ggs_baseline ggs_ablate_no_dn ggs_tau_0.3 ggs_tau_0.5 ggs_tau_1.2; do
  echo "  SLIM $name"
  for scan in "$ROOT/$name"/*; do
    [ -d "$scan" ] || continue
    slim_model_dir "$scan"
  done
done

# extract dirs: drop broken symlinks / unused; keep meshes+vis
for name in ggs_ex_trunc4; do
  echo "  SLIM extract $name (meshes only essentials)"
  for scan in "$ROOT/$name"/*; do
    [ -d "$scan" ] || continue
    # remove chkpnts if any copied
    slim_model_dir "$scan"
    # point_cloud symlink to no_dn — keep; if dangling after no_dn slim, ok if still points
  done
done

# --- 3) Optional: old unified_geometry_gs/output (legacy) ---
if [ -d "./output" ]; then
  sz=$(du -sh ./output | awk '{print $1}')
  echo "  DELETE legacy ./output ($sz)"
  if [ "$DRY" != "1" ]; then
    rm -rf ./output
  fi
fi

echo "[cleanup] after:"
du -sh "$ROOT" 2>/dev/null || true
df -h /mnt/c | tail -1
echo "[cleanup] kept top-level:"
ls -1 "$ROOT"
echo "Done."
