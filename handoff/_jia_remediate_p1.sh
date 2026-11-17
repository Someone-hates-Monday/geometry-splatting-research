#!/usr/bin/env bash
# P1 remediation only: compat grad, smoke (new dir), dual seeded eval, overwrite self-check.
# Does NOT extract trust meshes. Does NOT overwrite prior review artifacts.
set -eo pipefail
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs
source configs/dtu_path.env
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"

BASE=output_compare/ggs_trust_rebuild_20261116
SCAN24="$BASE/scan24"
MODEL="$SCAN24/model"
CTRL_MESH="$SCAN24/control_trunc4/recon_post.ply"
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$SCAN24/logs"
mkdir -p "$LOGDIR"

echo "===== 1) overwrite self-check (must fail) ====="
set +e
python evaluate_dtu_mesh.py \
  -m "$MODEL" -s "$UGGS_DTU_ROOT/scan24" -r 2 \
  --mesh_path "$CTRL_MESH" \
  --out_dir "$SCAN24/control_trunc4" \
  --eval_seed 0 \
  --DTU "$DTU_EVAL_ROOT" \
  >"$LOGDIR/overwrite_selfcheck_${STAMP}.log" 2>&1
rc=$?
set -e
echo "overwrite_selfcheck_exit=$rc"
if [[ "$rc" -eq 0 ]]; then
  echo "ERROR: overwrite into control_trunc4 should have failed" >&2
  exit 1
fi
echo "overwrite_selfcheck_ok" | tee "$LOGDIR/overwrite_selfcheck_${STAMP}.ok"

echo "===== 2) compat grad (HEAD vs current) ====="
COMPAT_DIR="$SCAN24/compat_grad_${STAMP}"
python scripts/compat_jia_patchmatch_grad.py \
  -m "$MODEL" -s "$UGGS_DTU_ROOT/scan24" -r 2 \
  --iteration 30000 \
  --out_dir "$COMPAT_DIR" \
  --repo_git /mnt/c/Users/Someone/Desktop/SuGaR-main \
  2>&1 | tee "$LOGDIR/compat_grad_${STAMP}.log"

echo "===== 3) smoke mask (new dir) ====="
SMOKE_DIR="$SCAN24/smoke_${STAMP}"
python scripts/smoke_jia_trust_fuse_scan24.py \
  -m "$MODEL" -s "$UGGS_DTU_ROOT/scan24" -r 2 \
  --iteration 30000 \
  --out_dir "$SMOKE_DIR" \
  2>&1 | tee "$LOGDIR/smoke_${STAMP}.log"

echo "===== 4) dual seeded eval of same control mesh ====="
for run in 1 2; do
  OUT="$SCAN24/control_trunc4_eval_seed0_run${run}_${STAMP}"
  python evaluate_dtu_mesh.py \
    -m "$MODEL" -s "$UGGS_DTU_ROOT/scan24" -r 2 \
    --mesh_path "$CTRL_MESH" \
    --out_dir "$OUT" \
    --eval_seed 0 \
    --forbid_root "$SCAN24/control_trunc4" \
    --DTU "$DTU_EVAL_ROOT" \
    2>&1 | tee "$LOGDIR/eval_seed0_run${run}_${STAMP}.log"
done

echo "===== 5) compare seeded results ====="
python3 - <<PY
import json, pathlib
base=pathlib.Path("$SCAN24")
stamp="$STAMP"
paths=[base/f"control_trunc4_eval_seed0_run{i}_{stamp}/vis/results.json" for i in (1,2)]
rs=[json.loads(p.read_text()) for p in paths]
print("run1", rs[0])
print("run2", rs[1])
same=rs[0]==rs[1]
print("bit_equal_results_json", same)
old=json.loads((base/"control_trunc4/vis/results.json").read_text())
print("legacy_unseeded_control", old)
print("legacy_kept_untouched", True)
receipt={
  "stamp": stamp,
  "seeded_run1": rs[0],
  "seeded_run2": rs[1],
  "seeded_bit_equal": same,
  "legacy_unseeded": old,
  "compat_dir": str(base/f"compat_grad_{stamp}"),
  "smoke_dir": str(base/f"smoke_{stamp}"),
}
out=base/f"logs/p1_remediation_summary_{stamp}.json"
out.write_text(json.dumps(receipt, indent=2))
print("summary", out)
if not same:
  raise SystemExit(3)
PY

echo "===== P1 remediation finished stamp=$STAMP ====="
