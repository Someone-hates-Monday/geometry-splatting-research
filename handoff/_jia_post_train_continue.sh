#!/usr/bin/env bash
set -eo pipefail
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs
source configs/dtu_path.env
BASE=output_compare/ggs_trust_rebuild_20261116
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"

echo "=== model check ==="
for s in scan24 scan105 scan122; do
  ply="$BASE/$s/model/point_cloud/iteration_30000/point_cloud.ply"
  cfg="$BASE/$s/model/cfg_args"
  if [ ! -f "$ply" ]; then echo "MISSING $ply"; exit 1; fi
  echo "$s ply=$(stat -c%s "$ply") cfg=$([ -f "$cfg" ] && echo ok || echo miss)"
  if [ -f "$BASE/$s/logs/train_exit.txt" ]; then
    echo "  exit: $(tr '\n' ' ' < "$BASE/$s/logs/train_exit.txt")"
  fi
done

# Load integrity all three
python - <<'PY'
from pathlib import Path
from scene.gaussian_model import GaussianModel
base = Path("output_compare/ggs_trust_rebuild_20261116")
for s in ("scan24", "scan105", "scan122"):
    ply = base / s / "model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    g = GaussianModel(3, 0)
    g.load_ply(str(ply))
    n = int(g.get_xyz.shape[0])
    print(f"load_ok {s} n={n}")
    assert n > 1000
    (base / s / "logs").mkdir(parents=True, exist_ok=True)
    (base / s / "logs" / "load_ok.txt").write_text(f"n_gaussians={n}\n", encoding="utf-8")
PY

# Control extract scan24 (no trust mask)
CTRL24="$BASE/scan24/control_trunc4"
mkdir -p "$CTRL24" "$BASE/scan24/logs"
if [ -f "$CTRL24/recon_post.ply" ] && [ -f "$CTRL24/vis/results.json" ]; then
  echo "SKIP control extract scan24 (exists)"
else
  echo "=== control extract scan24 ==="
  python mesh_extract.py \
    -m "$BASE/scan24/model" \
    --iteration 30000 \
    --trunc_voxel_multiplier 4 \
    --voxel_size 0.002 \
    --depth_max 8.0 \
    --num_cluster 1 \
    --write_dir "$CTRL24" \
    2>&1 | tee "$BASE/scan24/logs/control_extract.log"
  echo "=== control eval scan24 ==="
  python evaluate_dtu_mesh.py \
    -m "$BASE/scan24/model" \
    --iteration 30000 \
    --DTU "$DTU_EVAL_ROOT" \
    --mesh_path "$CTRL24/recon_post.ply" \
    --out_dir "$CTRL24" \
    2>&1 | tee "$BASE/scan24/logs/control_eval.log"
fi
test -f "$CTRL24/vis/results.json"
echo "control results:"; cat "$CTRL24/vis/results.json"

# Smoke
SMOKE="$BASE/scan24/smoke"
mkdir -p "$SMOKE"
echo "=== smoke scan24 ==="
python scripts/smoke_jia_trust_fuse_scan24.py \
  -m "$BASE/scan24/model" \
  -s "$UGGS_DTU_ROOT/scan24" \
  -r 2 \
  --iteration 30000 \
  --out_dir "$SMOKE" \
  2>&1 | tee "$BASE/scan24/logs/smoke.log"

echo "=== smoke receipt ==="
cat "$SMOKE/smoke_receipt.json"
echo "DONE_CONTINUE_PHASE"
