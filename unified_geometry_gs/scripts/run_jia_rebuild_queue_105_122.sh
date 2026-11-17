#!/usr/bin/env bash
# After scan24 rebuild finishes: verify, then train scan105 and scan122 (same frozen recipe).
# Does not extract trust meshes or start TGS. One GPU — sequential only.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$UGGS_ROOT"

BASE=./output_compare/ggs_trust_rebuild_20261116
LOG="$BASE/queue_105_122.log"
mkdir -p "$BASE"
exec > >(tee -a "$LOG") 2>&1

echo "=== queue start $(date -u) ==="
PLY24="$BASE/scan24/model/point_cloud/iteration_30000/point_cloud.ply"
EXIT24="$BASE/scan24/logs/train_exit.txt"

echo "Waiting for scan24 model: $PLY24"
while [ ! -f "$PLY24" ]; do
  sleep 60
done
echo "ply present $(date -u)"

# Prefer waiting for recorded exit if the train script wrote it; otherwise verify ply size
for i in $(seq 1 120); do
  if [ -f "$EXIT24" ]; then
    echo "train_exit.txt:"; cat "$EXIT24"
    break
  fi
  # If train process gone and ply exists, continue
  if ! pgrep -af "train.py.*ggs_trust_rebuild_20261116/scan24" >/dev/null 2>&1; then
    echo "no train.py for scan24; proceeding with ply check"
    break
  fi
  sleep 30
done

sz=$(stat -c%s "$PLY24")
echo "scan24 ply bytes=$sz"
if [ "$sz" -lt 1000000 ]; then
  echo "ERROR: scan24 ply suspiciously small"
  exit 1
fi

# Integrity: Scene load smoke (iteration 30000)
source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs
source configs/dtu_path.env
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
ap = ArgumentParser()
model = ModelParams(ap, sentinel=True)
PipelineParams(ap)
args = ap.parse_args([
    "-m", "output_compare/ggs_trust_rebuild_20261116/scan24/model",
    "-s", __import__("os").environ["UGGS_DTU_ROOT"] + "/scan24",
    "-r", "2",
])
safe_state(True)
ds = model.extract(args)
g = GaussianModel(ds.sh_degree, ds.sg_degree)
scene = Scene(ds, g, load_iteration=30000, shuffle=False)
n = g.get_xyz.shape[0]
print(f"load_ok n_gaussians={n} cams={len(scene.getTrainCameras())}")
assert n > 1000
Path = __import__("pathlib").Path
Path("output_compare/ggs_trust_rebuild_20261116/scan24/logs/load_ok.txt").write_text(
    f"n_gaussians={n}\n", encoding="utf-8"
)
PY

for SCAN in scan105 scan122; do
  echo "===== train $SCAN $(date -u) ====="
  bash "$SCRIPT_DIR/run_jia_rebuild_train_one.sh" "$SCAN"
  echo "===== done $SCAN exit=$? $(date -u) ====="
done

echo "=== queue complete $(date -u) ==="
echo "Next human/agent step: control extract + smoke on scan24 (review stop); then pair eval."
