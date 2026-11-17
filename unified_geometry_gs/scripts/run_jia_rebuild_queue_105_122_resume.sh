#!/usr/bin/env bash
# Resume 105/122 after scan24 succeeded; skip wait if ply already present.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$UGGS_ROOT"
BASE=./output_compare/ggs_trust_rebuild_20261116
LOG="$BASE/queue_105_122.log"
mkdir -p "$BASE"
exec > >(tee -a "$LOG") 2>&1

echo "=== queue resume $(date -u) ==="
PLY24="$BASE/scan24/model/point_cloud/iteration_30000/point_cloud.ply"
[ -f "$PLY24" ] || { echo "ERROR missing scan24 ply"; exit 1; }

source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs
source configs/dtu_path.env
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
# Integrity: load ply via GaussianModel without rebuilding multi_view (cfg merge edge cases).
from scene.gaussian_model import GaussianModel
import torch
ply = Path("output_compare/ggs_trust_rebuild_20261116/scan24/model/point_cloud/iteration_30000/point_cloud.ply")
assert ply.is_file(), ply
g = GaussianModel(3, 0)
g.load_ply(str(ply))
n = g.get_xyz.shape[0]
print(f"load_ok n_gaussians={n} device_xyz={g.get_xyz.device}")
assert n > 1000
Path("output_compare/ggs_trust_rebuild_20261116/scan24/logs/load_ok.txt").write_text(
    f"n_gaussians={n}\nply={ply}\n", encoding="utf-8"
)
PY

for SCAN in scan105 scan122; do
  ply="$BASE/$SCAN/model/point_cloud/iteration_30000/point_cloud.ply"
  if [ -f "$ply" ]; then
    echo "SKIP $SCAN already has final ply"
    continue
  fi
  echo "===== train $SCAN $(date -u) ====="
  bash "$SCRIPT_DIR/run_jia_rebuild_train_one.sh" "$SCAN"
  echo "===== done $SCAN $(date -u) ====="
done
echo "=== queue complete $(date -u) ==="
