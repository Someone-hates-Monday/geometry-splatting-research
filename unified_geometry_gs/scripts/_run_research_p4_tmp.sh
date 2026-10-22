#!/usr/bin/env bash
set -eo pipefail
source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
source configs/dtu_path.env

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export PYTHON=python

# Ensure only one train job
if pgrep -f "train.py .*ggs_tau" >/dev/null 2>&1; then
  echo "ERROR: train.py already running"
  pgrep -af "train.py" || true
  exit 1
fi

chmod +x scripts/run_research_p4.sh scripts/judge_g1.py
echo "=== RESEARCH_PROTOCOL P4 scan105 ==="
./scripts/run_research_p4.sh
echo "=== research p4 finished ==="
