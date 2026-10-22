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

if pgrep -f "python train.py" >/dev/null 2>&1; then
  echo "ERROR: train.py already running"
  pgrep -af "python train.py" || true
  exit 1
fi

# Incomplete interrupted run: wipe so train starts clean
for s in scan110 scan114 scan118; do
  p="./output_compare/ggs_pipeline/$s"
  if [ -d "$p" ] && [ ! -f "$p/vis/results.json" ]; then
    echo "REMOVE incomplete $p"
    rm -rf "$p"
  fi
done

echo "=== already done ==="
ls ./output_compare/ggs_pipeline/*/vis/results.json 2>/dev/null | sed 's|.*/ggs_pipeline/||;s|/vis/results.json||' | sort || true

chmod +x scripts/run_research_p5.sh scripts/judge_p5.py
echo "=== resume P5 ==="
./scripts/run_research_p5.sh
echo "=== research p5 finished ==="
