#!/usr/bin/env bash
set -e
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

chmod +x scripts/run_ggs_ablation.sh scripts/request_train_stop.sh

echo "=== G-GS leave-one-out: scan24 scan105 ==="
./scripts/run_ggs_ablation.sh scan24 scan105
echo "=== ggs ablation finished ==="
