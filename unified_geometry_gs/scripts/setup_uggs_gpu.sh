#!/usr/bin/env bash
# 检查当前 conda 环境、CUDA、安装 pip 依赖与 submodules。
# 用法: conda activate uggs 后执行 ./scripts/setup_uggs_gpu.sh [CUDA 路径]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$UGGS_ROOT"

echo "=== 1. Conda 环境 ==="
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
  echo "未检测到 conda 环境。请先: conda activate uggs  或  conda activate sugar"
  exit 1
fi
echo "当前环境: $CONDA_DEFAULT_ENV"

echo ""
echo "=== 2. PyTorch / CUDA ==="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('PyTorch CUDA version (built):', torch.version.cuda)
"
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "当前环境未检测到可用 GPU。若在 WSL，请确认已安装 NVIDIA 驱动且 nvidia-smi 可用。"
  exit 1
fi

echo ""
echo "=== 3. Pip 依赖 ==="
pip install -r requirements.txt

echo ""
echo "=== 4. Submodules（CUDA 扩展）==="
if [[ -n "$1" ]]; then
  ./scripts/install_submodules.sh "$1"
else
  ./scripts/install_submodules.sh
fi

echo ""
echo "=== 完成 ==="
echo "下一步：source configs/dtu_path.env && ./scripts/run_baseline.sh \$UGGS_DTU_ROOT/scan105 ./output/dtu/scan105"
echo "若尚未下载 DTU： ./scripts/download_dtu.sh"
