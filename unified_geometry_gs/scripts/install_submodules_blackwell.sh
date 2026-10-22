#!/usr/bin/env bash
# RTX 5090 / Blackwell (sm_120): PyTorch nightly cu128 + TORCH_CUDA_ARCH_LIST=12.0, 重编 submodules.
# 用法: ./scripts/install_submodules_blackwell.sh [CUDA 路径，如 /usr/local/cuda-12.8]
# 注意: PyTorch nightly 需要 Python 3.10+。若当前是 sugar (Python 3.9)，请改用: conda activate uggs
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$UGGS_ROOT"

PYVER=$(python -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>/dev/null || echo "0 0")
if [[ "$PYVER" == "3 9" ]]; then
  echo "当前为 Python 3.9。PyTorch nightly 可能不提供 3.9 的 wheel，建议改用 Python 3.12 环境再运行本脚本："
  echo "  conda activate uggs"
  echo "  $0 $*"
  echo "继续尝试安装 (若报 No matching distribution 请按上面切换环境) ..."
fi

unset CUDA_HOME
if [[ -n "$1" ]]; then
  CUDA_ROOT="$1"
else
  for cand in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -x "$cand/bin/nvcc" ]]; then
      ver="$("$cand/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
      if [[ "$ver" == 12.* ]]; then
        CUDA_ROOT="$cand"
        break
      fi
    fi
  done
fi

if [[ -z "$CUDA_ROOT" || ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "未找到 CUDA 12.x。请安装 CUDA 12.8+ 后执行: $0 /usr/local/cuda-12.8"
  exit 1
fi

export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"

for gcc in /usr/bin/gcc-11 /usr/bin/gcc-12; do
  if [[ -x "$gcc" ]]; then
    gxx="${gcc/gcc/g++}"
    if [[ -x "$gxx" ]]; then
      export CC="$gcc"
      export CXX="$gxx"
      echo "Using CC=$CC CXX=$CXX"
      break
    fi
  fi
done

echo "Using CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
nvcc --version

pip install ninja
# 必须用 nightly cu128 替换当前 PyTorch。若报 No matching distribution：1) 改用 conda activate uggs (Python 3.12) 2) 或网络不可达需代理/镜像
PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
if ! pip install --pre --force-reinstall torch torchvision torchaudio --index-url "$PYTORCH_INDEX" 2>/dev/null; then
  echo "尝试 cu126 源 ..."
  PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu126"
  pip install --pre --force-reinstall torch torchvision torchaudio --index-url "$PYTORCH_INDEX"
fi
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/warp-patch-ncc --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install "git+https://github.com/rahul-goel/fused-ssim/" --no-build-isolation || true
echo "Submodules (Blackwell sm_120) 安装完成。"
