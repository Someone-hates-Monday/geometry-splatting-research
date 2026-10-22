#!/usr/bin/env bash
# 在编译 submodules 前设置 CUDA 环境（需与当前 PyTorch 的 CUDA 版本一致）。
# 用法: ./scripts/install_submodules.sh [CUDA 路径]
# 例:   ./scripts/install_submodules.sh                    # 自动查找 CUDA 12.x（uggs 环境）
#       ./scripts/install_submodules.sh /usr/local/cuda-12.1
#       ./scripts/install_submodules.sh /usr/local/cuda-11.8  # 使用父项目 sugar 环境时
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$UGGS_ROOT"

unset CUDA_HOME

EXPLICIT_CUDA=""
if [[ -n "$1" ]]; then
  CUDA_ROOT="$1"
  EXPLICIT_CUDA=1
else
  for cand in /usr/local/cuda-12.8 /usr/local/cuda-12.1 /usr/local/cuda-12 /usr/local/cuda; do
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
  echo "未找到 CUDA。请安装后执行："
  echo "  $0 /path/to/cuda-12.1   # uggs 环境（PyTorch cu121）"
  echo "  $0 /path/to/cuda-11.8   # sugar 环境（PyTorch cu118）"
  echo "RTX 5090 请用: ./scripts/install_submodules_blackwell.sh"
  exit 1
fi

export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

for gcc in /usr/bin/gcc-11 /usr/bin/gcc-12; do
  if [[ -x "$gcc" ]]; then
    gxx="${gcc/gcc/g++}"
    if [[ -x "$gxx" ]]; then
      export CC="$gcc"
      export CXX="$gxx"
      echo "Using CC=$CC CXX=$CXX (nvcc host compiler)"
      break
    fi
  fi
done
if [[ -z "$CC" ]]; then
  echo "警告: 未找到 /usr/bin/gcc-11 或 gcc-12。若编译报 pybind11/cpp_function 错误，请安装: sudo apt install gcc-11 g++-11"
fi

echo "Using CUDA_HOME=$CUDA_HOME"
nvcc --version
ver="$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
if [[ -n "$EXPLICIT_CUDA" ]]; then
  if [[ "$ver" != 11.* && "$ver" != 12.* ]]; then
    echo "错误: nvcc 版本为 $ver，需要 11.x 或 12.x。"
    exit 1
  fi
else
  if [[ "$ver" != 12.* ]]; then
    echo "错误: 未指定路径时仅支持 CUDA 12.x（当前为 $ver）。使用 sugar 环境请传 11.8 路径: $0 /usr/local/cuda-11.8"
    exit 1
  fi
fi

pip install ninja
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/warp-patch-ncc --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install "git+https://github.com/rahul-goel/fused-ssim/" --no-build-isolation || true
echo "Submodules 安装完成（fused-ssim 若网络失败可稍后重试）。"
