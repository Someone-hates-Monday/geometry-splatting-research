# RTX 5090 / Blackwell (sm_120) 说明

RTX 5090 使用 **Blackwell 架构 (sm_120)**。当前 **稳定版 PyTorch** 的预编译包只支持到 sm_90，因此会报：

- `NVIDIA GeForce RTX 5090 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation`
- `RuntimeError: CUDA error: no kernel image is available for execution on the device`

---

## 方案一：先用 CPU 跑通流程（立即可用）

不换 PyTorch、不重编扩展，仅用 CPU 验证数据和脚本（**会很慢**，仅建议短跑或调试）：

```bash
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
conda activate uggs
source configs/dtu_path.env

# 禁用 GPU，用 CPU
export CUDA_VISIBLE_DEVICES=""

# 少跑几步看是否报错（例如 100 步）
python train.py -s "$UGGS_DTU_ROOT/scan105" -m ./output/dtu/scan105_cpu -r 2 --use_decoupled_appearance 3 --iterations 100
```

若只想确认“能跑起来”，用 `--iterations 100` 即可；正式训练请用方案二。

---

## 方案二：用脚本一键装好 GPU 支持（推荐）

在 WSL 里执行（会装 PyTorch nightly cu128 + 为 sm_120 重编子模块）：

```bash
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
conda activate uggs

# 若脚本有 CRLF 报错，先执行：
sed -i 's/\r$//' scripts/install_submodules_blackwell.sh

chmod +x scripts/install_submodules_blackwell.sh
./scripts/install_submodules_blackwell.sh
```

脚本会：检测 CUDA 12.8+、安装 PyTorch nightly (cu128)、设 `TORCH_CUDA_ARCH_LIST=12.0`、重编 diff-gaussian-rasterization / warp-patch-ncc / simple-knn。

**若系统没有 CUDA 12.8**，先安装 [CUDA 12.8+ Toolkit](https://developer.nvidia.com/cuda-downloads)（Linux → x86_64 → WSL），再执行：

```bash
./scripts/install_submodules_blackwell.sh /usr/local/cuda-12.8
```

**需要 gcc-11**（避免 nvcc 与 GCC 15 不兼容）：

```bash
sudo apt update && sudo apt install -y gcc-11 g++-11
```

安装完成后直接训练：

```bash
source configs/dtu_path.env
./scripts/run_baseline_scan105.sh
```

若仍有 `no kernel image`，运行前加 `export CUDA_LAUNCH_BLOCKING=1` 再试，并把完整报错贴到 PyTorch/扩展的 issue。

---

### 若安装 PyTorch nightly 时出现 SSL / 下载错误

下载 **nvidia-cudnn-cu12**（约 658MB）时反复报 `SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC`，多为当前网络/代理在长连接下不稳定。

#### 第一步：先恢复可用的 PyTorch（cu121）

避免环境里没有 torch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 第二步（二选一）：换方式装 nightly

**方式 A：换网络后重试**

关代理、换手机热点或改天再试，直接重试：

```bash
pip uninstall -y torch torchvision 2>/dev/null
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --timeout 300
```

**方式 B：先手动装 cudnn，再装 torch（推荐，避免 pip 一次拉 658MB）**

1. **在 Windows 浏览器**里打开（或能稳定下载的环境）：
   - https://pypi.nvidia.com/nvidia-cudnn-cu12/
   - 找到版本 **9.19.0.56**，下载 **manylinux**、**x86_64** 的 `.whl`（文件名类似 `nvidia_cudnn_cu12-9.19.0.56-py3-none-manylinux_2_27_x86_64.whl`），保存到 `C:\Users\Someone\Downloads\`。

2. **在 WSL** 里先装该 wheel，再装 PyTorch nightly：
   ```bash
   conda activate uggs
   pip install /mnt/c/Users/Someone/Downloads/nvidia_cudnn_cu12-9.19.0.56-*.whl
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --timeout 300
   ```
   若 Downloads 里文件名不同，把上面路径改成你实际的文件名。

3. **再重编子模块**（同上）：
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.8
   export TORCH_CUDA_ARCH_LIST=12.0
   export CC=/usr/bin/gcc-11
   export CXX=/usr/bin/g++-11
   cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
   rm -rf submodules/diff-gaussian-rasterization/build submodules/warp-patch-ncc/build submodules/simple-knn/build
   pip install ./submodules/diff-gaussian-rasterization ./submodules/warp-patch-ncc ./submodules/simple-knn --no-build-isolation
   ```

---

## 参考

- [PyTorch issue: Add official support for CUDA sm_120 (RTX 5090 / Blackwell)](https://github.com/pytorch/pytorch/issues/159207)
- [PyTorch Blackwell tracking](https://github.com/pytorch/pytorch/issues/145949)
- CUDA 12.8+ 才支持 sm_120；12.6/12.7 不支持 Blackwell。
