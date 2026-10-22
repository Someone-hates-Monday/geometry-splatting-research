# Install (Unified Geometry GS)

## 一键用 GPU 训练（推荐流程）

若希望**最少步骤直接跑 GPU 训练**，按顺序执行：

1. **创建并激活 conda 环境**（与父项目 SuGaR 一致，用 conda 装 PyTorch/CUDA，避免 pip 大包 SSL 问题）  
   ```bash
   # 在 unified_geometry_gs 目录下执行（不要照抄 /path/to，用你的实际路径，例如 WSL：/mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs）
   cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
   conda env create -f environment_uggs.yml
   conda activate uggs
   ```
   若已有父项目 **sugar** 环境（PyTorch CUDA 11.8），可复用：见 [docs/SUGAR_ENV.md](docs/SUGAR_ENV.md)。

2. **安装 pip 依赖并编译 CUDA 扩展**  
   ```bash
   pip install -r requirements.txt
   ./scripts/install_submodules.sh
   ```
   使用 sugar 环境时改为：`./scripts/install_submodules.sh /usr/local/cuda-11.8`（见 SUGAR_ENV.md）。  
   若脚本报错，见下文「CUDA 与 PyTorch 版本要对齐」和「做法一」。

3. **准备 DTU 并训练**  
   ```bash
   source configs/dtu_path.env   # 若已配置 UGGS_DTU_ROOT；或先运行 ./scripts/download_dtu.sh
   ./scripts/run_baseline.sh $UGGS_DTU_ROOT/scan105 ./output/dtu/scan105
   ```

也可使用脚本 **`./scripts/setup_uggs_gpu.sh`** 做环境检查与 submodules 安装（不创建 conda 环境，需先 `conda activate uggs`）。

---

## 推荐：与 SuGaR 父项目一致，用 conda 创建环境（避免 pip 大包 SSL 错误）

父项目 SuGaR 使用 **conda 安装 PyTorch/CUDA/cudnn**，不会从 PyPI 拉取 nvidia-cudnn（约 658MB），可避免 `SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC`。本子项目提供同样方式：

```bash
# 在 unified_geometry_gs 目录下执行
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs   # 或你的实际路径
conda env create -f environment_uggs.yml
conda activate uggs
pip install -r requirements.txt
# 然后按下面「做法一」编译 submodules（需 CUDA 12.1 与 gcc-11）
./scripts/install_submodules.sh
```

**若 conda 下载中断（IncompleteRead）**：删除环境后重试 `conda env remove -n uggs && conda env create -f environment_uggs.yml`；或换网络/镜像后再试。

**若已用 pip 创建过 uggs 环境**，可保留；仅当安装 PyTorch nightly（RTX 5090）时 pip 会拉取大包并可能 SSL 失败，此时见 `docs/RTX5090_BLACKWELL.md`（可先手动装 cudnn 再装 nightly）。

---

## CUDA 与 PyTorch 版本要对齐

编译 submodules 时，**当前环境里能看到的 CUDA（nvcc/CUDA_HOME）** 必须和 **PyTorch 的 CUDA 版本**一致，否则会报：

`RuntimeError: The detected CUDA version (11.8) mismatches the version that was used to compile PyTorch (12.1)`

- 显卡/驱动需要 **CUDA 12** → 在 **WSL 里也要用 CUDA 12 工具链**，并装 PyTorch **cu121**。
- 若 WSL 里只有 CUDA 11.8，要么在 WSL 装 CUDA 12 toolkit，要么暂时用 PyTorch cu118（仅当驱动兼容时）。

先看当前环境里的 CUDA：

```bash
nvcc --version
echo $CUDA_HOME
nvidia-smi
```

---

## 显卡需要 CUDA 12：在 WSL 里用 CUDA 12 工具链 + PyTorch cu121

WSL 里默认可能是旧版 CUDA 11.8，所以编译时会“检测到 11.8”。要让 submodules 按 **CUDA 12** 编译，需要让当前 shell 使用 **CUDA 12 的 nvcc**。

### 关键：编译时必须让 PyTorch 检测到 CUDA 12.1

pip 在**子进程**里编译扩展时，会按 **CUDA_HOME** 或默认路径找 nvcc。若你本机有多个 CUDA（例如 11.8 和 13.1），且 **CUDA_HOME 未设或指向 11.8**，就会一直报 “detected 11.8”。且 PyTorch cu121 **只认 12.1**，用 13.1 的 nvcc 也会报错。所以需要：

- **编译前**在**同一 shell** 里设置 **CUDA_HOME** 指向 **CUDA 12.1** 的安装目录；
- 然后再在该 shell 里执行 `pip install submodules/...`。

---

### 做法一：系统已装 CUDA 12.1，设 CUDA_HOME 再编译（推荐）

**方式 A：用脚本一键安装（会先 unset 再选用 12.1）**

```bash
conda activate uggs
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
./scripts/install_submodules.sh
# 若自动未找到 12.1，指定路径：
./scripts/install_submodules.sh /usr/local/cuda-12.1
```

**方式 B：本终端手动设环境变量再 pip**

若 WSL 里已有 CUDA 12.1（例如 `/usr/local/cuda-12.1`）：

```bash
conda activate uggs

# 必须：让编译子进程用 12.1，不要用 11.8
unset CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

nvcc --version   # 必须显示 12.1

cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
pip install ninja
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/warp-patch-ncc --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install "git+https://github.com/rahul-goel/fused-ssim/" --no-build-isolation
```

若没有 12.1，只有 11.8 和 13.1，需要先装 [CUDA 12.1 Toolkit](https://developer.nvidia.com/cuda-12-1-0-download-archive)（选 Linux → x86_64 → WSL）。

---

### 做法二：conda 装到的是 13.1 时

`conda install cuda-nvcc -c nvidia/label/cuda-12.1` 有时会装成 13.x，和 PyTorch 12.1 不一致。可以：

- 改用**系统安装的 CUDA 12.1**（做法一），并设 `CUDA_HOME`；
- 或先查 conda 里是否有真正的 12.1：`conda search cuda-toolkit -c nvidia`，装 12.1 再设 `CUDA_HOME=$CONDA_PREFIX`。

**不要**在编译 submodules 时用 13.1 的 nvcc 配 PyTorch cu121，版本检查会失败。

---

### 若 .bashrc 里写了 CUDA_HOME=11.8

编译前在本终端里覆盖掉再装：

```bash
unset CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
# 然后再 pip install submodules/...
```

### 编译报 pybind11 / cpp_function / __is_pointer 错误（GCC 15 与 nvcc 不兼容）

若错误里出现 `pybind11/pybind11.h ... cpp_function ... does not match`、`__is_pointer is undefined`、`type_traits` 等，多半是 **nvcc 在用 conda 的 GCC 15** 做 host 编译器，和 CUDA 12 不兼容。

**解决：** 安装系统 GCC 11，并在**同一终端**里设好 CC/CXX 后再装 submodules：

```bash
sudo apt update
sudo apt install gcc-11 g++-11

conda activate uggs
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
unset CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12   # 或你的 cuda-12.1 路径
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

rm -rf submodules/diff-gaussian-rasterization/build submodules/warp-patch-ncc/build submodules/simple-knn/build
./scripts/install_submodules.sh /usr/local/cuda-12
```

或手动执行三条 `pip install submodules/...`（保持上述 CC/CXX/CUDA_HOME 已设置）。

---

### 编译时仍检测到 11.8

说明 pip 子进程用的 nvcc 不是 12.1。请在本终端执行：

1. `echo $CUDA_HOME`：若指向 11.8 或未设但默认 cuda 是 11.8，就会这样。
2. `unset CUDA_HOME` 后设 `export CUDA_HOME=/usr/local/cuda-12.1`（或你机器上 12.1 的实际路径），再 `nvcc --version` 确认是 12.1。
3. 在同一终端里重新执行 `pip install submodules/...`，不要新开终端（避免丢失 CUDA_HOME）。

若 .bashrc 里有 `export CUDA_HOME=.../cuda-11.8`，编译前一定要覆盖或 unset。

---

### 做法三：仅用 CUDA 11.8（不推荐，除非驱动只支持 11）

若暂时不装 CUDA 12 工具链，可退回 PyTorch cu118 并保证 WSL 里用的是 11.8：

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 然后按原样装 submodules（此时检测到的应是 11.8）
```

---

## Marching Tetrahedra（可选）

```bash
conda install -y conda-forge::cgal
pip install submodules/tetra_triangulation --no-build-isolation
```

---

## 准备 DTU 数据（训练 + 评估）

训练用的是 **2DGS 提供的、已用 COLMAP 预处理好的 DTU**（与 G-GS 官方一致），不是 DTU 官网的原始图像包。

### 训练用数据（COLMAP 格式）— 推荐用脚本一键下载

**下载路径**：项目内固定为 `unified_geometry_gs/data/dtu`（见 `configs/paths.yaml` 的 `dtu_root`）。下载脚本会把数据写到这里，并生成 `configs/dtu_path.env`，后续脚本会自动使用 `UGGS_DTU_ROOT`。

**一键下载并写好路径**（在项目根或 `unified_geometry_gs` 下执行均可）：

```bash
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
./scripts/download_dtu.sh
```

脚本会：安装 `huggingface_hub`、将 2DGS DTU 下载到 `data/dtu`、写入 `configs/dtu_path.env`（内容为 `export UGGS_DTU_ROOT=<绝对路径>/data/dtu`）。

**后续训练用法**：

```bash
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
source configs/dtu_path.env   # 可选：run_baseline.sh / dtu.sh 会自动 source
# 单场景
./scripts/run_baseline.sh $UGGS_DTU_ROOT/scan105 ./output/dtu/scan105
# 多场景批量（dtu.sh 已使用 UGGS_DTU_ROOT）
./scripts/dtu.sh
```

其他来源（可选）：
- **Google Drive**（约 3.5GB）：[DTU+COLMAP](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9)
- **Hugging Face**：[2DGS DTU](https://huggingface.co/datasets/dylanebert/2DGS)

下载并解压后目录结构应类似：

```text
<DTU_ROOT>/
  scan24/
    images/
    sparse/0/
  scan37/ ... scan122/
```

`run_baseline.sh` 的 `<path_to_dtu>` 必须是**单个 scan** 的路径，例如 `$UGGS_DTU_ROOT/scan105`。

### 评估用数据（官方 DTU 点云）

要做 **DTU 几何评估**（Chamfer 等）时，需要官方 DTU 点云与目录结构，并放到本仓库的 `dtu_eval` 下：

- 官方数据页面：  
  [DTU Robot Image Dataset](https://roboimagedata.compute.dtu.dk/?page_id=36)  
  下载 Point clouds 等，按 [G-GS README](https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting#2-data-preparation) 放到：

```text
unified_geometry_gs/dtu_eval/Offical_DTU_Dataset/
  Calibration/
    cal18/          # 64 视角，或 cal49（49 视角）
      pos_001.txt ... pos_064.txt   # 相机投影矩阵
  ObsMask/
    ObsMask{scan}_10.mat   # 如 ObsMask105_10.mat
    Plane{scan}.mat        # 如 Plane105.mat
  Points/
    stl/
      stl{scan:03d}_total.ply   # 如 stl105_total.ply
```

- **Calibration**：来自 DTU 的 **SampleSet** 压缩包（内含 ReadMe、calibration、evaluation code）。
- **ObsMask**：同样在 SampleSet 中。
- **Points/stl**：来自 **Points.zip**（STL reference point clouds）。

评估命令示例：

```bash
python evaluate_dtu_mesh.py -m <output_dir>
```

其中 `<output_dir>` 为本次训练/网格导出的输出目录（与 `run_baseline.sh` 的第二个参数一致）。
