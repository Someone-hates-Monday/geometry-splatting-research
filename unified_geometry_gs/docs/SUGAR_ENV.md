# 使用父项目 SuGaR 的 conda 环境运行 uggs（可选）

若已在 SuGaR 父项目下用 `environment.yml` 创建了 **sugar** 环境（Python 3.9 + PyTorch 2.0.1 CUDA 11.8），可以直接复用该环境在 unified_geometry_gs 下用 GPU 训练，无需再建 uggs 环境。

## 步骤

```bash
# 1. 在 SuGaR 根目录创建 sugar 环境（若尚未创建）
cd /mnt/c/Users/Someone/Desktop/SuGaR-main   # 或你的 SuGaR 根目录
conda env create -f environment.yml
conda activate sugar

# 2. 进入 unified_geometry_gs 并安装 pip 依赖（若当前已在 unified_geometry_gs 下可省略 cd）
cd unified_geometry_gs
pip install -r requirements.txt

# 3. 编译 CUDA 扩展：必须使用与 PyTorch 一致的 CUDA 11.8
#    若系统有 /usr/local/cuda-11.8，直接传路径：
./scripts/install_submodules.sh /usr/local/cuda-11.8

#    若 conda 的 cuda 在 CONDA_PREFIX，可尝试（需 conda 已装 cuda-toolkit）：
#    ./scripts/install_submodules.sh "$CONDA_PREFIX"
#    否则先安装系统 CUDA 11.8 或使用 conda: conda install cuda-toolkit=11.8 -c nvidia
```

## 注意

- **sugar** 为 CUDA **11.8**；**uggs** 的 `environment_uggs.yml` 为 CUDA **12.1**。使用 sugar 时编译 submodules 需指定 11.8 的路径，否则会报版本不匹配。
- 若希望统一用 CUDA 12，请使用 uggs 环境：`conda env create -f environment_uggs.yml`，见 [INSTALL.md](../INSTALL.md)。

## 训练

```bash
conda activate sugar
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs   # 或你的路径（若已在此目录可省略）
source configs/dtu_path.env   # 若已配置 DTU 路径
./scripts/run_baseline.sh $UGGS_DTU_ROOT/scan105 ./output/dtu/scan105
```
