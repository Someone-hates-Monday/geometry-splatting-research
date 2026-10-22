# M1 调参与对比记录（精简版）

> 记号：Δ = optimized − baseline。  
> 图像：L1↓ / PSNR↑ / SSIM↑ 更好；网格：mean_d2s↓ / mean_s2d↓ / overall↓ 更好（DTU Chamfer）。

---

## 0. 环境与评估准备

- **Mesh 评估数据**：DTU 官方 `SampleSet.zip`（Calibration/ObsMask）+ `Points.zip`（Points/stl），整理到 `dtu_eval/Offical_DTU_Dataset/` 后才能跑网格指标。
- **对比脚本**：`scripts/compare_models.py`（已支持 DTU eval root 检查/解析）。

---

## 1. 单场景（scan105）尝试

### 1.1 M1-default（`run_m1.sh`）

- **思路**：在 G-GS 基线（深度-法线 + 多视图 NCC/几何）上，加 M1 的 `L_occ + L_depth_occ`。
- **参数**（来自脚本默认/示例）：`lambda_occ=0.02`，`lambda_depth_occ=0.05`，`reg_from=7000`。
- **对比结果**：`scan105_baseline_vs_m1.json`
  - **Image Δ**：L1 **−0.000766**，PSNR **+0.158978**，SSIM **−0.001049**
  - **Mesh Δ**：d2s **−0.000129**，s2d **+0.000535**，overall **+0.000203**
- **结论**：图像略优；网格综合（overall）略差（d2s↓但 s2d↑ 抵消并反超）。

### 1.2 M1-meshfocus（`run_m1_mesh_focus.sh`）

- **思路**：更早开正则 + 更强 depth-occ（目标：更好 mesh）。
- **参数（预设）**：`reg_from=4500`，`lambda_occ=0.06`，`lambda_depth_occ=0.15`（以及更强稀疏/更低 min_alpha）。
- **对比结果**：`scan105_baseline_vs_m1_meshfocus.json`
  - **Image Δ**：L1 **+0.000322**，PSNR **−0.366300**，SSIM **−0.003258**
  - **Mesh Δ**：d2s **−0.001363**，s2d **+0.005838**，overall **+0.002237**
- **结论**：强正则导致 s2d 明显变差，overall 变差；图像也变差。

### 1.3 M1-meshmid（`run_m1_mesh_mid.sh`）

- **思路**：meshfocus 的温和版。
- **参数（预设）**：`reg_from=5000`，`lambda_occ=0.04`，`lambda_depth_occ=0.10`（及轻度稀疏）。
- **对比结果**：`scan105_baseline_vs_m1_meshmid.json`
  - **Image Δ**：L1 **+0.000284**，PSNR **−0.366864**，SSIM **−0.003193**
  - **Mesh Δ**：d2s **+0.003147**，s2d **+0.010474**，overall **+0.006810**
- **结论**：全面变差（不采用）。

### 1.4 单变量扫 `lambda_depth_occ`（scan105）

固定：`lambda_occ=0.02`，`reg_from=7000`；仅调 `lambda_depth_occ`。

- **vd06**：`lambda_depth_occ=0.06`（`scan105_baseline_vs_m1_vd06.json`）
  - **Mesh Δ**：overall **+2.19e−05**（略差；d2s↓、s2d↑）
- **vd07**：`lambda_depth_occ=0.07`（`scan105_baseline_vs_m1_vd07.json`）
  - **Mesh Δ**：overall **+2.63e−05**（略差；d2s↓、s2d↑）
- **vd08**：`lambda_depth_occ=0.08`（`scan105_baseline_vs_m1_vd08.json`）
  - **Mesh Δ**：overall **−5.32e−05**（略好；d2s↓、s2d↑）
- **vd09**：`lambda_depth_occ=0.09`（`scan105_baseline_vs_m1_vd09.json`）
  - **Mesh Δ**：overall **+3.97e−04**（这次跑出来整体变差；此前不同 run 曾出现更好，说明单场景波动/不稳定）

**单场景结论**：在 scan105 上调 `lambda_depth_occ` 会出现明显的 d2s/s2d 权衡，且结果存在 run-to-run 波动；不能只凭单场景下结论。

---

## 2. 多场景验证（scan24/105/122）

### 2.1 vd09 多场景（`lambda_depth_occ=0.09`）

对比文件：
- `scan24_baseline_vs_m1_vd09.json` → **overall Δ +0.000158**
- `scan105_baseline_vs_m1_vd09.json` → **overall Δ +0.000397**
- `scan122_baseline_vs_m1_vd09.json` → **overall Δ −0.000385**

**平均**（3 scans）：overall **+0.000056**（略差）

**结论**：vd09 跨场景不稳定（不能作为推荐默认）。

### 2.2 “保守版”多场景（`m1_vd06_multi`: `lambda_depth_occ=0.06`）

对比文件：
- `scan24_baseline_vs_m1_vd06_multi.json` → **overall Δ +0.000230**
- `scan105_baseline_vs_m1_vd06_multi.json` → **overall Δ −0.000232**
- `scan122_baseline_vs_m1_vd06_multi.json` → **overall Δ −0.000606**

**平均**（3 scans）：overall **−0.000203**（更好）

**结论**：相对 vd09 更稳，3-scan 平均 overall 下降；但 scan24 仍出现 overall 上升（主要来自 s2d↑）。

---

## 3. 当前阶段结论（暂定）

- 若以 **多场景平均 mesh overall** 为主：`lambda_occ=0.02`、`lambda_depth_occ=0.06`、`reg_from=7000` 是目前更稳的候选。
- 若要进一步提升并减少 “d2s↓但 s2d↑” 的情况：需要转向更直接的几何策略（例如提取/后处理、或 M2 的加权几何），而不是继续盲目增大 M1 权重。

---

## 4. M1 消融协议（目录命名 + 一键脚本）

固定 **迭代数、数据、`mesh_extract`、评估数据路径**，只改 M1 项，便于写进论文式消融表。

### 4.1 消融档位（相对同一 `ggs_baseline`）

| 编号 | 含义 | `lambda_occ` | `lambda_depth_occ` | `lambda_depth_occ_rank` | 输出目录（每个 scan 下） |
|------|------|--------------|--------------------|-------------------------|--------------------------|
| M1-0 | 基线（无 M1） | 0 | 0 | 0 | `output_compare/ggs_baseline/<scan>`（需事先训好） |
| M1-1 | 仅 L_occ | 0.02 | 0 | 0 | `output_compare/m1_ablate_01_occ_only/<scan>` |
| M1-2 | 仅 L_depth_occ | 0 | 0.06 | 0 | `output_compare/m1_ablate_02_depth_occ_only/<scan>` |
| M1-3 | L_occ + L_depth_occ | 0.02 | 0.06 | 0 | `output_compare/m1_ablate_03_full/<scan>` |
| M1-4 | M1-3 + 结构化 ranking | 0.02 | 0.06 | 0.04 | `output_compare/m1_ablate_04_full_rank/<scan>` |

公共约定：`regularization_from_iter=7000`，其余与 `run_m1.sh` 一致（`-r 2`、`--use_decoupled_appearance 3` 等）。`run_m1_ablation.sh` 可用环境变量微调各档位权重：`LAMBDA_OCC_A1`、`LAMBDA_OCC_A3`、`LAMBDA_OCC_A4`、`LAMBDA_DEPTH_OCC_A2`、`LAMBDA_DEPTH_OCC_A3`、`LAMBDA_DEPTH_OCC_RANK_A4`、`REG_FROM_ITER`。

### 4.2 对比结果 JSON 命名

每个 scan、每个消融档位一条：

`output_compare/metrics/<scan>_baseline_vs_<档位目录名>.json`

例如：`scan105_baseline_vs_m1_ablate_03_full.json`。

### 4.3 命令（WSL / Linux）

```bash
cd /path/to/unified_geometry_gs
source configs/dtu_path.env
conda activate uggs   # 或你的训练环境
export PYTHON=python  # 勿用系统 python3

# 训练 M1-1～M1-4 并逐个与 baseline 对比（默认 scan24 scan105 scan122）
./scripts/run_m1_ablation.sh scan24 scan105 scan122

# 仅汇总已有 metrics（跳过训练）
ABLATION_SKIP_TRAIN=1 ./scripts/run_m1_ablation.sh scan24 scan105 scan122

# 汇总表：按档位打印多场景平均 Δ
python scripts/summarize_m1_ablation.py --metrics_dir ./output_compare/metrics
```

Windows（PowerShell，需已配置 `py -3` 与 DTU 路径；脚本会尝试把 `dtu_path.env` 里 `/mnt/c/...` 转成 `C:\...`）：

```powershell
cd <你的 unified_geometry_gs 根目录>
.\scripts\run_m1_ablation.ps1 scan24 scan105 scan122
```

### 4.4 填表建议

在下方或附录记录：**日期、scan 列表、seed（若有）、各档位平均 overall Δ 与 std、最差 scan**。

| 档位 | 平均 overall Δ | 平均 PSNR Δ | 备注 |
|------|------------------|-------------|------|
| M1-1 | −2.98e−4 | −0.47 | occ only；mesh 有升有降 |
| M1-2 | −6.1e−5 | −0.14 | depth_occ only；mesh 最弱 |
| M1-3 | **−7.83e−4** | −0.23 | **本批平均 mesh 最好** |
| M1-4 | −3.42e−4 | −0.29 | +rank；平均不及 M1-3 |

### 4.5 消融结果（2025-03-30，scan24 / scan105 / scan122）

**设定**：与 §4.1 一致；baseline 为 `ggs_baseline`；Δ = optimized − baseline（网格 overall↓ 更好）。

**多场景平均（`summarize_m1_ablation.py`）**

| 档位 | 平均 overall Δ | 平均 mean_d2s Δ | 平均 mean_s2d Δ | 平均 PSNR Δ | 平均 L1 Δ |
|------|----------------|-----------------|-----------------|-------------|-----------|
| M1-1 occ only | **−2.98e−4** | −7.98e−4 | +2.01e−4 | −0.47 | +6.86e−4 |
| M1-2 depth_occ only | −6.1e−5 | −8.4e−5 | −3.7e−5 | −0.14 | +1.6e−4 |
| M1-3 full | **−7.83e−4** | −5.77e−4 | −9.89e−4 | −0.23 | +2.8e−4 |
| M1-4 full+rank | −3.42e−4 | −8.83e−4 | +1.98e−4 | −0.29 | +6.7e−4 |

**分场景 overall Δ（便于看方差）**

| scan | M1-1 | M1-2 | M1-3 | M1-4 |
|------|------|------|------|------|
| scan24 | +1.38e−4 | +9.9e−5 | **−1.03e−3** | −4.75e−4 |
| scan105 | −8.84e−4 | −6.2e−5 | **−9.55e−4** | −3.6e−5 |
| scan122 | −1.49e−4 | −2.18e−4 | −3.65e−4 | **−5.16e−4** |

**简要结论**

- **平均 mesh overall**：**M1-3（L_occ + L_depth_occ）最好**，优于单独一项（M1-1、M1-2）与 **M1-4（+rank）**。本批实验里 **结构化 ranking 未带来平均收益**，但在 **scan122** 上 M1-4 的 overall 略优于 M1-3（−5.16e−4 vs −3.65e−4），需更多 scan 判断是否场景相关。
- **图像**：各档 PSNR 多为负，**scan122** 在 M1-1 上 PSNR Δ 约 −1.11，需权衡画质与 mesh。
- **下一步**：扩到更多 DTU scan 验证 M1-3 vs M1-4；若坚持 ranking，可扫 **`lambda_depth_occ_rank`**（如 0.02 / 0.03）或略降 **`lambda_occ`** 以减轻与图像目标的冲突。

### 4.6 路线 B（主会向）：法线引导 occupancy 锐化

**动机**：在渲染法线与深度法线一致的区域，对半透明 α（`4α(1−α)`）加权惩罚，抑制「几何已自洽却仍糊在半透明」的 floater / mush，作为贴近 mesh（尤其 s2d）目标的 surrogate；与 M1-3 的 occ/depth_occ 互补。

**实现**：`losses/mesh_aware_occ.py` 中 `compute_normal_guided_ambiguity_loss`；训练项 **`lambda_occ_ng_sharp`**（默认 0）、温度 **`occ_ng_sharp_beta`**（默认 5）。开启 B 时会与 depth-normal / patchmatch 共用 `depth_to_normal` 分支（无需单独再开 `lambda_depth_normal`，但可与之一同使用）。

**跑法**：`./scripts/run_m1_b_mesh_align.sh <scans...>`；可调环境变量 **`LAMBDA_OCC_NG_SHARP`**（建议先试 0.01–0.03）、**`OCC_NG_SHARP_BETA`**。

**填表**：主表建议 **baseline vs M1-3 vs M1-B**（同管线、同 mesh 协议）；对外方法（如 SuGaR）单独成章并注明公平性。

### 4.7 深度边缘加权法线一致（拐角 / 突变）

**动机**：全局 depth-normal 已在 baseline 上较饱和；在 **深度 Sobel 幅值大** 的像素提高权重，可把梯度推向 **窗洞、棱角** 等待改进区域。

**实现**：`losses/edge_depth_normal.py`；当 **`depth_normal_edge_boost > 0`** 且 **`lambda_depth_normal > 0`** 时，将原先均匀 mean 换为加权 mean：`w = 1 + boost * ê`，其中 `ê` 为深度梯度幅值在 valid 像素上分位数归一化到 [0,1]（默认用 5%–95% 分位作鲁棒 min-max）。

**参数**：`--depth_normal_edge_boost`（默认 0=关闭）、`--depth_normal_edge_q_low` / `--depth_normal_edge_q_high`。

**试跑示例**：在 **保留你原有** `-s` / `-m` / `-r` 等参数的前提下，**末尾追加**下列两项即可。**不要**把文档里的占位符 `...` 打进命令行，否则 argparse 会报 `unrecognized arguments: ...`。

DTU 风格（与 `scripts/dtu.sh`、`run_baseline_scan105.sh` 一致，按需改路径与 scan）：

```bash
python train.py -s "$UGGS_DTU_ROOT/scan105" -m ./output/dtu/scan105_edge_dn -r 2 --use_decoupled_appearance 3 \
  --lambda_depth_normal 0.05 --depth_normal_edge_boost 2.0
```

若你已有完整训练命令，只需在原命令后加上：`--lambda_depth_normal 0.05 --depth_normal_edge_boost 2.0`。

 