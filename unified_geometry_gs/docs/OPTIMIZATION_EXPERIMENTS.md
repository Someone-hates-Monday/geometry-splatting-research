# 在 M1 基础上的进一步优化尝试

当前基线：**G-GS**（深度–法线 + 多视图 NCC/几何）+ **M1**（`L_occ` + `L_depth_occ`）。改进空间主要来自 **调参**、**更早/更强的几何正则**、**多场景统计**。

---

## 0. 主目标：网格质量（推荐一键预设）

若 **主要优化 DTU mesh 指标**（`mean_d2s` / `mean_s2d` / `overall`，越小越好），可直接用：

```bash
./scripts/run_m1_mesh_focus.sh "$UGGS_DTU_ROOT/scan105" ./output_compare/m1_meshfocus/scan105
```

该预设相对默认 M1 做了：

| 变量 | 预设值 | 意图 |
|------|--------|------|
| `REG_FROM_ITER` | 4500 | 更早启用深度/多视图/M1，多训练几千步几何项 |
| `LAMBDA_OCC` | 0.06 | 略加强占用正则（默认 0.02） |
| `LAMBDA_DEPTH_OCC` | **0.15** | **重点加强**深度–占用耦合（与网格边界更相关） |
| `OCC_SPARSITY_WEIGHT` | 0.08 | 略鼓励更紧凑的 alpha |
| `DEPTH_OCC_MIN_ALPHA` | 0.04 | 略放宽前景 mask，让更多边界参与耦合 |

训练结束后与 baseline / 原 M1 对比：

```bash
python scripts/compare_models.py \
  --baseline ./output_compare/ggs_baseline/scan105 \
  --optimized ./output_compare/m1_meshfocus/scan105 \
  --dtu_eval_root ./dtu_eval/Offical_DTU_Dataset \
  --out ./output_compare/metrics/scan105_baseline_vs_m1_meshfocus.json
```

若 **overall 变好但 PSNR 掉太多**，可只减弱 `LAMBDA_OCC` 或把 `LAMBDA_DEPTH_OCC` 调到 `0.10`～`0.12` 再试。

### 温和档（推荐在 mesh_focus 失败后先试）

`run_m1_mesh_focus.sh` 过强时易出现 **d2s↓ 但 s2d↑、overall↑**。可用中间预设：

```bash
./scripts/run_m1_mesh_mid.sh "$UGGS_DTU_ROOT/scan105" ./output_compare/m1_meshmid/scan105
```

默认：`REG_FROM_ITER=5000`，`LAMBDA_OCC=0.04`，`LAMBDA_DEPTH_OCC=0.10`，`OCC_SPARSITY_WEIGHT=0.06`。

---

## 1. 用环境变量快速试参（推荐）

`scripts/run_m1.sh` 支持以下环境变量（默认值与当前 M1 一致）：

| 变量 | 默认 | 含义 |
|------|------|------|
| `LAMBDA_OCC` | 0.02 | 占用正则权重 |
| `LAMBDA_DEPTH_OCC` | 0.05 | 深度–占用耦合权重 |
| `REG_FROM_ITER` | 7000 | 从第几步开始启用深度/多视图/M1（越早越早用几何项） |
| `OCC_ENTROPY_WEIGHT` | 1.0 | alpha 熵项系数 |
| `OCC_SPARSITY_WEIGHT` | 0.05 | alpha 稀疏项系数 |
| `DEPTH_OCC_MIN_ALPHA` | 0.05 | 深度–占用损失中前景 mask 阈值 |

示例（更强 M1，更早开正则）：

```bash
cd unified_geometry_gs
source configs/dtu_path.env  # 若使用 UGGS_DTU_ROOT

REG_FROM_ITER=5000 LAMBDA_OCC=0.04 LAMBDA_DEPTH_OCC=0.08 \
  ./scripts/run_m1.sh "$UGGS_DTU_ROOT/scan105" ./output_compare/m1_strong/scan105
```

训练结束后用 `compare_models.py` 与 baseline / 原 M1 对比。

---

## 2. 建议的实验矩阵（单场景可先扫 2～3 组）

在 **scan105** 上可先试：

1. **M1-default**：当前默认（已有）。
2. **M1-strong**：`LAMBDA_OCC=0.04`，`LAMBDA_DEPTH_OCC=0.08`（观察 mesh overall 是否下降，PSNR 是否可接受）。
3. **M1-early**：`REG_FROM_ITER=5000` 或 `4000`，其余与 default 相同（让 M1 多训练几千步）。

若某组 **overall 明显下降** 且 **PSNR 掉太多**，再在该组附近微调（例如只加 `lambda_depth_occ`）。

---

## 3. 多场景更稳

单 scan 的 Chamfer 噪声大。建议在 **3～5 个 DTU scan** 上重复训练 + `compare_models`，报告 **平均 overall** 与标准差。

---

## 4. 与 M2 的关系

`docs/MVP_milestones.md` 中的 **M2（置信度/语义加权）**：

- **已实现（M2-w_geo）**：对 PatchMatch 几何项做置信度加权。  
  默认采用 **M2-v2（ncc-only）**：仅加权 `ncc_loss`，不加权 `geo_loss`（更稳）。  
  开关：`--use_w_geo`，参数：`--w_geo_beta`、`--w_geo_min`、`--w_geo_ncc_only`。  
  脚本：`scripts/run_m2.sh`。
- **未实现（M2-w_sem）**：语义 mask/分割加权（后续可做）。

说明：当前 w_geo 是**标量权重**（从 PatchMatch 有效匹配比例与 geo_loss 映射得到），先用于稳定跨场景表现；后续可扩展为像素级权重。

---

## 5. 评估命令

```bash
python scripts/compare_models.py \
  --baseline ./output_compare/ggs_baseline/scan105 \
  --optimized ./output_compare/<你的新输出>/scan105 \
  --dtu_eval_root ./dtu_eval/Offical_DTU_Dataset \
  --out ./output_compare/metrics/scan105_vs_<name>.json
```

确保双方均有 `recon_post.ply`（`mesh_extract.py` 已跑）。
