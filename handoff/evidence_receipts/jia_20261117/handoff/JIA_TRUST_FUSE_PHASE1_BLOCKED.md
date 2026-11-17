# 甲可信带融合：第一阶段核验回执（BLOCKED）

2026-11-16。按 `CURSOR_PROMPT_JIA_TRUST_FUSE.md` 与 `PLAN_JIA_TRUST_FUSE.md` §6 执行。本阶段只做资产／源码／参数核验；**未改生产代码，未启动 scan24 冒烟，未开始三场景提取。**

状态：**BLOCKED**（缺冻结高斯，且 Release 清单中也没有可恢复副本）。这不是科学 FAIL。

## 1. 仓库变更（保留其他组员工作）

`git status --short` 仅见未跟踪：

- `handoff/CURSOR_PROMPT_JIA_TRUST_FUSE.md`
- `handoff/PLAN_JIA_TRUST_FUSE.md`
- `handoff/TEAM_PLAN_2026_11_16.md`
- `.vscode/`
- 本核验产生的 `handoff/jia_trust_fuse_asset_check.json`、本回执与临时脚本

无已修改的已跟踪源码文件。未触碰乙／丙／丁交付物。

## 2. P5 三场景真实模型路径（不能用 ggs_pipeline 目录名）

`scripts/run_research_p5.sh`：

- 开发组 `scan24|scan105|scan122` 的 `resolve_model_dir` 在 `ggs_ex_trunc4/<scan>/vis/results.json` 存在时返回 **`output_compare/ggs_ex_trunc4/<scan>`**。
- 训练循环对这三个场景直接 `SKIP pilot reuse`，不写入 `ggs_pipeline`。
- `judge_p5.py` 对这三个场景读 `pilot_root`（默认 `ggs_ex_trunc4`），`source: "pilot_trunc4"`。

本地目录：

| 路径 | scan24/105/122 |
|---|---|
| `ggs_pipeline/` | **不存在** |
| `ggs_ex_trunc4/` | 有网格、`vis/results.json`、`cfg_args`、`multi_view.json`；**无 `point_cloud/`**；**无 `cameras.json`** |
| `ggs_ablate_no_dn/` | 有网格、JSON、`cameras.json`、`input.ply`；**无 `point_cloud/`** |

归档对照 JSON（主比较）已在本地，字段为 `mean_d2s` / `mean_s2d` / `overall`：

| scan | 路径 | overall | sha256 |
|---|---|---:|---|
| scan24 | `ggs_ex_trunc4/.../vis/results.json` | 0.35496639222779447 | `23965a2f732becba0f5c991772cec28a940b048646bcb5aaf9810f82273a3461` |
| scan105 | 同上 | 0.47048834938509165 | `df81244cbbe91468da85297846db87bd300ee8aa9dd442328f0a679a4b449260` |
| scan122 | 同上 | 0.3242774839490299 | `70039e504a7d566f8ff5ff7a52d3246cb3c16d772c8124591de0284186071386` |

同场景 `ggs_ablate_no_dn` 的 overall 更大（例如 scan24 为 0.368…），那是默认提取（历史 trunc8）结果，**不能**当本轮 trunc4 对照。

`ggs_ex_trunc4` 的 `cfg_args` 里 `model_path` 指向自身，但历史上该网格是冻结 `no_dn` 高斯再提 trunc4 得到的；当前目录下已无高斯。完整 SHA 表见 `handoff/jia_trust_fuse_asset_check.json`。

## 3. Release 也无法补齐冻结高斯

`.asset-transfer/COMPLETE.json` 存在（90 卷已核验）。最终 `ASSET_MANIFEST.json` 中：

- `ggs_ex_trunc4/scan24` 仅 8 个条目：cfg／multi_view／recon*／vis*，**无 point_cloud**。
- `ggs_ablate_no_dn/scan24` 仅 10 个条目：cameras／cfg／input.ply／multi_view／recon*／vis*，**无 point_cloud**。
- 清单里 `point_cloud/iteration_30000` 命中来自 `ggs_pipeline` 非试点场景与 `ggs_tau_*`，不是 no_dn／trunc4 的 24／105／122。

本地现存的 15 个 `iteration_30000` 目录同样只有 pipeline 非试点与 τ 扫的 scan105。**τ 模型与 no_dn 配方不同，禁止改用。**

结论：按计划“不自行换输入／对照”，当前无法加载与归档 trunc4 JSON 对应的冻结高斯，冒烟与提取都不能合法开始。

## 4. 源码／参数语义（已核，待冒烟时再机器确认）

### NCC / patch

- `arguments`：`multi_view_patch_size=3`，`multi_view_pixel_noise_th=1.0`，`ncc_mask_thresh=0.9`。
- `train.py` 构造 `PatchMatch(opt.multi_view_patch_size, opt.multi_view_pixel_noise_th, ...)`。
- Python：`total_patch_size=(2*patch_size+1)^2`，`offsets = patch_offsets(patch_size)*0.5`。**实际 CUDA 调用不传入 patch_size**；生产核硬编码：

```text
forward_mode_differentiation_CUDA<3, true>
```

即 RADIUS=3、HALF_STEP=true，采样偏移乘 0.5，共 49 点。与默认 `patch_size=3` 一致，但改 Python 参数不会改核。
- 核输出 `output_ncc = cross^2 / (variance_r*variance_n+1e-8)`；Python `ncc = clamp(1-cc,0,2)`。**不能再平方。**
- `valid_patch` 还要求 `variance_r>5e-6` 与 `variance_n>5e-6`。
- `cfg_args` 不记录 patch／τ／λ；P5 训练命令显式传了 `ncc_mask_thresh 0.9` 与 `multi_view_ncc_views 1`，未覆盖 patch_size（依赖默认 3）。历史配置身份：**部分已知（命令脚本）+ 默认值假设**；完整 OptimizationParams 未写入磁盘。

### 近邻

- 训练 `n_mv=1`：`sample(viewpoint_cam.nearest_id, 1)` 随机。
- 提取计划：`nearest_id[0]`（排序后的固定候选），不是逐次训练掩码复现。
- `ggs_ex_trunc4` 与 `ggs_ablate_no_dn` 的 `multi_view.json` 对 scan24 同哈希 `d222ac88…`，可作为近邻表来源之一；加载高斯时仍须核对 `nearest_id` 与当前 `train_cams` 索引空间一致，且**不得**重排相机列表后沿用旧索引。

### 稠密掩码（实现约束，尚未写代码）

- CUDA 只在 `d_mask` 的 `valid_indices` 上计算。
- 现有 `return_corr_map` 对未评估位置填 0；0 ≠ 已评估／可信。
- 新增返回必须散射重建：`evaluated`／`geometry_valid`／`warp_valid`／`error`(未评估 NaN)／`keep`；默认关闭时不改训练返回与图。

### 评测

- `dtu_eval/eval.py` 写 `mean_d2s`、`mean_s2d`、`overall=(d2s+s2d)/2`。
- `evaluate_dtu_mesh.py` 从 `model_path/recon_post.ply` 读网格，写 `recon_aligned.ply` 与 `vis/`，并用 `os.system` 调 eval（退出码不可靠）——实现阶段须按 §6.5 改显式路径与退出码检查。
- DTU 图像：`dtu/DTU/scan24` 本地存在。

## 5. 未执行项

- 未改 `loss_utils.py`／`mesh_extract.py`／`evaluate_dtu_mesh.py`。
- 无源码 diff（相对已跟踪文件为零）。
- 无默认训练路径前向／梯度兼容检查。
- 无 scan24 单视角两次一致性冒烟、无红图、无 `ggs_trust_fuse/` 输出。

## 6. 缺项清单（组织者决策）

1. **必需：** 与 `ggs_ex_trunc4` trunc4 网格对应的冻结高斯  
   预期身份：历史上驱动 trunc4 提取的 `ggs_ablate_no_dn`（或当时符号链接目标）的 `point_cloud/iteration_30000`（及加载所需 sidecar）。  
   本地与 Release 清单均无。
2. 可选加固：`ggs_ex_trunc4/*/cameras.json`（可用 DTU `source_path` + `no_dn/cameras.json` 核验，但不能替代高斯）。
3. 取回后须核验：高斯文件 SHA256、与 `no_dn`／trunc4 的 `multi_view.json`／`cfg_args` 对应、再进入实现 + scan24 冒烟。

**禁止自行采用的替代：** `ggs_tau_*` 的 scan105 高斯；任意 `ggs_pipeline` 非试点高斯；重训 no_dn；用 no_dn 旧网格 JSON 当 trunc4 对照。

## 7. 请组织者审查后指示

在补齐第 6 节高斯之前，甲保持停止。需要你确认：

1. 是否另有未进 Release 的本地／外置备份路径；或  
2. 是否授权新的资产取回／上传；或  
3. 是否另立实验（那会改变预注册对照，须单独记录，不能静默替换）。

审查通过并解除 BLOCKED 后，再按提示词做：最小 diff → scan24 两次冒烟 → 停交审查 → 再 24／105／122 提取。

## 8. 组织者审查后的跟进（仍 BLOCKED，2026-11-16）

已读 `JIA_TRUST_FUSE_REVIEW_DECISION.md`。**继续停止实现与 GPU 工作。** 不下载 90 卷，不切换 τ，不重训，不用 `input.ply`／网格反推高斯。

### 8.1 Manifest 统计修正

第一阶段 `jia_trust_fuse_asset_check.json` 的 `needle_counts` 是树遍历中的**字符串出现次数**（同一逻辑文件的 `path` 与相关字段可被重复计入），**不是**独立文件数。纠正后见 `jia_trust_fuse_manifest_path_dedup.json`：

| 口径 | 数量 |
|---|---:|
| `entries` 行数 | 17820 |
| 按 `entries[].path` 去重 | 17813 |
| `ggs_ex_trunc4/scan24` 独立 path | 8 |
| `ggs_ablate_no_dn/scan24` 独立 path | 10 |
| 三场景 no_dn／trunc4 下 `point_cloud` 或 `chkpnt` | **0** |

原始 needle 计数字段保留在旧 JSON；以后以 path 去重为准，并区分 `.asset-part-*` 分段。

### 8.2 快照 ZIP 只读复查

对 `unified_geometry_gs/research/snapshots/` 下 **39** 个 ZIP 做成员名扫描：无读取错误；**0** 条成员同时落在 `ggs_ablate_no_dn|ggs_ex_trunc4` 的 scan24／105／122 路径且含 `point_cloud`／`chkpnt`／`iteration_`。局限：只覆盖该目录 ZIP 成员名，不排除改名树、外置盘、组员备份或其他 archive。

### 8.3 向原运行者／组员索取（找到前不解除 BLOCKED）

每个场景（scan24、105、122）至少需要：

1. 最终高斯：`point_cloud/iteration_30000/point_cloud.ply`（或等价最终 ply）
2. 加载所需 sidecar（若当时启用 appearance／exposure 等）
3. `cfg_args` 或完整训练命令／OptimizationParams 记录
4. `cameras.json`，或能证明与 `multi_view.json` 索引空间一致的相机来源
5. 对应的 `multi_view.json`
6. 来源说明：哪台机器、哪条路径产出了与现有 `ggs_ex_trunc4/<scan>/vis/results.json` 匹配的那次 trunc4 提取

收到候选路径后：**只做只读身份核验**（真实路径、SHA256、sidecar、与旧相机／JSON 的对应），提交补充回执；**未经组织者审查确认，不解除 BLOCKED，不开始冒烟。** 能够 `Scene` 加载成功 ≠ 已证明是原模型。

若最终确认无备份：原协议无法执行；另立实验须新冻结输入、新提取控制组、新配对门槛，旧 trunc4 JSON 仅作历史参考。
