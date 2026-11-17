# 新配对实验：按冻结配方重建（非旧训练逐位复现）

2026-11-16。依据 `CURSOR_PROMPT_JIA_LOCAL_RECOVERY_REBUILD.md`。本地有界查找未恢复原 no_dn 高斯（见 `JIA_LOCAL_RECOVERY_AUDIT.md`）。旧 `ggs_ex_trunc4/*/vis/results.json` **仅历史参考**，不得作本轮主控制组。

## 为何不可恢复

- P5 试点模型身份是驱动 `ggs_ex_trunc4` 的冻结 no_dn 高斯。
- 本地与 Release 清单均无三场景 `point_cloud/iteration_30000`。
- 微信 `uggs-m1-guizang*.zip`、39 个 `research/snapshots` ZIP、Desktop/Documents/Downloads/WSL home、D: 浅层候选均未给出可接受原模型。

## 配置身份：有证据 vs 当前默认

**有脚本/命令证据（按此重建）：**

| 项 | 值 | 来源 |
|---|---|---|
| 数据 | `UGGS_DTU_ROOT=/mnt/c/Users/Someone/Desktop/SuGaR-main/dtu/DTU` | `configs/dtu_path.env` |
| 分辨率 | `-r 2` | `run_research_p5.sh` |
| appearance | `--use_decoupled_appearance 3` | 同上 |
| λ_depth_normal | 0 | 同上 |
| λ_multi_view_ncc | 0.6 | 同上 |
| λ_multi_view_geo | 0.02 | 同上 |
| multi_view_ncc_views | 1（训练时随机近邻） | 同上 |
| ncc_mask_thresh | 0.9 | 同上 |
| iterations | 30000 | `OptimizationParams` 默认 + P5 checkpoint |
| checkpoint | 仅 30000 | P5 `CHECKPOINT_ITERS` |
| seed | `safe_state`: random/np/torch seed 0 | `general_utils.safe_state` |
| 提取 | voxel 0.002, trunc 4, depth_max 8, cluster 1 | P5 extract |

**仅能采用当前代码默认（未写入历史 cfg_args，故称「按冻结配方重建」）：**

- densify_from/until/interval、grad_threshold、λ_dssim、appearance LR、patch_size 默认 3、pixel_noise_th 默认 1.0、CUDA 核硬编码 RADIUS=3 HALF_STEP、其余 OptimizationParams。

不声称与旧 no_dn 逐位相同。

## 目录

```
unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116/<scan>/
  model/              # 训练输出（-m）
  control_trunc4/     # 无掩码提取 + 历史未固定 shuffle 的评测（只读保留）
  control_trunc4_eval_seed0_run*/  # 同一 control 网格、显式 seed=0 的评测（主对照）
  trust_trunc4/       # 强掩码提取 + seed=0 评测（审查通过后）
  smoke_*/ compat_*/  # 冒烟与兼容回执（独立空目录）
  logs/
```

已存在则换新 run 名，禁止覆盖。显式 `--write_dir` / `--out_dir` 在写入前拒绝已有 `recon*.ply`、`vis/results.json` 等，并拒绝与 `model/` 同路径或嵌套冲突。

## 评测 seed（2026-11-17 冻结）

- `dtu_eval/eval.py --seed` 显式传入；`np.random.default_rng(seed)` 负责降采样前 shuffle。
- 父进程 `np.random.seed` **不能**替代该参数。
- 本重建实验主口径冻结 **`eval_seed=0`**。历史无 seed 的 `control_trunc4/vis/results.json`（scan24 overall≈0.3554418991）仅作诊断，不混作主对照。
- 调用使用 `sys.executable` + 仓库内 `dtu_eval/eval.py` 绝对路径；写 `eval_receipt.json`（命令、退出码、seed、结果）。

## 预算与顺序

- 场景最多 24、105、122；每场景只训一次到 30000。
- 不扫种子/λ/超参；不做 TGS。
- **协议顺序**：先只训 scan24 → control → 冒烟+兼容 → **停审** → 通过后再训 105/122 并提 trust。
- 实际执行曾在 scan24 审查停点前排队训完 105/122（见 `JIA_TRUST_REBUILD_ORDER_DEVIATION.md`）。**不丢弃、不重训**；冻结三份 ply 哈希继续用。
- 未经补审通过：不提 trust 网格、不写 PASS/FAIL gates。
- 故障：保留日志与最后有效 checkpoint；仅允许同一冻结配置续跑。OOM/NaN 停诊，不扩大预算。

## 配对评测（补审通过后）

同一 `model` 哈希上：

1. `control_trunc4` 网格（不重提）+ **seed=0** 新评测目录  
2. `trust_trunc4`：强掩码提取 + **seed=0** 评测  

差值 = trust − control（负 overall 更好）。门槛：

- mean Δoverall ≤ −5e−4 mm  
- mean Δs2d ≤ 0  
- scan122 Δoverall ≤ +5e−4 mm  

旧 trunc4 JSON 与未固定 seed 的 control JSON 不参与判定。keep 比例仅为掩码诊断，不是几何收益证据。
