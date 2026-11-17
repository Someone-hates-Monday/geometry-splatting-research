# 甲重建 P1 补审回执（仍 STOP，未提 trust）

2026-11-17。对应 `JIA_TRUST_REBUILD_REVIEW_20261117.md`。  
**未做** trust_trunc4 提取；**未写** PASS/FAIL gates；**未重训**。  
stamp：`20261117_145006`

## 对三条阻断的处理

### 1. loss／梯度兼容

- 脚本：`unified_geometry_gs/scripts/compat_jia_patchmatch_grad.py`
- 回执：`.../scan24/compat_grad_20261117_145006/compat_receipt.json`
- 方法：从 `git HEAD` 导出未改 `loss_utils.py` 作 baseline；与当前 `PatchMatch` 同输入比较。
- 预声明容差：`atol=0, rtol=0`（要求逐位一致）；`normal_enable_grad` 在 `torch.enable_grad` 下比较 ncc/geo 及 `median_depth`/`normal` 梯度。
- 结果：`ok=true`；用例均通过：`no_nearest/{default,return_trust_maps_False}`、`empty_d_mask`、`normal_enable_grad`、`single_positive_depth`、`current_trust_maps_no_nearest`。
- 冒烟脚本仍主要 `no_grad`，并在回执中写明梯度对照见 compat 脚本。

### 2. 评测 seed 固定 + 双跑

- `dtu_eval/eval.py` 增加可选 `--seed` → `np.random.default_rng(seed)`；省略则保留历史无参 `default_rng()`。
- `evaluate_dtu_mesh.py`：`--eval_seed` 透传；子进程用 `sys.executable` + 绝对路径 `dtu_eval/eval.py`；写 `eval_receipt.json`；校验有限 JSON。
- 协议冻结本实验主口径 **`eval_seed=0`**（已写入 `JIA_TRUST_REBUILD_PROTOCOL_20261116.md`）。
- 同一 `control_trunc4/recon_post.ply` 另存两次评测（不覆盖原目录）：
  - overall **0.35539816541829106**（两次 JSON 逐位相同，`eval_seed: 0`）
  - 原未固定 seed 诊断值仍保留：overall **0.3554418991327293**（sha256 未变 `d0d16341…`）

### 3. 输出防覆盖

- 新增 `utils/jia_io_guard.py`；显式 `--write_dir`/`--out_dir` 拒绝已有 `recon*.ply` / `vis/results.json` 等，并拒绝与 `model_path`（及 `--forbid_root`）同路径/嵌套冲突。
- 自检：向已有 `control_trunc4` 写入 → exit 1（`overwrite_selfcheck_ok`）。
- 旧入口（不传 write/out dir）行为保留。

## 顺序偏离

已记录：`JIA_TRUST_REBUILD_ORDER_DEVIATION.md`。105/122 模型保留，哈希不变。

## 边界诊断（部分 P2，已做／未齐）

已做：keep 显式 `isfinite(error)`；三类互斥 + geom/未评估/warp/非有限 error 细分；红图 outside_B 含非有限深度；提取前空网格检查 + `write_triangle_mesh` 返回值；冒烟公式重算 K。  
未声称全部 P2 穷尽（例如系统性 CUDA 全无效注入、gt_mask 边界专测仍可加强）。

## 主产物路径与哈希

| 项 | sha256 |
|---|---|
| compat_receipt.json | `f509aa49fe13aecea8bde86e48706de30723ff879e12745f2defba8afd0a365e` |
| smoke_receipt.json（新） | `b1cfe8fad08f9e61d2c1a0d789f7fbfe581d8c6ad17ebf7cbee3a5f86f34369e` |
| seed0 run1/2 results.json | `52fe52f57374301e5fbc28b7c87193dcf938666dd6423e300eb48ad791a1eab0` |
| legacy control results.json | `d0d163417837912350cd4f69ec9965195cf5703e525d063c3960ce0032792d4b` |
| p1_remediation_summary | `d349228f79e7bff1d90712e1b6731f5ebf7a348b59185a66232a66cce92ffb63` |
| impl_diff_p1_remediation.patch（已跟踪文件） | `a8be91f495b71196473241f3453d8c1d5995fc0bc60af6dbf9375f0db95948bf` |

**注意：** `unified_geometry_gs/dtu_eval/` 被 `.gitignore` 忽略；`--seed` 改动在本地工作树（源文件 sha256 `571d8401…`），不在 git diff 内。`jia_io_guard.py` 与两个脚本为未跟踪文件。

## 停点

请补审上述证据。通过前：**不**跑 trust 全量提取。  
keep≈74.8% 仍只是掩码诊断，不是几何收益。
