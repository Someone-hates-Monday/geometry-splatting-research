# 甲重建全量配对结果（交最终审查）

2026-11-17。stamp=`20261117_154810`。技术流程完成；**门槛未过**（`pass: false`）。未重训、未改 τ、未扩场景。

## 执行前自检

- 源码身份快照：`unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116/source_identity_20261117_154810/`  
  （含被 gitignore 的 `dtu_eval/eval.py` 副本 + 未跟踪辅助源码）
- `loss_utils` sha256=`39be74a64b7f954cdcfadf1aedccf5845c7512332eacb7a80a177a2c10455d2e`（与 compat 回执一致）
- 严格 `eval_seed`：缺字段 / null / 错值自检失败；正确 `0` 通过
- 全量日志：`.../full_extract_20261117_154810.log`

## 门槛判定（未舍入，三场景等权）

| 门槛 | 值 | 结果 |
|---|---|---|
| mean(Δoverall) ≤ −5e−4 | **+0.004531955606998261** | 否 |
| mean(Δs2d) ≤ 0 | **+0.027515469645682022** | 否 |
| scan122 Δoverall ≤ +5e−4 | **+0.011076474360725841** | 否 |

**verdict：科学 FAIL**（实验有效完成，不是 BLOCKED/INVALID）。

解读（不作过线辩护）：三场景 d2s 均下降（更好），但 s2d 均上升，拖累 overall；仅 scan105 overall 略降。keep 比例不是几何收益证据。

## 配对表（seed=0）

见 `paired_results_20261117_154810.csv` / `verdict_20261117_154810.json`。

| scan | control overall | trust overall | Δoverall | Δs2d | Δd2s |
|---|---|---|---|---|---|
| 24 | 0.35539816541829106 | 0.36111783774276435 | +0.0057196723244732905 | +0.019299073495948227 | −0.00785972884700159 |
| 105 | 0.47070824561806857 | 0.4675079657538642 | −0.0032002798642043473 | +0.024561614794965436 | −0.03096217452337413 |
| 122 | 0.3263132189111918 | 0.33738969327191765 | +0.011076474360725841 | +0.038685720646132404 | −0.016532771924680667 |

全部 `eval_seed: 0`。scan24 控制复用既有 seed0 评测（JSON sha=`52fe52f5…`）；旧未设 seed JSON 未改、未入主表。

## 产物路径（根：`.../ggs_trust_rebuild_20261116/`）

| 场景 | control 网格 | trust 网格 | 红图目录 |
|---|---|---|---|
| 24 | `scan24/control_trunc4/recon_post.ply` | `scan24/trust_trunc4_20261117_154810/` | `scan24/trust_preview_redmask_20261117_154810/` |
| 105 | `scan105/control_trunc4_20261117_154810/` | `scan105/trust_trunc4_20261117_154810/` | `scan105/trust_preview_redmask_20261117_154810/` |
| 122 | `scan122/control_trunc4_20261117_154810/` | `scan122/trust_trunc4_20261117_154810/` | `scan122/trust_preview_redmask_20261117_154810/` |

评测目录：`*_eval_seed0_20261117_154810/`（含 `vis/results.json`、`eval_receipt.json`）。  
每场景 trust 另有 `per_view_metrics.json`。

## 未测边界（保留）

- 单正深度用例 ≠ CUDA 恰一有效候选证明  
- gt_mask 边界未单独穷尽  
- CUDA 全无效注入未单独强制  

## 停点

已停止。请审查最终 verdict；不自动扩场景、不重训、不改论文结论。
