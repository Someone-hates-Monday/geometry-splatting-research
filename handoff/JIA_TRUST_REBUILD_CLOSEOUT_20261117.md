# 甲可信带融合：收尾封存

2026-11-17。最终审查接受 **VALID_SCIENTIFIC_FAIL**。甲任务状态：**完成验证，未过线**。整个科研目标未完成。

## 答辩可用负结果说明（一段）

在重新构建并冻结的 scan24、105、122 模型上，我们将固定最近候选近邻、\(\tau=0.9\) 的强可信掩码用于 trunc4 融合。相对同模型无掩码对照（评测 seed=0），三场景等权平均 overall 增加 0.004532 mm，平均 s2d 增加 0.027515 mm，未达到预设三条门槛，因此不采用此改动。分项上看，三场景 d2s 均改善、s2d 均变差，与更严格拒绝深度带来的准确性/覆盖度取舍相一致；本轮未单独证明缺面位置，不能把“挖空”说成已证实的唯一因果。本结果不外推为 8 mm 截断全局最优，也不否定全部掩码方案，更不使既有 11 场景 Pipeline\* 对照失效。

## 视角与保留率（材料修正）

| 场景 | 训练视角数 | 全视角加权保留率（按正深度 B） |
|---|---:|---:|
| scan24 | 49 | ≈86.73% |
| scan105 | 64 | ≈90.26% |
| scan122 | 64 | ≈91.05% |

单张 scan24 预览 keep≈74.8% **不是** 全场景保留率，也不是几何收益。

## 封存位置

| 项 | 路径 |
|---|---|
| 封存清单 | `unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116/seal_20261117_154810/SEAL_MANIFEST.json` |
| 清单索引 | `handoff/JIA_TRUST_REBUILD_SEAL_INDEX_20261117.json` |
| 最终审查 | `handoff/JIA_TRUST_REBUILD_FINAL_REVIEW_20261117.md` |
| 审计 JSON | `handoff/JIA_TRUST_REBUILD_FINAL_AUDIT_20261117.json` |
| verdict | `.../verdict_20261117_154810.json` |
| CSV | `.../paired_results_20261117_154810.csv` |
| 源码快照（含 gitignore 的 eval.py） | `.../source_identity_20261117_154810/` |

大模型/网格仍在 `ggs_trust_rebuild_20261116/<scan>/`；**未**改原 90 卷 Release，**未**自动上传。另建资产包需组织者另行授权。

## 明确未做 / 不做

- 局部截断第二刀  
- 调 τ / 换近邻 / 扩场景 / 重训  
- GPU 复跑  
- 向丙、丁自动发消息（由组织者通知：按原收尾做 11 场景表与 83/63/55 展示）

## 未测限制（保留）

CUDA 恰单有效点、gt_mask 边界穷尽、全无效强制注入——见 seal manifest `limitations`，不为包装隐去。
