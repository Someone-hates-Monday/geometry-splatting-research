# 执行顺序偏离记录（105/122 提前训练）

2026-11-17。按审查要求记录；**不丢弃模型、不重训、不按质量挑选**。

## 协议要求

`JIA_TRUST_REBUILD_PROTOCOL_20261116.md`：先只训 scan24 → control → 冒烟 → **停审**；未经审查不启 105/122。

## 实际发生

组织者授权「24 之后也重训其他需要场景」后，实现侧用排队脚本在 scan24 训练结束后自动启动 105→122，**未等 scan24 冒烟审查通过**。

| 场景 | 脚本 | 日志 | 结果 |
|---|---|---|---|
| scan24 | `scripts/run_jia_rebuild_scan24_train.sh` | `.../scan24/logs/train.log` | 完成；`Training complete.`（约 2026-11-17 00:28） |
| scan105→122 | `scripts/run_jia_rebuild_queue_105_122.sh`（及 resume） | `.../queue_105_122.log` | `queue complete`（约 2026-11-16 17:23 UTC / 本地 01:23） |

冻结模型哈希（`iteration_30000/point_cloud.ply`）：

| 场景 | sha256 |
|---|---|
| scan24 | `e772631d0800f5b49ac32e3ecc5f7d608d733e9d5483851d2e2ea0831303625c` |
| scan105 | `75281dabb4b8327224b672f01a500013405c10e03ecafcda684f359a419acf0f` |
| scan122 | `bd7239447fb43af4dbe50f910e426f59cb06b46628362226be2921576da0958d` |

## 原因（执行侧）

把「三场景都要补训」理解成可在 24 训完后立刻排队，把审查停点窄化为「不提 trust」，省略了「不启 105/122」的停点约束。

## 处置

- 保留上述三份模型与哈希，后续实验继续使用。  
- 不根据中间质量换模型。  
- 本偏离不构成放行 trust 全量的理由；trust 仍待补审通过。
