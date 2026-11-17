# 甲新成果的代码与资产交付

2026-11-17，用户授权提交本轮新增内容。科研状态仍为完成验证、三条门槛未过。

源码和文档提交包括本轮提取/评测/PatchMatch修改、显式seed评测源码、辅助脚本及全部相关handoff记录。只对 `dtu_eval/eval.py` 开例外，不纳入整个真值目录；个人 `.vscode` 排除。

51份精选小证据、封存及源码副本随Git保存，见 [JIA_COMMITTED_RECEIPTS_20261117.json](JIA_COMMITTED_RECEIPTS_20261117.json)。这些副本路径与哈希连接本地原证据，不能代替大模型和网格。

本轮整个 `unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116/` 另发 [jia-trust-rebuild-2026-11-17 Release](https://github.com/Someone-hates-Monday/geometry-splatting-research/releases/tag/jia-trust-rebuild-2026-11-17)，保留相对路径。原90卷Release不修改。

Release只有出现最终 `ASSET_MANIFEST.json` 且所有列出卷可核验时才算全部传完。下载和恢复可用：

```text
python handoff/deploy_assets.py --tag jia-trust-rebuild-2026-11-17 --cache /data/jia-assets --download --restore
```

目标已有文件与归档不同会拒绝覆盖。恢复原封存后，状态文档漂移与混合路径问题以 `seal_delivery_amendment_20261117/DELIVERY_MANIFEST.json` 为补充，不修改原封存。大资产与本仓库均为公开可见，未包含个人认证配置。
