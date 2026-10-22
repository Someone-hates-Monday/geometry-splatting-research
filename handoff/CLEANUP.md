# 遗留模型清理记录

2026-10-22。人类明确要求适当删除不必要的历史模型与点云；本轮依据现有用途，采取有界清理。返回 [交接入口](../HANDOFF.md)。

## 实际执行

删除 `unified_geometry_gs/output_compare/ggs_tau_{0.3,0.5,1.2}/scan105/` 内从 `chkpnt10000.pth` 到 `chkpnt29000.pth`、间隔 1000 步的 **60 个中间优化器检查点**。删除文件逻辑大小合计 **15,622,688,680 字节（14.550 GiB）**；这是文件大小合计，没有用磁盘剩余空间变化来测量实际释放量。

依据：三个扫描均已有 `chkpnt30000.pth`、最终 iteration_30000 点云与 `vis/results.json`；`judge_g1.py` 从最终评估 JSON 判定 τ 曲线，P3／P4 脚本产生中间检查点但没有读取它们完成该评估。早期 `scripts/cleanup_outputs.sh` 原本提出更宽泛裁剪，本轮没有运行该脚本。

清理前，对 6036 个文本文件检索候选完整后缀路径／SHA256，无直接引用命中、无扫描错误；这是字面引用审计，不证明外部程序或动态拼接从未依赖这些文件。未将“同样大小”当作内容重复，也未批量删除某一扩展名。

## 保留与代价

- 保留三组 `chkpnt30000.pth`，以及 iteration_7000／30000 点云、输入点云、最终网格、配置、相机、multi_view 和评估记录。每组 11 个重要文件，合计 **33 个**，删除前后 SHA256 全部一致。
- 保留原始数据、公共 TGS 初始模型、研究共同起点、完整训练最终对照、封存实验与失败回执、来源代码、原快照。其他研究目录的模型与点云没有在本轮删除。
- **失去这 60 个历史中间步精确恢复训练／查看中间优化器状态的能力。** 没有为它们另做恢复副本。保留的哈希只能识别原文件，不能恢复内容；重训也未必逐位相同。依赖完整历史目录的旧检查可能因此不再通过，应附本记录解释，不能改旧 manifest 掩盖缺失。
- 旧点云是否重复仍有待依赖和内容核验，当前没有足够依据继续删。清理不是把研究资产压成可直接上传 GitHub 的源码包；大文件分离按[移交说明](REPRODUCTION_AND_GITHUB.md)处理。

## 审计入口

- [cleanup_plan.json](cleanup_plan.json)：精确 60 个目标及大小、SHA256，33 个保护文件，引用审计范围与局限。
- [cleanup_deletions.jsonl](cleanup_deletions.jsonl)：每个实际删除的路径、哈希、大小与 UTC 时间。
- [cleanup_result.json](cleanup_result.json)：执行完成、失败为 null、删除数量与保护文件前后核验结果。
- [清理前元数据清单](workspace_inventory.before_cleanup.json)与[清理前大文件 CSV](large_file_candidates.before_cleanup.csv)。
- [清理后元数据清单](workspace_inventory.json)与[当前大文件 CSV](large_file_candidates.csv)。文档／回执继续整理会产生少量新文件，因此整体目录差不能严格等同被删除大小。
- [计划生成器](plan_legacy_cleanup.py)与[执行脚本](execute_legacy_cleanup.ps1)：先解析绝对路径并限制目录、拒绝 reparse 目标／祖先，逐文件 SHA256 匹配后使用 PowerShell `Remove-Item -LiteralPath` 删除，没有递归删目录。已有审计后拒绝覆盖，未提供批量新目标入口。

如再扩大清理，应先检查来源、共同起点、最终对照及 manifest 依赖，确认替代副本确实可用，再作独立审计并记录范围。不要直接重跑旧的大范围清理脚本。
