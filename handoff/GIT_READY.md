# Git 提交与仓库信息

仓库名：**`geometry-splatting-research`**。

建议 GitHub 简介：`Research code, mathematical audits, and reproducible diagnostics for geometry-aware Gaussian and triangle splatting.`

2026-10-22 在当前项目根初始化本地仓库，分支 `main`。按用户提供的信息设置仅本仓库的提交作者 `someonehatesmonday <2872397866@qq.com>`。仓库名称不声称已经获得新的 splatting 方法或论文成果；最新结果与下一步见 [HANDOFF.md](../HANDOFF.md)。

## 提交范围

纳入原 SuGaR 与独立 G-GS 子项目代码、现有第三方源码和许可证、研究顶层笔记／协议／诊断实现／LaTeX、展示材料，以及完整交接小证据包。研究索引的 929 份代码／笔记／文档和 286 份小证据副本应全部随提交保留。

根 `.gitignore` 已补入大文件资产排除规则，并显式保留 `handoff/evidence_receipts/`，避免旧 `output*` 规则吞掉其中的历史指标。原始数据、训练模型、研究 `experiments/`／`sources/`／`snapshots/`／`results/`、编译产物、个人环境配置和与科研无关的嵌套工具仓库未纳入。它们仍在本地，另按 [移交说明](REPRODUCTION_AND_GITHUB.md) 传输。

根和子项目 `.gitattributes` 保留现有文件字节，避免 Git 的自动换行转换破坏代码与回执哈希。新 shell 脚本仍须保存为 LF；已有封存文件不批量转换换行。

## 验收记录

[git_submission_audit.json](git_submission_audit.json) 记录候选／暂存文件的路径、字节数与 SHA256，检查 50 MiB 以上文件、资产误纳入、交接索引遗漏、证据副本哈希，以及常见凭据模式。报告自身无法包含自己的内容哈希，明确从该哈希清单中排除。此检查不等于完整的保密或许可证审计。

可用有效 Python 运行：

```text
python handoff/audit_git_submission.py --staged
```

该命令会更新审计 JSON；更新后需重新暂存报告。已提交的版本可用 `git status --short` 和 `git log -1 --oneline` 确认。本次准备不包含新研究实验、CUDA 编译、拟合或训练。

## GitHub 仓库

2026-10-22 根据用户授权，在 `Someone-hates-Monday` 账号创建私有仓库：[geometry-splatting-research](https://github.com/Someone-hates-Monday/geometry-splatting-research)。本地 `origin` 使用 HTTPS，上传 `main`；作者信息与上文一致。组员需由仓库所有者授予访问权限，本轮未向其他人发送消息或邀请。

接收组员可在获得访问权限后克隆：

```text
git clone https://github.com/Someone-hates-Monday/geometry-splatting-research.git
cd geometry-splatting-research
```

仓库按组内交接用途先设为私有，没有公开发布研究资产。完整外部资产包及其获取地址仍待维护者安排，代码包无法独立重跑全部历史数字。历史准备记录中的“尚未上传”描述按当时状态理解；当前远端以此仓库为准。
