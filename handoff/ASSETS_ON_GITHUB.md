# 数据与模型的 GitHub Release

2026-11-15 用户授权补传数据、点云、模型和先前未上传的研究资产。

下载入口：[research-assets-2026-11-15](https://github.com/Someone-hates-Monday/geometry-splatting-research/releases/tag/research-assets-2026-11-15)。仓库仍为私有，组员需要仓库访问权限。

本轮计划覆盖 17813 个文件、125745456224 字节，分为 90 个 ZIP 卷。包括 DTU、评估数据、output_compare、research 的 results／experiments／sources／snapshots，保留仓库相对路径。编译缓存、认证配置与无关工具未纳入。7 个不可访问链接列在计划中，不声称完成无损全盘备份。

**传输中不能把计划视为已上传证明。** Release 中 `plan.json` 是传输计划；最终 `ASSET_MANIFEST.json` 才列出全部已传卷的大小、SHA256、下载地址和逐文件／分段映射。只有这个最终清单存在且列出的90卷均可下载，才表示本轮完成。原始资产保持不变，生成的本地 ZIP 在远端大小和 SHA256 匹配后删除以节省临时空间。

通过已登录的 GitHub CLI 下载（约126 GB原始数据，先准备足够磁盘）：

```text
gh release download research-assets-2026-11-15 --repo Someone-hates-Monday/geometry-splatting-research --dir research-assets
```

先按最终 manifest 核验每卷 SHA256，再解压至空目录。超过分卷预算的原文件以 `.asset-part-数字偏移` 拆分，须按数值偏移顺序连接，逐段核验后去掉分段后缀；不能直接把这些分段当 checkpoint 或 ZIP 使用。不要直接覆盖已有研究证据。

`upload_research_assets.py` 可在原机器继续中断的上传；中间状态在被 Git 忽略的 `.asset-transfer/`。它校验源文件大小／修改时间，保留卷回执；上传重试不会覆盖原始模型。未上传完成时，可根据本地回执和 GitHub 实际附件判断进度。
