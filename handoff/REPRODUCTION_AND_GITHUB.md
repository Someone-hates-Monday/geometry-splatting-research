# 环境、复现与 GitHub 移交

2026-10-22；返回 [交接入口](../HANDOFF.md)。初次交接时尚无根级 Git 历史；随后用户要求准备 Git 提交，本地仓库状态、名字和提交范围见 [GIT_READY.md](GIT_READY.md)。未创建远端或推送。`tools/guizang-ppt-skill/.git` 是与本研究提交无关的嵌套工具仓库，已排除，不能误当项目主历史。

## 最近成功环境与当前快照

以下 WSL 环境来自既有成功运行回执，本轮没有重装或重新验证整套环境：Ubuntu，Python 3.12.12，解释器 `/home/someone/miniconda3/envs/uggs/bin/python`；Torch `2.12.0.dev20260314+cu128`、CUDA 12.8、NumPy 2.4.2、SymPy 1.14.0、SciPy 1.17.1。扩展记录使用 `CUDA_HOME=/usr/local/cuda-12.8`、gcc11／g++11、`TORCH_CUDA_ARCH_LIST=12.0`、`MAX_JOBS=2`。这是一份历史成功配置，不保证今天任意机器上同名 wheel 仍可取得。

本轮只读 NVIDIA 查询见 RTX 5090 Laptop GPU、驱动 616.92、24463 MiB；当时 compute-apps 列表为空。Windows 查询也未列出 python／python3／wsl 进程，但未做完整 WSL `ps` 审计。**这只能说明采样时的状态，不能替代 CUDA 渲染、backward 或完整训练验证。** 最近 v21 控制接口为 forward-only；旧完整训练曾成功结束与新环境可训练是两个问题。

本轮交接脚本使用 Windows 自带的 Codex bundled Python（stdlib），不依赖 Torch；该 Python 与 WSL 训练环境不同。接收者用自己的有效 Python 运行脚本。旧 `docs/RTX5090_BLACKWELL.md` 关于 stable Torch 缺 sm120 的说法是早期记录，不能作为当前普遍结论；迁移以实际环境检查和成功回执为准。

## 路径与依赖迁移

原实验大量使用 `/mnt/c/Users/Someone/Desktop/SuGaR-main/...`、Windows 绝对路径、WSL 扩展 `.so` 路径。复制到其他目录不会自动复现。保留原 manifest，另写路径映射与新环境／源码哈希，重新编译本机扩展；不要修改旧回执，让它看起来在新路径原样通过。

需要独立移交的资产包括原始 DTU／TGS 数据及相机划分、公共初始 checkpoint、共同 6999／7000 起点、最终对照 checkpoint／点云／网格、TGS 使用的来源代码与版本、封存实验的 patch／原生图像／GT 身份数据、完整依赖清单。先按所选阶段 manifest 逐项解析依赖，再规划传输，不能只按文件扩展名挑选模型。

元数据扫描有 7 个 `point_cloud` 路径在 Windows 返回 OSError，列于 [workspace_inventory.json](workspace_inventory.json)。可能涉及 WSL 链接，原因未在本轮证实；必须在原环境解析并验证目标，不能把该扫描称为完整备份。大文件清单按逻辑大小统计，跳过部分缓存、`.git` 和链接，不代表物理磁盘用量。

## 将代码与大文件分开交付

GitHub 普通 Git 对超过 50 MiB 的文件给出提示，超过 100 MiB 的单文件会拒收；大文件需另选 Git LFS 或团队资产存储。这里远超普通源码仓库规模，不能把整个文件夹直接批量提交。[GitHub 官方大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

建议交付两部分：

1. **代码与小证据包**：根项目和独立子项目源代码、原许可证、研究顶层说明／协议／实现／LaTeX、`docs/`、`handoff/`。精选小 JSON、失败文本和图表已复制到 `handoff/evidence_receipts/`，见 [copied_receipts.json](copied_receipts.json)。这是可供审阅的证据包，不含全部原始数据。
2. **外部研究资产**：上述模型、数据、实验张量和快照按原相对路径打包或同步，附字节数与 SHA256；记录存储位置、访问权限和获取方法。当前尚未创建这样的完整外部包或填写下载位置。被本轮裁剪的 60 个中间优化器状态只能靠重训重建，没有恢复副本，见 [清理记录](CLEANUP.md)。

可参考初次交接的 [gitignore.recommended](gitignore.recommended)。后续 Git 准备已将适用的大资产排除规则合入根 `.gitignore`，另补小证据目录例外并审计实际提交清单，见 [GIT_READY.md](GIT_READY.md)。忽略模板不能替代最终逐文件审阅。

移交前由维护者执行这些具体步骤：

1. 确定团队 GitHub 仓库、可见性和负责人，明确哪些原始数据及第三方模型允许再分发。保留根项目、G-GS、TGS、子模块和其他第三方的原许可；本轮没有做完整许可证或内容级密钥审计，文件名扫描也不能证明没有秘密。
2. 为所需外部资产建立真实存储和哈希清单，处理 7 个不可访问路径。确认接收方可下载后再删除其他资产；目前不继续清理封存实验、公共源模型或大快照。
3. 合并适用的忽略规则，明确处理嵌套 Git 目录。只暂存经过审核的源文件与小证据，审阅暂存清单和大小；不要先把大文件提交后再删，因为历史仍会携带它们。
4. 用接收方机器做最小只读验收：验证小证据副本哈希、相对文档链接、所选阶段依赖完整性。若只是代码包，要在 README 明示缺少资产及获取办法；不要宣称可完整复现历史数字。
5. 人类确定目标远端与上传范围后，再提交和推送。本轮未执行这些发布操作。

## 交接工具与验收边界

在项目根用有效 Python 调用：

```text
python handoff/build_handoff_index.py verify
```

`verify` 仅检查交接 Markdown 的本地链接；`index` 重建记录索引；`inventory` 重做元数据清单；`receipts` 复制精选小回执并校验哈希。后面三项会更新交接文件，**不会训练、编译 CUDA 或验证论文结论**。交接快照本身需要长期保存时，请另建版本，避免覆盖本轮清理前后审计。

`plan_legacy_cleanup.py` 和 `execute_legacy_cleanup.ps1` 是本次一次性清理的可审计实现；已有回执后会拒绝覆盖，不要当作新机器上的默认启动命令。其清理范围与未来研究资产保留策略无关。
