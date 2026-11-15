# 给接收 agent 的部署手册

2026-11-15。目标是恢复代码、资产与可验证环境；部署不等于科研目标完成，不自动运行拟合、优化器或训练。首先阅读 [HANDOFF.md](../HANDOFF.md)、[AGENT_START.md](AGENT_START.md)、[环境历史](REPRODUCTION_AND_GITHUB.md)。

## 可直接交给 agent 的任务

> 克隆私有仓库 Someone-hates-Monday/geometry-splatting-research，先检查用户是否已有本地修改、是否已有训练进程，不覆盖已有目录。使用 handoff/deploy_assets.py 校验和部署资产，等待 Release 最终清单时明确报告状态。先完成 stdlib 资产恢复，再在 Linux 或 WSL2 的独立环境中安装经目标 GPU 验证的 PyTorch 和依赖，重新编译本机扩展；不得直接使用原机器的 .so。保留旧 manifest 和哈希，另外记录新机器的路径映射、环境与构建回执。最终报告资产完成度、7 个缺失链接、Torch/CUDA 可用性及尚未通过的检查，不宣称完整训练成功，不启动新训练。

## 1. 代码与认证

接收者先取得私有仓库访问权限，并在自己的机器上安装 Git、GitHub CLI、Python3.10+。GPU运行建议使用 Linux／WSL2；只下载、核验和恢复文件也支持 Windows。每位组员使用自己的 GitHub 凭据，不能共享上传者令牌。

```bash
gh auth login
git clone https://github.com/Someone-hates-Monday/geometry-splatting-research.git
cd geometry-splatting-research
python handoff/deploy_assets.py --check-env
```

`--check-env` 只记录 Python、平台及工具路径，**不证明 GPU 可用**，不导入模型或执行 checkpoint。回执默认保存于 `~/geometry-splatting-assets/deployment_receipt.json`。建议预留约350 GB空间，容纳压缩卷、约126 GB原始内容及临时／拼接文件；实际占用由最终 manifest 和文件系统决定。

## 2. 自动下载与恢复

在项目根运行（Windows 与 Linux 均可，替换 `python` 为已验证解释器）：

```bash
python handoff/deploy_assets.py --download --restore --check-env
```

根目录默认根据脚本位置确定，缓存默认在个人主目录。更换缓存或目标目录：

```bash
python handoff/deploy_assets.py --cache /data/splat-cache --root /work/geometry-splatting-research --download --restore
```

Release 尚未完成时，默认报错并停止；不会拿不完整分卷伪装部署成功。若希望无人值守等上传结束，显式增加 `--wait`：

```bash
python handoff/deploy_assets.py --download --restore --check-env --wait
```

该选项每60秒重新尝试最终清单；认证失败或仓库地址错误也会持续等待，先单独确认 `gh auth status` 和仓库访问正常。它不会启动训练。任务应在终端持久会话或受控后台进程中执行，日志需要保存。

工具会校验每卷大小／SHA256、ZIP成员集合、逐文件／逐段SHA256，拒绝路径穿越和链接目标，将分段文件按连续偏移自动拼接，并生成本次部署回执。已正确恢复的文件可重复核验；已有内容不同则停止，绝不静默覆盖。下载缓存损坏时，先检查网络、把坏卷移到别处，再重试。脚本不删除旧证据、不自动清理分段／缓存；确认成功后由维护者按回执决定清理。

离线环境先由有权限的机器下载清单和全部分卷，再复制到接收机：

```bash
python handoff/deploy_assets.py --cache /data/splat-cache --manifest /data/splat-cache/ASSET_MANIFEST.json --restore
```

数据公开来源及自己下载的备用路线见 [DATA_SOURCES.md](DATA_SOURCES.md)。公开数据重下后仍需核对相机、分辨率、划分和哈希；不能代替项目生成的历史模型与中间状态。

## 3. 运行环境由接收机器单独准备

项目旧 `environment_uggs.yml` 指向 Torch2.5／CUDA12.1，不能作为所有显卡通用自动部署方案。历史成功环境使用 Python3.12、Torch2.12 nightly+cu128、CUDA12.8；旧 nightly 是否仍可获取尚未验证。RTX5090 需要合适的架构支持，不能仅看 `nvidia-smi` 显示正常就判定成功。

agent 应先记录 GPU／驱动／nvcc／gcc，再按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 为接收硬件选择兼容的 wheel。稳定版与历史 nightly 的环境漂移必须写入新回执。使用全新、项目专用环境，例如：

```bash
conda create -n geometry-splatting python=3.12 pip -y
conda activate geometry-splatting
# 按目标机器从官方选择器获得 torch/torchvision 安装命令，在此执行。
python -m pip install -r unified_geometry_gs/requirements.txt
python -m pip install sympy numpy scipy matplotlib pyyaml ninja configargparse
```

源码扩展需要本机编译。确认当前 Python 属于新环境，`CUDA_HOME` 指向与所选 Torch 构建兼容的 toolkit，CC／CXX 为可用编译器；GPU架构按实际设备设置，不能给所有机器固定12.0。

```bash
cd unified_geometry_gs
bash scripts/install_submodules.sh "$CUDA_HOME"
cd ..
```

该旧脚本对CUDA11／12有明确版本检查，对不同toolkit应由agent审查并在新部署记录中适配；它的可选 fused-ssim 安装失败不会导致整个脚本失败，所以要单独核对实际已安装模块，不能只看 exit0。TGS研究探针另有源码和构建协议，并不由上述G-GS安装脚本自动全部部署。保存每一步日志和 `python -m pip freeze`，不要把安装脚本完成写成全部实验环境通过。

运行以下轻量检查，日志写到接收机器的部署记录目录，不修改旧封存实验：

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list()); print(torch.ones(1, device='cuda').cpu())"
```

这只验证小张量和环境，不能证明 renderer／backward／完整训练已经工作。目标探针的固定输入渲染验收须在新协议、独立输出目录进行。不要把旧绝对路径批量替换进原 manifest；以新路径映射连接旧相对路径和新工作区。

## 4. 接收验收必须报告

- 代码 commit、Release tag、manifest哈希、已验证卷数和恢复条目数，是否全部成功。
- 原始7个不可读取路径仍是已知缺口；工具不会猜测目标或伪造内容。
- 目标硬件、驱动、Python、Torch／CUDA、编译器、架构与扩展构建日志。
- 能做到的是只读资产恢复、环境检查，还是已通过指定固定输入渲染。分别列出失败和未执行项。
- 科研继续位置是TGS v29负结果后的提案，没有已实现v30，也没有部署后自动开始训练的授权。

## 已做的工具验证与限制

本工具使用小型离线构造测试覆盖正常／重复恢复、大文件分段拼接、坏卷、错误成员哈希、路径穿越、已有文件冲突及非连续分段。它不是在另一台GPU机器上完成的整套部署验证。当前全量Release仍在传输，真实90卷全量恢复验收必须等最终清单可用后在接收方执行。
