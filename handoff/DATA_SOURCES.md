# 原始数据与公开模型来源

2026-11-15 补充。当前决定仍是继续全部90卷传输，下面作为组员自行下载的备用路线；公开原始数据不能替代本项目生成的共同起点、最终模型、网格、实验输出或封存回执。

## DTU 训练数据

本项目脚本 `unified_geometry_gs/scripts/download_dtu.sh` 使用 Hugging Face 数据集 `dylanebert/2DGS`：

- [数据集入口](https://huggingface.co/datasets/dylanebert/2DGS)
- 本地当前历史数据位置：`dtu/DTU/`。
- 脚本默认下载位置：`unified_geometry_gs/data/dtu/`，不会自动恢复历史绝对路径；接手时明确设置 `UGGS_DTU_ROOT`。

在有效 Python 环境安装 `huggingface_hub` 后下载：

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="dylanebert/2DGS", repo_type="dataset",
                  local_dir="unified_geometry_gs/data/dtu")
```

这是仓库中已记录的下载方式，本轮没有重新下载整套数据；该 Hugging Face 页面在本轮网页工具中访问失败，不保证当前镜像可用或内容与旧镜像逐位一致。旧备用脚本 `download_dtu_gdrive.sh` 记录的 [Google Drive 文件夹](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) 同样需接收者自行核对可访问性和目录结构。

## DTU 评估真值

[DTU 官方 MVS 数据页面](https://roboimagedata.compute.dtu.dk/?page_id=36) 提供 SampleSet（含标定、可见性掩码、评估代码等）和 Points（全部场景结构光参考点云）入口。本项目本地位置为 `unified_geometry_gs/dtu_eval/Offical_DTU_Dataset/`，`Offical` 是历史路径原拼写，不要擅自改名。

训练图像包不等于评估真值。根据实际评估脚本的文件需求整理 ObsMask、Points 等目录，并用本项目最终资产清单或旧实验输入哈希比对；仅下载官方最新版不证明与历史评估字节相同。数据引用按官方页面给出的 CVPR 2014／IJCV 2016 论文办理。

## Mip-NeRF 360 原图与相机

[作者官方页面](https://jonbarron.info/mipnerf360/) 提供 Dataset Pt.1／Pt.2。本项目 v29 输入计划记录的 flowers／room 原始图像来源为：

- [360_extra_scenes.zip](https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip)
- v29 原输入计划：`unified_geometry_gs/research/experiments/tgs_joint_policy_v29/input_plan.json`，小副本也在 `handoff/evidence_receipts/` 同名相对路径。

历史脚本通过 HTTP Range 提取特定成员，不代表本机保存了完整数据集。恢复时保留相机名、图像尺寸、原 split 和字节身份核验，不重新挑选测试视角。公开 TGS 渲染图也不能自动当作相机身份已对齐的原始 GT。

## TGS 公开预训练 checkpoint

[Triangle Splatting 官方仓库的模型与测试渲染入口](https://github.com/trianglesplatting/triangle-splatting#trained-models-and-rendered-test-views-on-mipnerf360) 链接到作者发布的模型。历史使用了 stump 及 flowers／room 的 iteration7000；flowers7000 为938101个三角形，room7000为2100000个三角形。不要以 room30000 或自行重训版本替代。

公开来源及选择依据详见 `TGS_CROSS_SCENE_RESULT_2026_10_20.md`、对应 input note 和 source manifest；下载后要按这些记录验证成员路径和 SHA256。公开初始 checkpoint 与我们训练得到的 checkpoint 在科学归属上不同，不能混称。

## 来源代码与历史版本

原 GGGS 链接 [HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting](https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting) 本次访问重定向到 `Gaussian-Splatting-with-Stochastic-Solids`。这只说明上游当前入口发生变化，不证明当前 main 等同历史实验源码。复现先使用本项目保留的源码和哈希，不直接升级上游替换已封存实现。

## 哪些必须从我们的 Release 获取

`output_compare/` 的训练结果与提取点云／网格、research 实验原始输出、共同起点与最终对照、诊断 patch／张量、失败回执和源码快照属于项目证据，应从 [数据与模型 Release](https://github.com/Someone-hates-Monday/geometry-splatting-research/releases/tag/research-assets-2026-11-15) 获取。使用公开原始数据重新训练不能恢复这些历史状态。

范围、分卷恢复和传输完成判定见 [ASSETS_ON_GITHUB.md](ASSETS_ON_GITHUB.md)。最后 `ASSET_MANIFEST.json` 缺失时仍在传输或尚未成功收尾，不能宣称已完整备份。
