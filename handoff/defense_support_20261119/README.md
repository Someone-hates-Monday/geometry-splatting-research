# 结项答辩补充：乙的 11 场景口径与丙的相机／出图回执

日期：2026-11-19。本包整理历史评测证据、复现出图和最后一次强掩码实验。十一场景主表沿用历史评测，强掩码实验使用下文单独的三场景配对结果。

## 乙：十一场景主表

Baseline 均取 `unified_geometry_gs/output_compare/ggs_baseline/scanXX/vis/results.json`。Pipeline\* 的 **scan24、105、122** 是复用的试点 `ggs_ex_trunc4`；其他场景是 `ggs_pipeline`。这两个候选目录共同构成旧 Pipeline\* 主表，不能把三个试点误写为 `ggs_pipeline`。来源依据：[P5 脚本](../../unified_geometry_gs/scripts/run_research_p5.sh)、[判定脚本](../../unified_geometry_gs/scripts/judge_p5.py)与[证据审计](../evidence_receipts/unified_geometry_gs/research/results/evidence_audit.json)。

下表所有路径均以 `unified_geometry_gs/output_compare/` 为前缀。点击路径可打开本包的**原始字节副本**；[source_index.json](source_index.json)记录完整 Release 路径、SHA256、数值和本包文件路径。粗体的“其余八组”是 11 月 17 日三张图以外的八个成对场景。

| 场景 | Baseline 原路径／副本 | Pipeline\* 原路径／副本 | Baseline overall | Pipeline overall | Δ (mm) |
|---|---|---|---:|---:|---:|
| **其余八组** scan24 | [ggs_baseline/scan24/vis/results.json](results/ggs_baseline/scan24/results.json) | [ggs_ex_trunc4/scan24/vis/results.json](results/ggs_ex_trunc4/scan24/results.json) | 0.3747108106 | 0.3549663922 | -0.0197444183 |
| **其余八组** scan37 | [ggs_baseline/scan37/vis/results.json](results/ggs_baseline/scan37/results.json) | [ggs_pipeline/scan37/vis/results.json](results/ggs_pipeline/scan37/results.json) | 0.4974867462 | 0.4606038041 | -0.0368829421 |
| **其余八组** scan40 | [ggs_baseline/scan40/vis/results.json](results/ggs_baseline/scan40/results.json) | [ggs_pipeline/scan40/vis/results.json](results/ggs_pipeline/scan40/results.json) | 0.2678304472 | 0.2558869347 | -0.0119435125 |
| scan55 | [ggs_baseline/scan55/vis/results.json](results/ggs_baseline/scan55/results.json) | [ggs_pipeline/scan55/vis/results.json](results/ggs_pipeline/scan55/results.json) | 0.3102629971 | 0.3138563023 | +0.0035933051 |
| scan63 | [ggs_baseline/scan63/vis/results.json](results/ggs_baseline/scan63/results.json) | [ggs_pipeline/scan63/vis/results.json](results/ggs_pipeline/scan63/results.json) | 0.8054308179 | 0.7368805734 | -0.0685502445 |
| **其余八组** scan65 | [ggs_baseline/scan65/vis/results.json](results/ggs_baseline/scan65/results.json) | [ggs_pipeline/scan65/vis/results.json](results/ggs_pipeline/scan65/results.json) | 0.4345847071 | 0.4335832319 | -0.0010014752 |
| scan69 | —（缺记录） | [ggs_pipeline/scan69/vis/results.json](results/ggs_pipeline/scan69/results.json) | — | 0.4297548910 | — |
| scan83 | [ggs_baseline/scan83/vis/results.json](results/ggs_baseline/scan83/results.json) | [ggs_pipeline/scan83/vis/results.json](results/ggs_pipeline/scan83/results.json) | 1.0420576934 | 0.9572175846 | -0.0848401088 |
| **其余八组** scan97 | [ggs_baseline/scan97/vis/results.json](results/ggs_baseline/scan97/results.json) | [ggs_pipeline/scan97/vis/results.json](results/ggs_pipeline/scan97/results.json) | 0.6416662599 | 0.6139492274 | -0.0277170325 |
| **其余八组** scan105 | [ggs_baseline/scan105/vis/results.json](results/ggs_baseline/scan105/results.json) | [ggs_ex_trunc4/scan105/vis/results.json](results/ggs_ex_trunc4/scan105/results.json) | 0.5163422423 | 0.4704883494 | -0.0458538929 |
| scan106 | —（缺记录） | [ggs_pipeline/scan106/vis/results.json](results/ggs_pipeline/scan106/results.json) | — | 0.3166543790 | — |
| **其余八组** scan110 | [ggs_baseline/scan110/vis/results.json](results/ggs_baseline/scan110/results.json) | [ggs_pipeline/scan110/vis/results.json](results/ggs_pipeline/scan110/results.json) | 0.5729140180 | 0.5471360841 | -0.0257779340 |
| scan114 | —（缺记录） | [ggs_pipeline/scan114/vis/results.json](results/ggs_pipeline/scan114/results.json) | — | 0.3090563626 | — |
| scan118 | —（缺记录） | [ggs_pipeline/scan118/vis/results.json](results/ggs_pipeline/scan118/results.json) | — | 0.3145915283 | — |
| **其余八组** scan122 | [ggs_baseline/scan122/vis/results.json](results/ggs_baseline/scan122/results.json) | [ggs_ex_trunc4/scan122/vis/results.json](results/ggs_ex_trunc4/scan122/results.json) | 0.3307786400 | 0.3242774839 | -0.0065011561 |

**可比的 11 组：10 改善、1 退化；按未舍入原值计算，平均 Δoverall = -0.0295654011 mm。** scan69、106、114、118 只有 Pipeline\*，缺 Baseline；这四组不能参与成对均值或称为 15 场景严格对照。

### 评价口径与边界

- 评价入口是 [`evaluate_dtu_mesh.py`](../../unified_geometry_gs/evaluate_dtu_mesh.py)，先按相机对齐网格，再调用 [`dtu_eval/eval.py`](../../unified_geometry_gs/dtu_eval/eval.py) 的 `--mode mesh`。实现的默认参数：`downsample_density=0.2`、`patch_size=60`、`max_dist=20`、`visualize_threshold=10`，GT 是 DTU 官方点云、观测掩码和地面平面。
- `mean_d2s` 是重建网格采样点到 GT 的最近邻距离均值，`mean_s2d` 是 GT 到重建点的最近邻距离均值；`overall=(mean_d2s+mean_s2d)/2`，单位为 **mm**，越低越好。各场景等权平均的是未舍入的 `Pipeline\* overall − Baseline overall`。
- **一致性的准确说法：两侧采用相同的 DTU 网格评价口径和默认设置，但不能证明历史运行使用了完全相同的随机采样。** 旧 `results.json` 只记录三个指标，未保存逐次 `eval_seed` 或完整命令；评测实现默认 `default_rng()` 无显式 seed，点采样可能不同。也没有逐次运行回执可证明历史版本与所有参数逐项一致。上述数据属于历史描述性对照，不能宣称严格同种子复测。

## 丙：scan83、63、55 的两侧相机与出图

六份原始 `cameras.json` 均在本包 `cameras/` 下；同一场景的 Baseline 与 Pipeline **完整文件 SHA256 相同**，不仅是选中的 `0000` 条目相同。逐份原始路径与 SHA256 见 [source_index.json](source_index.json)。

| 场景 | 左侧 Baseline 相机 | 右侧 Pipeline 相机 |
|---|---|---|
| scan83 | [unified_geometry_gs/output_compare/ggs_baseline/scan83/cameras.json](cameras/ggs_baseline/scan83/cameras.json) | [unified_geometry_gs/output_compare/ggs_pipeline/scan83/cameras.json](cameras/ggs_pipeline/scan83/cameras.json) |
| scan63 | [unified_geometry_gs/output_compare/ggs_baseline/scan63/cameras.json](cameras/ggs_baseline/scan63/cameras.json) | [unified_geometry_gs/output_compare/ggs_pipeline/scan63/cameras.json](cameras/ggs_pipeline/scan63/cameras.json) |
| scan55 | [unified_geometry_gs/output_compare/ggs_baseline/scan55/cameras.json](cameras/ggs_baseline/scan55/cameras.json) | [unified_geometry_gs/output_compare/ggs_pipeline/scan55/cameras.json](cameras/ggs_pipeline/scan55/cameras.json) |

原图及完整相机／渲染内外参在[三张对照图包](../scan_comparison_20261117/README.md)和[原元数据](../scan_comparison_20261117/metadata.json)。使用每组相机 `img_name=0000`，左右各 1000×748 px、相同材料和白色背景，焦距按分辨率缩放后再除以 1.45 以统一扩大视野。渲染环境：**Open3D 0.20.0**，Filament Vulkan 后端，软件设备 llvmpipe。

11 月 17 日原始出图是在 D 盘 WSL 的 `/opt/gsenv` 中，对每个场景执行如下命令（`SCAN` 依次为 83、63、55）：

```bash
source /opt/gsenv/bin/activate
python /mnt/c/Users/28943/Documents/Codex/2026-10-22/github-geometry-splatting-research/work/render_compare.py --scan SCAN --width 1000 --out /mnt/c/Users/28943/Desktop/GGS_Pipeline_同视角对照_scan83_63_55/scanSCAN_same_view.png
```

原始脚本后来整理为仓库中的 [`generate_comparisons.py`](../scan_comparison_20261117/generate_comparisons.py)，命令行参数改为 `--scans`。原始终端 stdout 未封存；2026-11-19 使用仓库脚本按下列**实际复现命令**重渲染，输出保存在[复现运行日志](render_reproduction.txt)。三张复现图与已交付 PNG **逐像素相同**：

```bash
source /opt/gsenv/bin/activate
python /mnt/c/Users/28943/Documents/Codex/2026-10-22/github-geometry-splatting-research/work/repo/handoff/scan_comparison_20261117/generate_comparisons.py --root /opt/geometry-compare-assets --outdir /opt/geometry-compare-repro-20261119 --scans 83 63 55 --width 1000
```

这次复现只重渲染现有网格，没有重新训练或重新评价指标。

## 最后一次强掩码实验

训练时，多视角可信带会筛选监督信号。我们想检验，把同样的可信判据用于 TSDF 融合，能否过滤不可靠深度、进一步改善网格。我们在 scan24、105、122 上分别冻结同一份重建模型做配对。两组沿用相同的 trunc4 提取设置和 DTU 评测（seed=0）。控制组按原流程融合，实验组每个视角固定取排序首位近邻，仅保留几何检查通过、warp 有效且 NCC 误差小于 0.9 的深度，其余置零。

三场景的重建到真值误差（d2s）均下降，但真值到重建误差（s2d）均上升，说明局部精度的改善伴随覆盖度指标退化。三条预设门槛均未达到，尤其是 scan122 的整体误差上升。因此，我们保留原融合流程，在这一轮结束后停止继续调整强掩码。

Δ 为实验组减控制组，单位为 mm，均值按三个场景等权计算。

| 预设门槛 | 实测值（mm） | 通过 |
| --- | ---: | :---: |
| 平均 Δoverall ≤ −0.000500 | +0.004532 | 否 |
| 平均 Δs2d ≤ 0 | +0.027515 | 否 |
| scan122 Δoverall ≤ +0.000500 | +0.011076 | 否 |

[实验封存包](https://github.com/Someone-hates-Monday/geometry-splatting-research/releases/tag/jia-trust-rebuild-2026-11-17)
