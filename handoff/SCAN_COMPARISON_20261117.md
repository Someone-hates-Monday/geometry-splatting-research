# scan83 / scan63 / scan55 同视角网格对照交接

2026-11-17。此包用于组内汇报：比较**原始 G-GS 基线**（`ggs_baseline`）与**已有 Pipeline 最终配方**（`ggs_pipeline`）。Pipeline 指去掉 depth-normal、采用既定可信几何监督，并以 voxel=0.002、`trunc4` 提取网格的历史组合。它不是甲本轮的新强掩码结果；甲本轮的独立审查仍以 [最终审查](JIA_TRUST_REBUILD_FINAL_REVIEW_20261117.md) 为准。

| 场景 | 同视角图 | Baseline overall | Pipeline overall | Pipeline − Baseline | 判读 |
|---|---|---:|---:|---:|---|
| scan83 | [PNG](scan_comparison_20261117/scan83_same_view.png) | 1.0420576934 mm | 0.9572175846 mm | −0.0848401088 mm | 改善 |
| scan63 | [PNG](scan_comparison_20261117/scan63_same_view.png) | 0.8054308179 mm | 0.7368805734 mm | −0.0685502445 mm | 改善 |
| scan55 | [PNG](scan_comparison_20261117/scan55_same_view.png) | 0.3102629971 mm | 0.3138563023 mm | +0.0035933051 mm | **退化案例** |

每张图左侧为 Baseline，右侧为 Pipeline。以上数值直接来自六份历史 `vis/results.json`，文件原样保存在 [results](scan_comparison_20261117/results/)；它们是 DTU 网格评测指标，不是由渲染图重新计算的指标。三张图的视觉差异也不能单独证明几何质量提升。

## 输入与渲染契约

- 六个实际渲染网格均为各方法 `scanXX/recon_post.ply`。网格路径与 SHA256、两侧各自的指标 JSON 路径与 SHA256、源图像和输入点云哈希，见 [metadata.json](scan_comparison_20261117/metadata.json) 及 [包内 README](scan_comparison_20261117/README.md)。哈希已与 Release 资产清单逐项核对；这里的文件名按清单确定，不凭目录名推断模型身份。
- 每组两侧使用各自 `cameras.json` 的 `img_name=0000`；对应相机条目逐项相同。原始 `rotation` / `position` 为 camera-to-world，渲染外参取逆矩阵。完整原始相机参数、实际 world-to-camera 矩阵及渲染内参都在 `metadata.json`。
- 两侧各为 1000×748 px，保持原网格坐标尺度，不做裁剪或归一化。原始相机 `fx/fy` 按输出分辨率缩放后除以 1.45，统一扩大视野以展示完整物体。原相机 JSON 未给主点，因此按画面中心 `(500, 374)` 设定。
- 两侧均使用 Open3D `defaultLit`、白色基色、原网格顶点颜色、计算得到的顶点法线及同一白色背景。渲染脚本为 [generate_comparisons.py](scan_comparison_20261117/generate_comparisons.py)。

## 复现与资产边界

原始网格、相机和 DTU 图像来自私有 GitHub Release [`research-assets-2026-11-15`](https://github.com/Someone-hates-Monday/geometry-splatting-research/releases/tag/research-assets-2026-11-15)。仓库只提交三张 PNG、六份小型结果 JSON、哈希／相机元数据和生成脚本；大网格仍按 [资产移交说明](ASSETS_ON_GITHUB.md) 恢复。恢复后在安装 Open3D、NumPy、Pillow 的 Python 环境运行：

```text
python handoff/scan_comparison_20261117/generate_comparisons.py --root <恢复的资产根目录> --outdir <新的输出目录>
```

脚本默认生成全部三组，使用 `--scans 83` 等参数可只生成指定场景。原始结果 JSON 的复制文件名与路径映射见包内 README。scan55 应始终保留“退化案例”标识，不能换用甲的新掩码网格替代 Pipeline 右侧。
