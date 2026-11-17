# G-GS Baseline / 既有 Pipeline 同视角对照

左：原始 G-GS（`ggs_baseline`）；右：Pipeline（既有优化配方，`ggs_pipeline`）。Pipeline 为去掉 depth-normal、采用既定可信几何监督、提取 voxel=0.002 + trunc4 的历史结果；这里不包含新强掩码结果。

| 场景 | 图像 | Baseline overall (mm) | Pipeline overall (mm) | Δ (mm) | 图注 |
|---|---|---:|---:|---:|---|
| scan83 | [同视角图](scan83_same_view.png) | 1.0420576934 | 0.9572175846 | -0.0848401088 | 改善 |
| scan63 | [同视角图](scan63_same_view.png) | 0.8054308179 | 0.7368805734 | -0.0685502445 | 改善 |
| scan55 | [同视角图](scan55_same_view.png) | 0.3102629971 | 0.3138563023 | +0.0035933051 | 退化案例 |

## 分场景来源

### scan83

scan83：相同相机与渲染条件下，Pipeline 相对 G-GS Baseline 的历史 overall 变化为 -0.0848401088 mm（改善）。

- 左侧网格：`unified_geometry_gs/output_compare/ggs_baseline/scan83/recon_post.ply`；SHA256 `a7e662f75dc6323ee44768dd4b9ec46b2a05a7d655d3f857d26b01e80bec8169`
- 对应指标：`unified_geometry_gs/output_compare/ggs_baseline/scan83/vis/results.json`
- 右侧网格：`unified_geometry_gs/output_compare/ggs_pipeline/scan83/recon_post.ply`；SHA256 `289237b937ecf5a3b8e04ef89a776e8e30158d3a9f18ddf655bd94aa54509719`
- 对应指标：`unified_geometry_gs/output_compare/ggs_pipeline/scan83/vis/results.json`
- 相机源：`unified_geometry_gs/output_compare/ggs_baseline/scan83/cameras.json`（两侧相机 0000 参数逐项相同）

### scan63

scan63：相同相机与渲染条件下，Pipeline 相对 G-GS Baseline 的历史 overall 变化为 -0.0685502445 mm（改善）。

- 左侧网格：`unified_geometry_gs/output_compare/ggs_baseline/scan63/recon_post.ply`；SHA256 `a972e98bf5884c21f06a32c2066ebc6325e07acb735fdc5b2b734b502d13ecde`
- 对应指标：`unified_geometry_gs/output_compare/ggs_baseline/scan63/vis/results.json`
- 右侧网格：`unified_geometry_gs/output_compare/ggs_pipeline/scan63/recon_post.ply`；SHA256 `c5479b425e991f8a9907278fdc6bc40aa2caa413677aaa047733bbf99ba5c592`
- 对应指标：`unified_geometry_gs/output_compare/ggs_pipeline/scan63/vis/results.json`
- 相机源：`unified_geometry_gs/output_compare/ggs_baseline/scan63/cameras.json`（两侧相机 0000 参数逐项相同）

### scan55

scan55：相同相机与渲染条件下，Pipeline 相对 G-GS Baseline 的历史 overall 变化为 +0.0035933051 mm（退化案例）。

- 左侧网格：`unified_geometry_gs/output_compare/ggs_baseline/scan55/recon_post.ply`；SHA256 `1e011704d7214db1731c5360ec58b936c545a44582acad41ee5f03a082a01170`
- 对应指标：`unified_geometry_gs/output_compare/ggs_baseline/scan55/vis/results.json`
- 右侧网格：`unified_geometry_gs/output_compare/ggs_pipeline/scan55/recon_post.ply`；SHA256 `5e3904cdfc41a77b7ee2612d33611295633f81cc2d7dc0cd96a488e97fdb3a5b`
- 对应指标：`unified_geometry_gs/output_compare/ggs_pipeline/scan55/vis/results.json`
- 相机源：`unified_geometry_gs/output_compare/ggs_baseline/scan55/cameras.json`（两侧相机 0000 参数逐项相同）

## 渲染与来源

- 三组均使用 `cameras.json` 的 `img_name=0000`。左右面板各 1000×748 px，原始 mesh 坐标及尺度不变，没有裁剪；视野统一扩大 1.45 倍。
- 相机旋转/位置为 camera-to-world，渲染用其逆矩阵。`cameras.json` 未存储主点，按画面中心 `(500, 374)` 设定。
- 两侧均使用 Open3D `defaultLit`、白色基色、网格顶点颜色、计算得到的顶点法线，以及相同白色背景与渲染分辨率。
- `metadata.json` 包含每组完整相机条目、渲染焦距、源输入照片及点云 SHA256、网格 SHA256、指标 JSON 路径和历史指标。
- 来源：GitHub 仓库 `Someone-hates-Monday/geometry-splatting-research` 的 Release `research-assets-2026-11-15`；资产路径以发布清单 `ASSET_MANIFEST.json` 为准。
- 这些图是网格可视化，历史 overall 为 DTU 网格评测结果；scan55 数值略有退化。

## 补充：复现脚本与原始指标

- `generate_comparisons.py`：复现三张同视角对照图。需安装 Open3D、NumPy、Pillow，并先按 Release 清单恢复原始网格和相机文件。运行 `python handoff/scan_comparison_20261117/generate_comparisons.py --root <恢复的资产根目录> --outdir <新输出目录>`。请把 `--outdir` 指向新目录，保留本交接中的原始图像不变。
- `results/`：六份从历史资产原样复制的 `vis/results.json`，按 `scanXX_方法_results.json` 命名。原始路径及 SHA256 见 `metadata.json` 的 `files.*.metric_json`。
