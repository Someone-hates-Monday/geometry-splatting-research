# 与 SuGaR / 其他方法公平对比时要注意什么

## 要不要比？

**建议比**，但要有清晰边界，否则容易被审稿人或自己误导。

- **有价值**：同一 DTU scan、同一评估脚本（你们已有 `evaluate_dtu_mesh.py` + 官方数据）、同一 mesh 评测定义。
- **成本高**：SuGaR、PGSR、其他 SOTA 往往是**另一套训练代码与默认超参**；若输入预处理、迭代数、mesh 提取流程不一致，数值差异可能来自**管线**而非方法本身。

结论：**可以做对外对比，但必须写清「公平性假设」和「不可控差异」。**

---

## 公平对比最低标准

1. **数据**：同一 `scanXXX` 目录（COLMAP/图像一致）。
2. **评测**：同一 `dtu_eval/Offical_DTU_Dataset`，同一 `recon_post.ply` 评测入口（或各自官方评测脚本但同一 GT）。
3. **报告**：写清各方法的 **训练迭代数 / 是否用 depth-normal / mesh 提取命令**，避免“我们 30k iter，对方官方默认 7k”这类隐性不公。
4. **多 scan**：至少 3～5 个，最好 8+，报平均与最差 scan。

---

## SuGaR（父仓库）怎么对齐

- SuGaR 走 **sugar_trainers** 管线，与 `unified_geometry_gs` 的 G-GS **不是同一训练循环**。
- 可行做法：
  1. 在 **同一 DTU scan** 上分别训 SuGaR 与 UGGS；
  2. 各自按官方流程导出用于 DTU 评测的 mesh（或统一转为 `recon_post.ply` 若路径一致）；
  3. 用**同一 Python 评测脚本**对两个 mesh 跑 Chamfer（或各自仓库脚本但核对公式一致）。

若 SuGaR 输出路径与 UGGS 不同，建议写一个薄封装：仅统一调用 `evaluate_dtu_mesh.py` 的 `--DTU` 与模型路径。

---

## “最先进”开源方法

常见对比对象（举例，按需选）：

- **PGSR / 2DGS / GOF** 等：各仓库默认 DTU 协议可能略有不同（图像分辨率、mask、迭代数）。  
  **务必**记录：官方 README 的默认命令 + 你们是否改成与 UGGS 相同 `-r` / 视角划分。

不要声称“全面 SOTA 表格”除非你真的复现了对方默认设置。

---

## 推荐汇报策略（论文/报告）

1. **主表**：UGGS baseline vs 你们的 M1/M1+（同一管线内消融）——这是最硬的因果证据。  
2. **副表**：选 1～2 个外部方法 + SuGaR（若资源够），标注 **训练预算与提取流程差异**。  
3. **补充材料**：单 scan 对比全部放附录；正文强调 **平均值 + 方差 + 最差 scan**。

---

## 仓库内一键多方法对比（同一 baseline）

```bash
cd unified_geometry_gs
export PYTHON=python

./scripts/compare_scan_methods.sh scan105 ggs_baseline \
  "M1-3:m1_ablate_03_full" \
  "M1-router:m1_router"
```

生成 `output_compare/metrics/scan105_baseline_vs_<tag>.json`。

外部方法只要把输出目录链到 `output_compare/<name>/<scan>` 或改脚本路径即可。
