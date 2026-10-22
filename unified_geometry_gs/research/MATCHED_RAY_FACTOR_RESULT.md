# 匹配原基元后的射线因素检查

2026-10-13。`matched_ray_factors_v1` complete、exit0；同24个原查询均保留。
本结果是冻结CPU模型的机制检查，不是GT质量、生产CUDA或新方法证据。

24相机的所有投影数组、24射线alpha/peak/width/init与旧CPU参考bitwise相等。
全部192个固定集合混合根、全部筛选路径中间根均有限；未用fallback补样本。
最大旧affine根差9.28035e−10，最终world camera根与上一轮保存结果完全相同。
三因素的全部6种加入顺序平均分配差值（标准Shapley式，非原创或因果识别），
再与固定筛选路径有符号增量相加；最大总和误差4.33681e−19。

各项绝对增量/旧深度的24点中位数（百分数）：

- peak：.00209194%；width：.0000389273%；peak amplitude：.000826298%。
- 取消affine power/alpha eligibility：.0115894%。
- 取消像素tile membership：.000350566%。
- prefix、projection-valid/image-extent、near-plane及有限起点的中位项更小；
  完整各项min/median/max与全部24行保存，不能据中位量推断所有尾部都小。
- selected集合world opacity cap这一项在这24例无影响；不推断其他射线也无影响。

总增量存在相消：1−|sum增量|/sum|增量|的中位数.291519，最大.870742。
最大尾部scan37/cam0041/pixel(601,131)：peak分配+.204732767，
width+.000044245，amplitude+.003647897，取消eligibility −.041186829，
其余项合计后net+.164084657个世界坐标单位。不能把net全归因某一项。

`results/matched_ray_factors_report_v1.json`：69实际源/archive、50产物、3PLY、
24 NPZ membership/全部行复核；原log_fields独立重解192个固定集合根，
最大差2.68230e−13；分配和路径算术通过。
大集合选择阶段根没有独立再次重解，不能扩大独立验证范围。
集合membership的完整packed mask、原模型ID、affine/world selected系数均保存。

结论：典型射线的筛选影响比width差更突出，特定尾部的peak变化值得进一步检查。
但当前是已观测旧结果后的探索性、固定选择24射线，不能当场景普遍性或因果归因。
由此启发的affine packet透视重参数另见 `PERSPECTIVE_REPARAM_RESULT.md`。
排列分配参考[DU-Shapley Eq4](https://proceedings.neurips.cc/paper_files/paper/2024/file/03cd3cf3f74d4f9ce5958de269960884-Paper-Conference.pdf)。
