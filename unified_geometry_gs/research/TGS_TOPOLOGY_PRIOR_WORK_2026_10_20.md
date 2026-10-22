# 固定容量细分：定向先行工作审查

2026-10-20。沿用literature-review流程；这是单人定向审查，不是系统综述或新颖性证明。
只用原始论文/作者站支持技术判断；第三方摘要和自动批评仅作线索，未采用其结论。
没有把未公开推导、具体公式或本地模型上传给检索服务。

## 本轮新增的贡献边界

[Rivara (1984), Algorithms for refining triangular grids suitable for adaptive and multigrid techniques](https://doi.org/10.1002/nme.1620200412)
已经提出基于最长边二分的自适应网格算法。该条只取得出版社索引/摘要，全文未取到，不声称核读其证明。

[Stamm, Eidenbenz, Pajarola (1998), A Modified Longest Side Bisection Triangulation](https://www.ifi.uzh.ch/vmml/publications/older-puclications/CCCG98.pdf)
已核读§2–4：最长边二分与连续两轮细分、角度质量和改进规则均有成熟研究。
作者PDF已归档`sources/tgs_topology_review_v1/`。本轮world_longest4只实现普通最长边连续两轮，
没有声称复现该论文的完整修改算法、邻接传播或复杂度结果。

[Elastic Triangle Splatting §6.2](https://arxiv.org/html/2608.29106v1)明确在网格重建中选面积较大三角形，
沿最长边和邻接三角形共同细分以维持连接；其NVS部分仍沿用TGS/MCMC增密。
因此“把最长边细分用于triangle splatting”也不是本项目的首创。§7.1已有小基元过滤消融，
本轮不能把发现剔除现象本身当作创新。

[MeshSplatting §3.5](https://arxiv.org/html/2512.06818v1)使用世界中点四分，在共享顶点中转移颜色和opacity。
其连接结构/顶点参数化应在未来方法比较中考虑；不能将本轮独立三角形TGS操作直接称为该方法的复现。

[CF-Splat v1 (2026-10-15), §III-A–F、IV-A](https://arxiv.org/html/2610.22108v1)在已知网格上建立
多视角SH线性最小二乘和矩阵自由求解，预先根据图像梯度向三角形内添加顶点形成三子。
这限制了未来“固定几何SH求解”和图像引导细分的一般新颖性主张。本轮只审方法相关段，
没有复现其性能、数据或公开代码，不将它当作当前TGS四子协议的等价实现。

本项目投影内切半径锐界的特定表述，尚未在这些已读段落中确认完全相同的结论。
这不是排除经典网格理论/投影失真文献中的先例，也不足以认定其可发表新颖性。
普通有限目录搜索和maximin选择同样是通用技术；本轮试验失败，不能包装成新的有效方法。

## 可复核检索和来源状态

工具：网页检索，直接打开arXiv原始HTML、出版社条目和大学作者PDF。通用查询如下：

- triangle splatting densification anisotropic subdivision longest edge
- projective triangle midpoint refinement inradius shape regularity mesh
- Rivara longest edge bisection triangle refinement 1984 1995 minimum angle
- MeshSplatting triangle split densification four children subdivision
- "triangle splatting" "anisotropic" subdivision
- "MeshSplatting" "subdivision"
- "triangle splatting" "longest"
- "inradius" "perspective" triangle subdivision refinement

多词perspective检索还命中了欧氏几何的透视三角形/cevian问题，不能把这些直接当针孔相机先行工作。
Larsson2012各向异性网格策略正文访问失败，General triangular midpoint subdivision2014未阅读全文，
均不用于本轮技术排除判断。没有声称完整覆盖所有检索记录或使用PRISMA计数。

本地下载：1998作者PDF成功，三个arXiv HTML在Windows urllib因证书链验证失败未归档；
网页工具已成功核读对应原文。保留失败收据，不关闭TLS验证或把失败文件记为全文存档。
详见`sources/tgs_topology_review_v1/manifest.json`。
文献方法沿用已有`TGS_RELATED_WORK_UPDATE_2026_10_16.md`所述流程。
