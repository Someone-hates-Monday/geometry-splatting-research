# TGS 细分选题：相关工作补充与贡献边界

检索与原文核读开始于 2026-10-16；中断后于 2026-10-18 整理落盘。这里记录的是单人定向核读，不是系统综述或截至整理日的完整文献覆盖。没有运行论文代码、复现论文表格或向搜索服务提交未公开推导。

## 已核读的原文与限定

**Elastic Triangle Splatting — Shi, Qian, Cremers，arXiv:2608.29106v1，2026-08-29。**

[原文](https://arxiv.org/html/2608.29106v1)，重点 §3.1–3.2、§6.1–6.2、§7.1。
该工作比较单侧、双侧及可变边界值核。§6.1 区分统一距离参数化的形状重建实验与保留原实现的 NVS 对比；不能把其统一后的核当作原 TGS 的逐点等价实现。§7.1 已讨论细分后的小三角形剔除，并报告关闭小尺寸剔除对其 NVS 指标影响很小。上述是论文报告，未在本机复现。

对本项目的约束：边界核分析与关闭小尺寸剔除已经有直接先行工作；“发现细分可能产生缝隙”不足以构成新的贡献。真实事件诊断必须允许剔除影响很小这一竞争解释。

**Deformable Triangle Splatting: Flexible Primitives for Real-Time Radiance Field Rendering — Jiménez-Ayguadé, Agudo，arXiv:2607.22446v1，2026-07-24。**

[原文](https://arxiv.org/html/2607.22446v1)，重点 §3.1–3.2。该工作已明确讨论原 TGS 像空间距离比的视角依赖，以重心坐标描述可变形边界，并在细分时转移父边控制点。原文的投影后坐标映射与其视角一致性表述需要结合完整源码进一步核对；本次不把这些表述当作已独立证明的几何性质，也不据此认定其代码有错。

对本项目的约束：“像空间核依赖视角”及“采用重心坐标”不能被包装成我们的首创。当前草稿更窄的问题是固定子形状、固定 sigma 时，共享 opacity 的零/一阶矩约束是否可行；这与已有窗口设计有区别，但区别不等于新颖性成立。

**Revising Densification in Gaussian Splatting — Rota Bulò, Porzi, Kontschieder，arXiv:2404.06109v1，2024-04-09。**

[原文](https://arxiv.org/html/2404.06109v1)，重点 §3.3。它已给出克隆的中心 opacity 平方根修正，并明确说明全像素偏差无法因此完全消除。不能把本项目静态审查中的相同现象称为新发现。最初尝试的 v2 地址未返回正文；随后摘要页确认版本为 v1，并成功核读 v1。ECVA PDF 本次未成功取得，因此不声称完成会议版逐字比较。

## 有限检索记录

检索工具：网页搜索，随后直接打开 arXiv 原文及摘要元数据。通用查询：

- Gaussian splatting densification "function preserving"
- Gaussian splatting densification "rendering" "preserving" split opacity conservation
- triangle splatting subdivision opacity conservation refinement

本轮采用上述三份原文支持对应判断，未作全领域排除。搜索中的其他结果仅为线索；例如 DOI 10.1016/j.patrec.2026.09.034 未完成全文方法核对，不能据摘要宣布其等价或不等价于本项目推导。

原 TGS 会议最终版的 KAUST 存档地址在这次尝试返回 Internal Error；连同此前作者站点、Oxford、OpenReview 的访问缺口，会议版与当前使用的 arXiv v1 的差异仍未解决。没有绕过访问控制。

后续状态更新（2026-10-18）：通过作者公开托管地址成功归档 10 页会议相关稿，见 `sources/tgs_conference_2026/READING_NOTE.md`。第 3–5 页核心窗口和细分规则已核读、目视并与固定代码对照；不再把这些段落列为未读。尚未与 IEEE 出版社版本或全部补充材料全面比较。上段保留为此前的访问记录。

## 当前可提出的研究主张

本项目目前只有理想 TGS 核的解析草稿：总积分保持、一阶矩偏移、单视角有界 opacity 可行域、多视角共享参数障碍，以及同内参针孔相机的存在性推广。它们没有证明真实场景质量改进，也没有证明与先行工作相比有足够发表价值。

下一步应验证“不可行性是否对应真实优化中的持续负担”，而非重复证明已有的窗口或克隆现象。恢复尝试前，执行范围应限制在数学核验和冻结输入的渲染诊断；不因解析结果自动扩大为新训练项目。

方法流程沿用 literature-review skill，出处：[Kassis et al., Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。该工具方法引用不为本项目结论背书。
