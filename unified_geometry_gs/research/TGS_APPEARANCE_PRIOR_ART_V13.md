# 外观迁移方向的一手文献边界

2026-10-21。本次为有目标的叙述性核读，采用一个 agent，不是系统综述或新颖性认证。
问题、纳排与数据边界先写于 `TGS_APPEARANCE_TRANSPORT_SCOPE_V13.md`。完整工具返回、
精确查询、获取失败和版本信息保存在 `literature/appearance_transport_search_v13*.json`。
没有上传本地模型、代码、未发表公式或实验结果到外部检索/AI 服务。

## 方法与获取限制

检索以通用 web 发现、arXiv 原文、CVF/ECVA/NeurIPS 会议页和作者项目页交叉核对。
补充 Hugging Face paper_search 实际返回 UNAVAILABLE，故没有将其算作成功的独立数据库。
Scaffold-GS 的 CVF PDF 返回403，ECVA Revising PDF 超大小；分别取得 arXiv v1 HTML
核读。对标题/摘要页、选定方法/附录段落和完整报告可访问性分别记录，不声称已经逐页读完
每篇论文或完整筛查所有命中。快速索引可能滞后，不报告全领域搜索覆盖率或无先例结论。

## 已确认的直接重叠

**外观与几何中心分开并非足够的新意。** [Scaffold-GS](https://arxiv.org/html/2312.00109v1)
§3.2.1–3.2.2 使用相机相对 anchor 的方向和距离，而高斯中心是 anchor 加学习偏移；颜色等
属性从 anchor 特征解码。因此“改用父/独立锚点的观察方向”这一宽泛提案明显与已有方法重叠。
其外观由 MLP 产生，与保留显式 SH 的代数迁移不是同一实现；差异本身尚不足以成立创新。
身份由作者 [CVPR 2024 项目页](https://city-super.github.io/scaffold-gs/) 和 arXiv 记录核对。

**近似保持表示的密化也已有直接先例。** [MCMC-3DGS](https://arxiv.org/html/2404.09591v3)
§3.4、附录 A 通过重定位及 opacity/scale 调整，近似保留原高斯的渲染作用；不是简单复制。
它说明“密化前后更稳定”不是可直接主张的新目标。本项目若研究外观中心，需明确在它已经
处理的覆盖/合成误差之外还能提供什么。不能把该文的近似保持误写为普遍精确保持。

**理论密化已有较强工作。** [SteepGS](https://arxiv.org/html/2505.05587v1)
§4 与附录 C 给出加权子函数混合的 splitting matrix 分析。附录的损失先把子函数加权合为
父槽位，再进入后续函数；若直接将各子基元逐项 alpha 合成，需要另查零阶与排序条件。
这是目标模型的边界核对，不是对论文定理的反证。不能把标准泰勒展开或均值抵消重新当作
原创数学。作者 [项目页](https://vita-group.github.io/SteepGS/) 可核对论文身份与理论概述。

**opacity 和抽样收益不能混入外观归因。** [Revising Densification](https://arxiv.org/html/2404.06109v1)
§3.2–3.4 已研究像素误差引导、克隆 opacity 修正及增长预算。保持 clone 中心位置时，
opacity 修正与 SH 方向迁移是不同问题。更改它们后图像改善不能全部归因于 SH 运输。
[Decomposing Densification](https://arxiv.org/html/2507.20239v1) 的方法部分另区分 split/clone
行为并联合多分辨率与 pruning；这类联合变化要求分量控制，不能仅以总指标解释机制。

## 必须纳入的近期反证与竞争基线

[Compact Neural Appearance Models](https://arxiv.org/html/2609.05255v1)，2026-09-04 预印本，
§5–6 与附录 A 核读：在统一框架比较多种外观参数化，讨论颜色激活、几何恢复与采集瞬变
之间的混淆。它提醒我们更好的训练图像不必然代表更好的几何或物理外观；其紧凑共享网络
也使“共享 MLP 更省显存”不能作为本项目独有贡献。其报告收益不是本地复现结果。

数学方面，[NIST DLMF 18.12.11](https://dlmf.nist.gov/18.12.E11) 与
[14.30.8–9](https://dlmf.nist.gov/14.30) 已提供 Legendre 生成函数、SH 正交性与加法定理。
新 `TGS_SH_SHELL_THEORY_V13.md` 是把这些标准工具用于本项目具体外观迁移问题的补充证明，
不是宣称首次发现 SH 平移/正交性。检索还遇到经典固体调和函数平移文献；其径向因子与
仅依赖归一化方向的辐射场不同，尚需专门核读后才能作等价/不等价或新颖性结论。

Scitepress 143097 的发现摘要实际属于风机叶片重建（BladeR）；原 PDF 获取403，作者机构
元数据确认主题为高分辨率分块重建，故本次不作为外观运输数学的支持证据。摘要不足以
排除其所有细节存在关系，后续若需要引用仍须获取原文。普通博客、论坛与自动论文摘要不
作为技术结论依据。

## 对当前选题的约束

目前能具体推进的是带假设与误差的显式 SH reanchoring/运输问题：完整球面上的有限阶
不可精确表示性、具体谱泄漏、有限阶投影和系数相关上界。它比“加锚点”可证伪，但其
学术新颖性、实际效用与成本优势仍待核验。固定半径球面也不是实际多距离相机集。
本轮安排真实公开 SH 系数的 CPU 场函数诊断，用于检验纯 degree1 例子的代表性；不把
它伪装成真实捕获视图、图像质量或训练实验。

研究工作流使用 `literature-review`、`hypothesis-generation`、`sympy` skills。
流程来源：Kassis, Timothy; Agarwal, Vinayak; He, Yuhuan; Patel, Darshil; Brueckner, Aubrey M.
(2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*,
[arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。已在线核对作者、标题及当前
记录（v2；2026-09-02修订），没有使用带版本的 DOI，也没有从流程工具可用性推断科学有效性。
