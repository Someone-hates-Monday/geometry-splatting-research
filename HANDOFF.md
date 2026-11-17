# 科研项目交接入口

> **最新交付（2026-11-17）**：甲的三场景重建配对实验已完成验证、三条门槛未过并封存，不进入第二刀。先读[本轮代码与资产交付](handoff/JIA_UPLOAD_20261117.md)、[最终审查](handoff/JIA_TRUST_REBUILD_FINAL_REVIEW_20261117.md)、[封存补充索引](handoff/JIA_SEAL_DELIVERY_AMENDMENT_20261117.json)和[汇报骨架](handoff/REPORT_OUTLINE_AND_ACCEPTANCE_20261117.md)。下文10月交接与TGS状态按历史快照理解，结项后续以11月分工与审查为准。

交接日期：2026-10-22。面向接手组员及其 agent。**科研总目标尚未达成；本轮只做现有工作的收尾、证据整理、存储清理与 GitHub 交接，没有启动新训练或新研究实验。**

## 目标与当前边界

在 3DGS、Geometry-Grounded Gaussian Splatting（GGGS / G-GS）、2DGS 和 Triangle Gaussian Splatting（TGS，三角形基元）基础上，争取有学术价值的数学补充、创新方法或几何／图像／效率改进。以原始论文核查、可证伪假设、可复现对照和消融支持结论，不能把合成验证、局部计时或未验证设想写成真实场景上的学术成果。

用户最初要求核实成熟的 3D、数学、编程和科研工具，审查 GGGS 消融及失败探索，并完成实现与实证验证。现有资产包括公式审计、诊断实现、真实 checkpoint 上的固定输入核验，以及多条负结果；**尚未建立可投稿的新颖性、稳定的留出质量收益或完整训练加速。**

最后研究授权为数学核验与小规模固定输入 CUDA 诊断，不包含新拟合／优化器步骤／完整训练。随后用户要求最小收尾与交接，并适当清理遗留模型。本交接不扩张研究执行范围。接收者应先读记录，再与人类确定下一阶段预算。

## 接手阅读顺序

需要恢复资产或配置新机器时，先看 [agent部署手册](handoff/DEPLOYMENT_FOR_AGENTS.md)；自动接收入口为 `handoff/deploy_assets.py`，默认不启动训练。

1. [研究脉络与结果边界](handoff/RESEARCH_MAP.md)：原 GGGS、数学与渲染诊断、TGS 的主要证据和失败原因。
2. [给组员和 agent 的启动说明](handoff/AGENT_START.md)：当前停止位置、约束、下一项尚未实现的工作。
3. [环境、复现与 GitHub 移交](handoff/REPRODUCTION_AND_GITHUB.md)：代码与大文件分离、历史环境及迁移限制。
4. [清理记录](handoff/CLEANUP.md)：删掉什么、保留什么、失去哪些历史恢复能力。
5. [完整记录索引](handoff/RECORDS_INDEX.md)及[机器可读索引](handoff/research_records.json)：929 份顶层代码／研究笔记／子项目文档，以及 539 份选定的 manifest、plan、结果 JSON。索引不是逐项重新验证证明。
6. [三人分工与结项推进计划](handoff/TEAM_PLAN_2026_10_23.md)：面向 12 月中旬结项的岗位分工、首周交付、SuGaR 展示与 M1 对照范围，以及论文／专利争取安排。具体执行规模仍按资产、算力与导师确认结果确定。

## 到目前为止最重要的结果

- **GGGS 历史结果需要限定口径。** DTU 15 个扫描中只有 11 个成对基线；10 改善、1 退化，平均 overall 差约 -0.02957。缺失的 4 个基线、训练视角指标和不完整配置限制结论，不能包装为完整 15 场景严格对照。
- **实际训练探索未达到预设质量目标。** scan55 factor 的几何改善约 0.602%，未达 1% 门槛；另一次留出对照 TEST PSNR +0.0901 dB、SSIM 略降，未过门槛。history-gap 完整训练几何退化约 0.264%，耗时增加约 31.98%。后者已经正常结束，旧“正在训练”记录不能当作当前状态。
- **TGS 固定查询的数值和效率验证有积累，但泛化失败。** v26 热缓存 8 候选查询 A/D 比约 6.20–21.47，属于特定固定前向查询口径。v29 等预算联合细分主规则在 flowers、room 两场景均改善校准分数、退化留出分数；没有证明训练提速或最终质量收益。
- **数学补充有明确适用条件。** 固定编辑预算下平均 opacity 项可正确分解；双视角独立三阶 SH 每通道至少 14 维不可观测方向。前者不等于完整训练目标改善，后者不是实际图像 Jacobian 秩或新颖性证明。最近文献复核发现 CF-Splat 已有三角加密，必须收窄创新叙述。

最新封存阶段是 [v29 结果](unified_geometry_gs/research/TGS_JOINT_POLICY_RESULT_2026_10_21.md)。下一份 [外观适应提案](unified_geometry_gs/research/TGS_BOUNDED_APPEARANCE_NEXT_PLAN_V29.md) **只有设计，没有 v30 实现、协议或拟合结果**。优先准备颜色响应系数和固定正负 DC 扰动的原生参考核验；通过后才讨论有界拟合。

## 目录与证据位置

2026-11-15 补充：[数据与模型 Release 移交](handoff/ASSETS_ON_GITHUB.md)。先前仅在本地的大资产正在分卷补传；具体完成状态以 Release 的最终 `ASSET_MANIFEST.json` 和附件核验为准。

仓库根目录是原 SuGaR 项目；`unified_geometry_gs/` 是基于 G-GS 的独立子项目，生产入口包括 `train.py`、`gaussian_renderer/`、`scene/`、`losses/`。`unified_geometry_gs/research/` 保存独立探针、协议、结果、理论笔记与封存实验。早期 README 中的“最新”仅指当时快照。

小型交接证据放在 [evidence_receipts](handoff/evidence_receipts/)，逐文件原路径、复制路径、大小和 SHA256 见 [copied_receipts.json](handoff/copied_receipts.json)。包括近期 manifest／计时／失败回执、历史质量审计和 GGGS 指标；原始图像、模型、二进制、全部原始张量仍需另行移交。**仅拿到 GitHub 代码和这些回执，不足以重算全部历史数字。**

GitHub 私有仓库：[Someone-hates-Monday/geometry-splatting-research](https://github.com/Someone-hates-Monday/geometry-splatting-research)，分支 `main`。提交范围与接收步骤见 [GIT_READY.md](handoff/GIT_READY.md)。原项目、G-GS 及其他第三方许可保留原样，不能把整个研究目录统一声明为 MIT。请按 [移交说明](handoff/REPRODUCTION_AND_GITHUB.md) 分开传输代码和研究资产。
