# Triangle 原文/实现与薄层分位敏感性的定向审计

2026-10-13，Asia/Shanghai。仅原论文、作者项目/仓库和官方会议论文用于技术判断；第三方搜索结果只作发现线索。未训练、未运行新数值实验、未改旧审计、LaTeX或CONTINUATION。按收束指示停止扩检索；这是定向审计，不是系统综述或新颖性认证。

## TGS 身份和实际模型

用户所说三角形基元对应 **Triangle Splatting for Real-Time Radiance Field Rendering**。本次取得[arXiv:2505.19175v1原文](https://arxiv.org/html/2505.19175v1)与[官方实现](https://github.com/trianglesplatting/triangle-splatting)。项目/仓库当前标3DV2026；本次读到的arXiv HTML自标v1、2025-05-25，不把它与可能修订的最终版混写。

原文§3.1：每个基元是三顶点三角形，另有颜色、opacity和屏幕边缘锐度sigma。投影后，以三条有向边线性函数的最大值phi构造Eq1窗函数

    I(pixel) = ReLU(phi(pixel)/phi(incenter))^sigma.

它在三角形外/边界为0，内心为1；sigma→0在严格内部趋于实心指示窗。这是**屏幕footprint的锐度极限**，不是GGGS沿射线厚度epsilon→0。正文明确选择实际max而非LogSumExp，故不能泛称处处光滑。该公式和紧支撑三角形不能作为本项目的新设计。原文没有本次可检索到的median公式；实现的深度行为另以下列固定版本为准。[原文§3.1](https://arxiv.org/html/2505.19175v1#S3.SS1)

## 固定提交下的实现核验

只读GitHub API确认主仓库提交为`685ff270a8273afc3a2b094ae3148dd0b98e31e9`，其固定raster子模块为`6d61f4c0a571c635e37ffb740250706fa30a5541`，当时子模块main也为该提交。四份固定源码在内存读取、计算SHA256、提取所列行；未安装或编译。哈希见配套JSON。浏览器缓存main的行数较旧，以下链接与行号均采用固定提交的实际字节。

- [包装层 triangle_renderer/__init__.py](https://github.com/trianglesplatting/triangle-splatting/blob/685ff270a8273afc3a2b094ae3148dd0b98e31e9/triangle_renderer/__init__.py#L132)：132行读取median channel；136–138行将depth加权和除累计alpha并清理非有限值；148行用`pipe.depth_ratio`混合expected与median。只看到包装层并不能宣称实际训练一直使用median；默认参数与训练覆盖本轮未核验。
- [forward.cu](https://github.com/trianglesplatting/diff-triangle-rasterization/blob/6d61f4c0a571c635e37ffb740250706fa30a5541/cuda_rasterizer/forward.cu#L418)：418行把`p_view_triangle.z`保存为每基元深度，而不是在每个像素重新求一个GGGS连续根。所读main预处理正文由三角形中心构造该view点；因此当前读取到的辅助深度管线与一般透视ray-plane交点规则也须区分。
- 同一[forward.cu深度选择](https://github.com/trianglesplatting/diff-triangle-rasterization/blob/6d61f4c0a571c635e37ffb740250706fa30a5541/cuda_rasterizer/forward.cu#L588)：588–590行在更新当前层T前，若T>0.5就保存当前基元深度及contributor。它对应离散largest-visible规则；不足0.5累计opacity时保留最后一个被实际接受的基元，完全无贡献时保持初始化0。这里“实际接受”受阈值、裁剪和提前终止约束。
- 同一forward的141/145行使用0.01预处理opacity筛选；562行求幂窗，566行跳过alpha<1/255，569行对候选test_T<1e-4提前终止。浮点边界、几何退化与选择导数不能由理想薄层定理覆盖。
- [backward.cu](https://github.com/trianglesplatting/diff-triangle-rasterization/blob/6d61f4c0a571c635e37ffb740250706fa30a5541/cuda_rasterizer/backward.cu#L659)：659–660行仅把median通道上游加到已选contributor的深度上；285–287行另将预处理深度链式项按三顶点均分。这支持“固定选择分支的一层深度VJP”，不等同GGGS分配到所有连续衰减项的IFT梯度，也不代表整个三角形参数梯度只有该项。
- [rasterizer_impl.cu](https://github.com/trianglesplatting/diff-triangle-rasterization/blob/6d61f4c0a571c635e37ffb740250706fa30a5541/cuda_rasterizer/rasterizer_impl.cu#L115)：115行将预处理depth的位模式放入tile排序key。该实现审计没有覆盖完整排序正确性、退化三角形、近裁剪或所有CUDA反向项。

这补齐了“只能看到项目页”的旧审计缺口，但没有把旧文档改写为当时已经查完。薄GGGS与TGS目前最多可以比较**在冻结、同序层深度与同opacity下的离散选择规则**。它不能证明三角形footprint、sigma极限、顶点VJP、遮挡或完整渲染器等价。

## 分位敏感性中哪些已经是已有理论

1. [Implicit Reparameterization Gradients，Figurnov等2018](https://arxiv.org/html/1805.08498)，§3 Eq6和单变量CDF表达已有隐式导数`-partial_theta F / density`，§4包含mixture分布，附录B Eq11使用log-density表达。故“通过CDF/透射率隐式求导”及分母为密度/根斜率不是新贡献；改成log求值也不能消除真实小斜率导致的条件数。
2. [Peng、Fu、Glynn、Hu，WSC2017原论文](https://www.informs-sim.org/wsc17papers/includes/files/191.pdf)，§2 A0明确要求目标分位处密度连续且严格正，Eq1给同一分位敏感性比值。本文主问题是Monte Carlo敏感性估计的一致性/CLT，而非GGGS薄化；这些结论不能被引用为“每个epsilon的密度正就保证epsilon→0统一稳定”。本次已读范围没有提供当前三层族的统一斜率下界。
3. [Pesenti、Millossovich、Tsanakas，Differential Quantile-Based Sensitivity in Discontinuous Models](https://arxiv.org/html/2310.06151)，§2–4研究不连续/离散输入扰动下的VaR/ES敏感性；附录B.1以广义函数弱极限处理阶跃扰动。它表明“离散模型的敏感性”与“平滑近似逐点导数的极限”必须指定扰动含义，不能默认相同。所读模型不是当前Gaussian乘积透射率薄化族；没有核验到本项目完全相同的两重裕量定理。不能由此声称该定理已获新颖性证明。
4. [GGGS原文v2](https://arxiv.org/html/2601.17835v2) Eq10–13已经给连续中位根与隐式梯度；[2DGS原文](https://arxiv.org/html/2403.17888)附录B Eq19给离散largest-visible与低opacity约定。现有`THIN_QUANTILE_LIMIT.md`是对这些既有定义的奇异极限补充候选，不能把深度定义或一般逆函数微分记为新成果。

本轮还发现Hong2009、Liu/Hong2009和广义逆连续性的原始论文线索；因收束未完成其全文逐项核验，**没有将搜索摘要当作已读证明**。奇异扰动、generalized inverse、非正则quantile和vanishing-density文献仍有覆盖缺口；未找到相同表述不等于原创。

## 两种裕量和反例是否值得主攻

当前建议：**值得一次有界的机制验证，尚不值得认定为唯一论文主线。**

可保留的具体内容是：固定有限GGGS射线模型中，严格平台跨越使深度趋于选中峰位；额外排除内部峰值临界概率，使位置Jacobian集中到该层。三层例虽然每个有限厚度根正则，其位置权重恒为(1/2,0,1/2)，且opacity敏感性指数发散。平台等号例则区分深度极限本身。这些具体反例比泛称“IFT需要非零分母”信息更多，本次已读原文尚不能直接取代它们。

但现有证明的标准部分很多，孤立精确对称例可能不足以承担完整贡献。尤其现有公式只证明理想实数模型：极小opacity扰动、GPU舍入、root精度、导数floor、alpha/prefix截断都会改变可观察性。临界邻域是否比FP32参数间距更窄、真实训练射线是否受影响，以及两种裕量是否比直接根斜率更有预测价值，均未测量。不能从1.446e37的高精度构造导数宣称真实训练不稳定由它主导。

## 唯一有限验证建议：原始冻结射线的舍入尺度敏感性检查

建议另写协议，在已有三场景3072条预定射线的**原参数**上冻结选择/排序/alpha规则，不训练、不按失败筛样本。记录所有有效/无效输入、严格跨越层j、平台裕量、峰值裕量、根斜率和完整opacity Jacobian；同时区分理想连续模型与实际生产FP32/floor实现。以事先固定的每场景32条均匀索引（共96条，不按裕量挑选）做独立高精度root/VJP复核，以及对当前层opacity的相邻FP32可表示值扰动重解根，记录有限差分响应、GT可见性和生产梯度偏差。

这项检查只回答现有模型中的可观察机制，不直接证明图像改善。实际宽度、相机和模型均不缩放；如后来人为缩小宽度，必须另标为受控压力实验。建议预先限定主比较为“两种裕量是否在控制实际根斜率后仍解释额外的大偏差/非集中现象”；全部报告，禁止只展示最坏临界例。若未出现超出独立参考数值误差的大效应，或效应完全由原斜率条件数解释，则保留反例为理论适用性补充，不以此启动训练或宣称新的稳定性方法。若出现可重复的生产效应，再提出明确误差界/修正机制并做独立场景确认。

## 审计限制

本轮没有编译/执行Triangle官方renderer，没有验证完整CUDA导数、训练默认depth_ratio或最终会议版本与arXiv v1的差异；也未复跑本地薄层证明脚本。固定源码哈希及所核对行足以定位静态实现证据，不能代替运行时验证。两种裕量的新颖性及理论主线价值仍为待验证判断。
