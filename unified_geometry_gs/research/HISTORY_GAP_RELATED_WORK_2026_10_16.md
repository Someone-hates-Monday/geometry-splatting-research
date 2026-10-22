# 当前候选的先行范围核对

2026-10-16。使用literature-review workflow作**定向原文核对**，不是系统综述，
不能从检索未命中推导新颖性。本次未向外部服务提交本地数据或未公开公式。

[GGGS v2 §4.2](https://arxiv.org/html/2601.17835v2) 已定义乘积连续T、中位根、
隐式反传，并明确前峰条件下T与vacancy一致。这些定义不是本项目创新。
当前平方历史差的C1结论和临界反例须以有限正系数域的条件性数学补充呈现，
不能扩成整个renderer/visibility/无穷logit的全局正则性。

[2DGS §5、附录A](https://arxiv.org/html/2403.17888) 已有沿ray的深度聚集正则，
正文以距离表达，附录实际用NDC深度的平方差做CUDA累加。故“集中多层贡献”是既有目标；
本候选区别在连续median处的历史差及其正则性，不是首次表面聚集，也未证明优于distortion。

[Triangle Splatting v1 §3.1–3.3](https://arxiv.org/html/2505.19175v1) 使用有界支持
的三角形window，含learnable sharpness；其总loss已有opacity、2DGS distortion/normal和size项。
因此本Gaussian系数证明不能自动覆盖三角window。§3.2 midpoint细分保持面积/区域，
与保持alpha场或采样概率是不同命题，值得单独检查，尚不因此指控方法无效。
[官方实现](https://github.com/trianglesplatting/triangle-splatting/blob/main/scene/triangle_model.py)
的_update_params在读取时确实生成四个中点child并重复opacity/sigma；这是未固定commit的main观察，
未编译运行，不能当作已经复现其实际图像或完整MCMC。

[Gaussian Wrapping §3–4](https://arxiv.org/html/2604.07337v1) 已用learnable oriented normal、
相应attenuation、normal alignment与定向增密处理几何场一致性。不能把“指出GGGS多视图
一致性不保证”或“新增一致性正则”本身视为新贡献；它还改变了场与meshing，不能把它的
报告结果移植到本项目。本文仅核对定义和loss，没有复现其算法或验证SOTA数字。

当前判断：固定history-gap pilot失败后，没有依据声称得到新的质量方法。
保留数学补充与数值实现，新的TGS问题必须先满足原文/实现精确对应和相关工作检索，
随后独立实验，不把简单max、alpha合成或面积恒等式包装为原创理论。

检索范围：公开web发现＋上述四篇primary全文/官方源码；未完成多数据库或独立双人筛选，
未审核Triangle Splatting+、MeshSplatting及所有守恒densification/MCMC后续全文。
审查TGS细分前这些也是必需对照，不把本页有限覆盖称作新颖性证明。
