# 极端样本的独立前向差分复核（v2）

2026-10-13，写于v2数值检查前、v1失败之后。v1独立q空间全VJP自适应积分
最终info.success为false，exit1；未输出具体积分状态码，不能断言精确失败原因。
实际源、方案和日志保存于results/quantile_band_oracle_failure_v1.zip。
原失败不撤回、不当作已通过，也不改主开发门槛。

保留同两个极端样本选择：最大band opacity范数、最大band/median范数比。
替代的独立数值图使用product-T、标量自适应空间积分及重新求根，
对每个opacity和peak分量作完整前向中心差分，不调用解析T参数导数。
opacity步长min(1e−6,alpha/10,(1−alpha)/10)，peak步长1e−5 median(width)。
前向绝对误差<=1e−8；完整梯度向量L2相对误差<=1e−5、floor1e−8。
所有分量、样本和失败保留；不能仅选通过分量。逐样本保存进度和最终完整报告。

这是另一种数值交叉检查，不等同修复v1 q空间全VJP积分算法。
成功也仅支持保存极端样本的空间梯度，不提供普遍误差界、真实质量或CUDA梯度结论。
