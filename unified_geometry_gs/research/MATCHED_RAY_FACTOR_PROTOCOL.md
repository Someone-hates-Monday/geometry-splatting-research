# 匹配基元的射线因素分离方案

2026-10-13，写于本检查运行前。上一轮24点与world/affine尾部已知，本轮为探索性机制检查，
不是盲确认实验、质量对照或训练方法。保留原3模型×8个预定查询射线，无重选。

## 身份与源文件核验

原affine_rays_cpu.py、原world/affine实际sourceszip与PLY SHA必须保持。
新CPU副本只传播原PLY行ID和阶段membership，不改投影/基元规则。
对24相机原project_model返回的每个数组/标量与新副本逐元素bitwise相等；
ray_parameters的alpha/peak/width/init也必须bitwise相等。
原affine根复算与保存深度差<=max(1e−8,2*旧absolute_error_bound)。
这些CPU一致性检查不是生产CUDA一致性或故障率检查。

## 固定基元集与形状因素

J为原CPU affine最终selected集合（中心near-plane、projection有效、像素tile、
power/alpha>=1/255、按camera-distance排序prefix stop之后）。
在同J上，从完整世界Gaussian得到p_W、sigma_W、alpha_W；alpha_W先同样cap .99。
以原p_A、sigma_A、alpha_A和世界系数组成全部2^3=8个1D混合，
三因素为peak、width、peak amplitude；均从−infinity开始、q=.5。
保存每个根、终态光学厚度和有限性，不用fallback补足不存在的根。
仅当8根全有限，按三因素全部6种加入顺序平均增量分配差值（即Shapley式分配），
并核验三项和等于t_WJcap−t_A。这是标准排列分配，不是新方法或可辨识因果效应。
混合参数射线只是受控1D数学构造，不宣称对应一个合法共享3D场。

## 固定顺序的其余差值

随后固定世界射线形状，依序比较以下嵌套集合（前后差值为有符号世界坐标距离）：

1. J：解除world peak alpha的.99 cap；
2. eligible：取消原affine prefix stop，保留其余筛选；
3. tile：取消affine power/alpha eligibility；
4. valid：取消查询像素tile membership；
5. near：取消projection有效/image-extent限制，仅保留center camera-z>.2；
6. full：取消center near-plane限制，全部原Gaussian；
7. full camera：由−infinity起点改为有限相机起点T_minus(t)/T_minus(0)。

集合ID和阶段计数保存（NPZ packed membership，原模型行次序）。
这条固定顺序的telescoping分配依赖路径，不能称唯一/独立/物理因果归因。
若任一中间根不存在，完整样本保留，差值/总分配记未定义，不删样本或换路径。
全部根的signed增量和应等于最终世界相机median−重算affine median（1e−9尺度余量）。
最终世界相机median还需匹配上一轮结果<=1e−8*max(1,|旧world根|)。

## 数值规则与范围

使用上一轮完整世界Gaussian restriction，从全部原参数开始，无距离/半径裁剪。
记录所有nominal alpha underflow=0；求和时可跳过这些名义精确零项以减少运算，
不将其解释为真实Gaussian消失；原模型与membership完整保留。
brentq xtol5e−13、rtol1e−13，最多32次括区扩展；最终logT residual<=1e−8。
分开报告全部24样本的有限性、三因素及各路径增量的绝对值/旧深度的分布、
相消量与原最大尾部例。不依据观测删极端例或仅报告最大贡献项。
这些量不是GT误差、方法收益、加速或显著性统计；不启动训练。
全部24点、归档、NPZ、逐点progress与终态保存，异常标failed并保存traceback。
