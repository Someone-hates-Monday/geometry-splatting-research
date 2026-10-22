# Packet浮点与原参数反向检查

2026-10-13，写于本检查执行前。含裁剪double前向已知通过，本检查为后续开发门槛。
保持原24射线1992输入与128个固定SPD构造，不根据输出重采样或删例。

1. FP32 packet：只将已有double packet的所有数值输入转为float32，再以NumPy float32
   运算调用含裁剪公式；核验输出dtype。此项仅检查packet舍入/求值，
   **不是实际FP32/CUDA预处理器**，不能覆盖raw quaternion归一化、cov/conic生成等误差。
   前向对保存double世界系数：peak相对误差<=1e−5、width相对误差<=1e−4、
   alpha绝对误差<=1e−4（p/width分母floor1e−8）。三阈值是开发筛查，不是论文质量门槛。
   1992真实输入与128合成分别报告全部失败、最大/分位与clip分类。
2. FP64完整raw参数链：对所有1992输入，以原PLY mean3、logscale3、raw quaternion4、
   opacity logit1、filter radius1共12变量构造torch图。新图先按原clip/kernel0规则构造
   affine conic/plane/width，再用一般packet公式；独立参照直接世界因子空间ray限制，
   不经过screen covariance/conic或packet。
3. 对peak/width/alpha三个输出分别作单位cotangent，得到每输入3×12个分量。
   比较前向原1e−8/floor1e−12门槛；梯度每分量scaled error
   abs(new−ref)/max(1e−8,abs(ref))<=1e−4；所有71712个分量记录，不能只比较总范数。
   两图都是autograd，属于独立代数图交叉核验，非CUDA backward或解析VJP验证。
4. 记录raw中心到clip阈值距离；若恰在边界(<=1e−10归一化坐标)，
   标记非唯一可微性，不当成唯一Jacobian通过。所有样本仍保留。
5. 固定数值容差不在观测后放宽；非有限、负energy/det/den不裁正。
   前向/梯度开发门槛失败原样保存，完成采集不等于方法通过。

本检查不包含GPU计时、实际生产FP32 packet、完整renderer/NCC或GT质量。
不因这个局部检查自动启动训练；通过后还需真实CUDA前向/反向与资源/开销评估。
