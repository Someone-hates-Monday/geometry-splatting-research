# 从affine射线量恢复精确透视系数：验证预案

2026-10-13，写于本候选运行前。候选由24射线因素检查启发；不是盲确认。
透视正确3DGS已有HTGS、3DGUT等先例，不将精确Gaussian射线限制或一般透视校正称原创。
候选仅是现有GGGS affine量的一种代数重参数表达；实际效率、新颖性和质量未知。

## 拟证身份式

相机原点0，Gaussian均值mu=z*q0、q0=(u0,v0,1)、z>0，P正定且Sigma=P^−1。
像素方向q=q0+delta，delta=(du,dv,0)，单位射线d=q/||q||。
设r=||q0||²、b=q0^T P q0、t0=z sqrt(r)、s0=sqrt(r/b)。
affine峰位为t_A=t0+Delta_A，其中Delta_A是精确峰位函数在q0的线性项。
E_A为无屏幕kernel的affine投影conic二次型（未用.99截断后alpha倒推）。

    g=q0 dot delta/r，k=||delta||²/r，f=1+2g+k，
    l=g−Delta_A/t0，chi=E_A*s0²/t0²，D=(1+l)²+chi。
    p_exact=t0 sqrt(f)(1+l)/D，
    sigma_exact=s0 sqrt(f)/sqrt(D)，
    alpha_exact=amplitude exp(−E_A/(2D))。

拟证明E_A=z²[b*(delta^T P delta)−(delta^T P q0)²]/b，
从而chi=h−l²、h=delta^T P delta/b，D=q^T P q/b>0。
conic二次型等式来自projection Jacobian和协方差逆的Schur补。
这只要求现有conic、affine峰位线性项、中心depth、宽度、2D中心与相机参数，
但不据操作数或无需新增学习参数宣称实际加速/零带宽成本。
夹角中心裁剪、屏幕kernel、无效conic、非逆P/Sigma及浮点误差可能破坏身份式。

## 固定核验

1. SymPy精确检查一般P符号的ray peak/width²/peak energy身份及Taylor导数；
   Schur等式使用一般3×3对称P与q0、二维delta，不只测试各向同性。
2. 固定seed13015，64个SPD例、q0xy in[−.8,.8]、z in[.5,5]、
   delta_xy in[−.15,.15]、随机orthogonal R、轴长10^uniform[−2,0]。
   与直接因子空间完整射线限制比较p/width/alpha；double相对误差<=1e−8，floor1e−12。
   失败保留并停止通过声明，不改范围或放宽阈值。
3. 真实输入为刚完成匹配因素检查的全部24射线、全部selected原基元。
   原NPZ/源/archive哈希须匹配；不得仅选大误差或有效样本重新凑数量。
   按原camera、model重构q0与pixel方向；列出中心触发u/v clipping、conic非正等域排除数，
   所有被排除输入仍保存，不将它们算作通过。
4. 对适用域的每一输入比较候选与保存原world restriction的p/width/alpha，
   double误差阈值同1e−8/floor1e−12。另比较原affine系数的误差，
   逐点保存全部输入/系数/误差，不能将平均值当尾部通过。
5. 本检查不包含CUDA forward/VJP、图像/NCC、GT或计时，均不得先行声称。
   主生产与冻结参考源保持不变；新脚本失败则归档当前源码/方案/日志再修复。
