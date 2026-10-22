# 含中心裁剪的affine数据仍可编码精确世界二次型

2026-10-13。使用标准Schur分解的Gaussian packet表达；不声称发明矩阵分解或透视正确渲染。
主条件仍为kernel0、P正定且Sigma=P^−1，原均值mu=z*q0、z>0，q0/qbar/qray的第三项均1。
qbar可为被裁剪的投影参考中心，不必等于真实q0；qray也不必与它接近。
本结论不保证有限精度的cov/precision实际互逆。

## 六个自由度的可恢复性

令rbar=||qbar||²、b=qbar^T P qbar>0，Mbar=[e1,e2,qbar]。
在此基底中写

    Mbar^T P Mbar = b [[K+c c^T, c],[c^T,1]]，

其中c=(P qbar)_xy/b，K=(P_xy,xy−b c c^T)/b正定。
固定qbar后，(b>0,c∈R²,K∈SPD2)与P∈SPD3互相唯一确定；
这就是已知的块Schur/LDL分解，维数为1+2+3=6，不是新的分解定理。

GGGS packet已有的sbar满足sbar²=rbar/b。两个plane系数乘回focal后为
L=z/sqrt(rbar)*(rbar c−qbar_xy)。screen conic乘回focal矩阵F后满足
F*conic*F=z² b K。又t0=||mu||=z||q0||，所以

    z=t0/||q0||，b=rbar/sbar²，
    c=[L sqrt(rbar)/z+qbar_xy]/rbar，K=F conic F/(z² b)。

q0可由保存2D中心和相机内参恢复；没有额外从raw 3D mean帮助候选。
因此即使投影Jacobian中心clip，这六个packet量在精确算术下仍编码完整P。
这不表示已有量的GPU读写/运算成本为零，或此表达较已有HTGS/3DGUT更快。

## 射线系数和非负能量

令a=qray_xy−qbar_xy、m=q0_xy−qbar_xy、A=1+c·a、B=1+c·m，

    den=A²+a^T K a >0，cross=A B+a^T K m，
    p=z||qray|| cross/den，sigma=||qray||/sqrt(b*den)。

证明只需将qray和q0在Mbar基底中的坐标写为(a,1)、(m,1)，
得到qray^T P qray=b*den、qray^T P q0=b*cross。
峰值能量的直接表达是z²b[(B²+m^T K m)−cross²/den]。
为避免直接相减两个大的近似数，令v=A m−B a，则Gram行列式身份为

    den*(B²+m^T K m)−cross²
       =det(K)*(a_x m_y−a_y m_x)²+v^T K v = gdet >=0，
    energy=z² b*gdet/den，alpha=amplitude exp(−energy/2)。

展开两边即可证明身份；SPD2确保右边非负。
浮点det(K)和交叉乘积仍可能相消，不作普遍浮点稳定性声明。
qbar=q0时退化为前一版unclipped公式。
精确复合映射与参考qbar选择无关，因此clip的中间非光滑性不自动导致最终真实ray映射非光滑；
本轮raw AD样本均离clip边界>1e−10，边界导数与实际FP32仍未测试。

## 当前核验

6个一般SymPy身份与seed13016的128个固定SPD构造通过，其中112触发clip。
1992个真实原selected输入全部保留并通过，包含原46 clip输入；
独立80位原PLY因子空间重算全部1992，最大scaled errors(p,width,alpha)
=(4.72e−15,1.52e−15,2.61e−12)，原1e−8/floor1e−12门槛不变。
与旧1946域内公式的最大差小于8.48e−14。
具体结果见 `CLIPPED_PACKET_RESULT.md`，FP32开发门槛失败另列，不能被本证明掩盖。
