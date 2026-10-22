# 含投影中心裁剪的packet恢复：方案

2026-10-13，写于本候选执行前；上一轮1946通过/46域外已知，属于后续开发而非盲确认。
旧v1源码、方案和结果保持，不能把新方法通过算成旧方法已覆盖clip。

## 数学候选

保留kernel0、Sigma=P^−1、z>0条件，但参考qbar=(ubar,vbar,1)可以是clip后的中心，
真实均值为z*q0，像素方向q不必接近qbar。现有数据给出t0=||mu||、
sbar²=||qbar||²/(qbar^T P qbar)、两个plane系数L（乘回fx/fy）、screen conic与amplitude。
由z=t0/||q0||、b=||qbar||²/sbar²，恢复

    c=[L*||qbar||/z+qbar_xy]/||qbar||²，
    K=diag(fx,fy)*conic*diag(fx,fy)/(z² b)。

在Mbar=[e1,e2,qbar]基底中，P/b的Gram为[[K+c c^T,c],[c^T,1]]。
令a=q_xy−qbar_xy、m=q0_xy−qbar_xy、A=1+c dot a、B=1+c dot m，
den=A²+a^T K a，cross=A B+a^T K m。

    p=z||q|| cross/den，sigma=||q||/sqrt(b*den)。

用非负分解计算峰值能量的Gram行列式，避免直接mu_quad−cross²/den相消：

    v=A m−B a，gdet=det(K)*(a_x m_y−a_y m_x)²+v^T K v，
    energy=z² b*gdet/den，alpha=amplitude exp(−energy/2)。

拟证明packet可恢复此P的完整二次型，clip不必丢失全部信息；
已有conic舍入/precision不互逆等问题仍不自动解决，不能宣称全生产域已覆盖。
此身份式的独立新颖性、GPU稳定性/速度未知；不把标准Schur工具或透视正确3DGS当原创。

## 固定检查

1. SymPy一般K/c/qbar/q0/q，检查metric、cross、positive Gram分解和plane编码逆；
   给出完整证明，不用有限采样替代。
2. 固定seed13016，128个SPD例：q0xy uniform[−1.8,1.8]，qbar=clip(q0xy,±.65)，
   qrayxy uniform[−.8,.8]，z uniform[.5,5]，随机QR旋转、轴长10^uniform[−2,0]，amp=.75。
   全部保留、不按是否clip重新抽样；直接因子空间oracle对比，
   double三分量scaled error<=1e−8、floor1e−12。
3. 同一24射线全部1992 selected输入，包括原46 clip输入；不改选择集合或筛选规则。
   从旧camera、CPU projection重新取得完整packet，参数与保存selected列表bitwise匹配。
   从packet的2D中心反算q0、从t0反算z（不额外读取原3D mean来帮助候选），
   qbar按原1.3视锥阈值恢复；与保存world restriction比较，门槛同上。
   每个输入保存packet/候选/reference/误差/clip标志，域失效或坏数值全部列出，不fallback。
4. 从原PLY再次独立80位因子空间重算全部1992输入，使用原保存world ray方向。
   源/archive/产物/ID分类均核验；通过也只支持此CPU输入范围，非CUDA/GT/速度。
5. 旧unclipped v1保持；同1946域上新旧表达应匹配原1e−8门槛。
   负数detK/energy/den、非有限或任何超阈值不裁成0隐藏；记录验证失败并保留当前源。

主生产/旧参考/旧负结果不改。不训练，不宣称图像、几何、梯度或效率收益。
逐点原子progress与终态；数值门槛可失败，完成采集不等于验证通过。
