# 现有affine量的精确透视重参数身份

2026-10-13。数学补充/候选实现表达；不声称发明精确Gaussian射线相交或透视正确3DGS。
[HTGS](https://arxiv.org/html/2410.08129v2)已有无矩阵逆的Plucker射线评估，
[3DGUT](https://arxiv.org/html/2412.12507v2) Eq11已有精确最大响应射线距离。
本表达复用现有GGGS的affine conic/plane/width，是否有实现优势与新颖性尚未知。

## 条件与记号

相机原点0，均值mu=z q0、q0=(u0,v0,1)、z>0，P正定、Sigma=P^−1。
q=q0+delta、delta=(du,dv,0)，单位射线d=q/||q||。
使用真正的未clip中心Jacobian，无额外screen-space kernel；world滤波可以先吸收入Sigma。
令r=q0^T q0、b=q0^T P q0，t0=z sqrt(r)、s0=sqrt(r/b)。
未加额外kernel的normal-coordinate投影协方差为C=J Sigma J^T，
J=(1/z)[[1,0,−u0],[0,1,−v0]]。screen fx/fy通过delta变换吸收。
令E_A=delta_xy^T C^−1 delta_xy（未从cap后的alpha逆推），
Delta_A为精确射线峰位在q0的线性Taylor项：

    Delta_A=z[(q0 dot delta)/sqrt(r)−sqrt(r)(delta^T P q0)/b]。

这是GGGS affine plane对像素偏移的线性值，计算时直接保存plane dot diff，
不要先加t0再相减而引入不必要的相消。

## 身份式与证明

定义

    g=(q0 dot delta)/r，k=||delta||²/r，f=1+2g+k，
    ell=g−Delta_A/t0，chi=E_A*(s0/t0)²，D=(1+ell)²+chi。

则精确世界Gaussian限制到该射线的三系数为

    p=t0 sqrt(f)(1+ell)/D，
    sigma=s0 sqrt(f)/sqrt(D)，
    alpha=amplitude exp(−E_A/(2D))。

证明：令M=[e1,e2,q0]，则detM=1、JM=(1/z)[I2,0]。
写Q=M^T P M，其右下元素为b。由分块矩阵逆的Schur补身份，

    C^−1=z²[ P_xy,xy − (P q0)_xy (P q0)_xy^T/b ]，
    E_A=z²[delta^T P delta−(delta^T P q0)²/b]。

所以ell=delta^T P q0/b，chi=delta^T P delta/b−ell²>=0，
D=q^T P q/b>0，f=||q||²/r>0。
射线二次型的最大值位置、宽度、最小Mahalanobis能量分别为

    p=z ||q|| (q^T P q0)/(q^T P q)，
    sigma²=||q||²/(q^T P q)，
    E_W=z²[b−(q^T P q0)²/(q^T P q)]。

代入上式得到p、sigma身份；E_W=E_A/D给出alpha身份。
这些是同一精确射线模型的代数等式，不是第二阶截断近似。
SymPy一般对称3×3矩阵、一般q0/delta检查Schur身份、Taylor线性项与所有代数转换。

## 数值与实现边界

- 原投影中心clip会使conic/plane来源的Jacobian不再对应mu，不能套用上述证明。
- screen kernel、opacity cap、非逆的P/Sigma、conic determinant舍入均会破坏输入身份。
- D正性与理论准确不保证FP32稳定；既有conic的坏条件可能被继承。
- 复用现有packet不等于GPU更快、没有额外bandwidth或backward成本。
- 变换只修正单基元ray shape；不会自动修正tile/alpha/prefix筛选、排序或跨视图全局几何。
- 理论不证明摄影测量/图像/网格质量提升，也不保证此重参数形式在既有文献中未出现。

`results/perspective_reparam_theory_v1.json`：9项精确符号身份、seed13015的64个
预定SPD构造通过；double最大scaled errors(p,width,alpha)=
(9.1404e−14,4.3393e−14,8.3847e−13)，原1e−8/floor1e−12门槛不变。
真实模型检查须另看实际终态、全体selected输入和域排除数，不从合成检查预报其通过。
