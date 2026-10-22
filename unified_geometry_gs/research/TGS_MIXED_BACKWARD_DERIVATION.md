# 固定贡献分支下的 TGS 合成与 SH 几何梯度

这是上一阶段窗口导数补全的推广说明，不是新颖性声明。假设相机固定，三角形非退化，
每个选定像素的贡献者及深度排序固定，窗口有唯一 active edge，alpha 未饱和，SH 颜色
的 clamp 状态固定；涉及阈值的选择在一个开邻域内不改变。

## RGB 合成

一个像素处，深度顺序为 1,…,N，颜色 ci，alpha ai，背景 b。
Ti=product_{j<i}(1-aj)，最终 C=sum_i Ti ai ci + T(N+1)b。
令 Bi 为 i 之后所有贡献者与背景单独从 T=1 开始合成的 RGB。分离第 i 项后，
ai 的两处影响分别是自身权重和它对后面透射率的衰减，因此

```
partial C / partial ai = Ti (ci - Bi)
partial C / partial ci = Ti ai I3
```

这可直接从 C=C_before_i + Ti[ai ci + (1-ai)Bi] 求导得到。
令像素上游 RGB 梯度为 g，则 dL/dai=Ti g·(ci-Bi)，dL/dci=Ti ai g。
背景非零时必须包含在 Bi 内，不能只用后面的三角形颜色。

把第一式作为窗口的上游梯度，接上 `TGS_BACKWARD_DERIVATION.md` 的完整
phi/r、sigma、opacity 导数，就得到窗口对顶点的几何项。对多个像素相加即可。
这说明窗口半径缺项在遮挡合成中仍需传播；它不是单三角形特例。

## SH 颜色的另一条几何路径

设均值 m=(v0+v1+v2)/3、相机中心 c，d=(m-c)/|m-c|。
未被 clamp 的颜色是 .5+sum_l Y_l(d) beta_l。对归一化方向有

```
partial d / partial m = (I - d d^T) / |m-c|
dL/d beta_l = Y_l(d) * dL/d color
dL/dm = (I-d d^T)/|m-c| * sum_l grad_d Y_l * (beta_l · dL/dcolor)
dL/dvi += (dL/dm)/3
```

这里颜色 clamp 为负的通道先将其上游梯度置零；其余通道按上式传播。
几何总梯度需要把颜色方向路径与窗口路径相加。像素坐标的 W/2、H/2 因子只属于窗口
投影链，不应作用到已经在世界坐标下的 SH 均值梯度。

## 可验证性及限制

本阶段 Torch float64 显式图同时微分两个路径；独立 NumPy 用面积/周长窗口和真实 SH
多项式做复步长求导。复步长计算的范数采用 sqrt(sum(z*z))，不使用共轭；排序、方向符号、
clamp 等决策按实部冻结。只有已验证处于同一光滑分支时，这种复延拓才代表所需导数。

本轮不证明分支切换处存在普通导数，也不规定应该使用哪一种代理梯度。若 ai 的 0.99 cap
在某开邻域内激活，精确窗口导数应为零；这和保留代理梯度是不同约定，下一步需单独核验。
normal/depth/median/distortion 等辅助输出不在这段 RGB 推导或本轮实验覆盖内。
