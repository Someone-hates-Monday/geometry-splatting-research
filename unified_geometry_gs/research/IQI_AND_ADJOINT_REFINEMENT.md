# 四轮求根的插值与反向精化：条件性数学说明和负结果

日期：2026-10-12。所有变体均为 `research` 内隔离 JIT；原安装后端与训练默认五轮求根不变。
本轮是已有四轮冻结渲染实验的后续开发，不是独立留出测试。

## 1. 已知工具与本项目问题

逆二次插值（IQI）是经典求根工具，参见 [SciPy Brent 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)。
快速迭代及一步 Jacobian 估计的近似分析已有文献：Bolte, Pauwels, Vaiter,
[One-step differentiation of iterative algorithms, NeurIPS 2023](https://arxiv.org/abs/2305.13768)。
[GGGS 原文](https://arxiv.org/html/2601.17835) 已使用隐式深度反传和细分求根，并讨论自适应括区。
Halley 公式亦见 [NIST DLMF 3.8.12](https://dlmf.nist.gov/3.8.E12)。
因此这里不能把 IQI、Newton、Halley 或一般隐式微分包装成原创贡献。
待检验的工程问题是：能否省去 GGGS 的一轮细分，同时用原有反向遍历恢复足够精确的梯度。
若没有实际训练的效率和质量证据，这只是一项数值实现探索。

设有效射线上的透明度 `T(t)=exp(-H(t))`，目标 `q=1/2`，
`g_i=alpha_i exp[-(t-p_i)^2/(2 sigma_i^2)]`，`z_i=(t-p_i)/sigma_i`。
其中有效 alpha、峰值和宽度均固定于同一射线；公式不能自动证明投影、裁剪或 tile 筛选的等价性。
定义

\[
R=qH_t=\tfrac14\sum_i\frac{g_i|z_i|}{\sigma_i(1-g_i)}.
\]

在不含任何峰值、且 `R>0` 的开区间上，

\[
R'=\tfrac14\sum_i\frac{g_i\operatorname{sgn}(z_i)}{\sigma_i^2}
\frac{(1-g_i)-z_i^2}{(1-g_i)^2},
\]
\[
R''=\tfrac14\sum_i\frac{g_i|z_i|}{\sigma_i^3}
\frac{(1+g_i)z_i^2-3(1-g_i)}{(1-g_i)^3}.
\]

峰值处 `H_t` 连续但一般不可微；先前已证明的临界根处还可能 `H_t=0`。
以下高阶结论均排除这些情形，并不构成对所有 GGGS 射线的一致保证。

## 2. IQI 的条件性误差与 CUDA 负结果

若三节点 `y_j=T(t_j)` 互异，逆函数 `f=T^{-1}` 在覆盖全部三节点的区间上属于 `C^3`，
以这三点对 `f` 作二次 Lagrange 插值，则

\[
f(q)-P_2(q)=\frac{f'''(\xi)}{6}\prod_{j=0}^2(q-y_j),\qquad
f'''=\frac{3(T'')^2-T'T'''}{(T')^5}.
\]

这是标准插值余项。还须保证逆函数存在、`T'` 远离零，且第三节点也处于同一光滑区间。
根的当前括区不跨峰，不能推出三节点区间不跨峰。CUDA 的互异/有限值/落在括区检查没有验证这些假设。
`verify_iqi_math.py` 的精确代数和光滑指数例验证了公式及约三阶误差，不能代替上述前提。

`median_iqi_frozen_v1.json`：3 个冻结场景、各 8 个视角，随机顺序 CUDA event 计时；
多数像素误差减小，最大深度差却增至 `9.5367431640625e-5`（同批普通四轮为 `5.8650970458984375e-5`）。
RGB、alpha、normal 和有效掩码保持相同；这些是固定输入的前向比较。
`iqi_backward_v1.json`：6 个固定视角、24 组 xyz/scale/rotation/opacity 梯度全部有限，
但最大相对 L2 漂移约 **4.6336%**，明显超过既定 **0.1%** 门槛。
拒绝用于训练，未启动 IQI 完整训练；保留原始结果，不因典型像素改善而忽略尾部退化。

## 3. 原有反向遍历内的一阶精化

令粗根为 `t0`，`F=T-q`。利用反向传播原本用于累积 `R(t0)` 的遍历，额外累积 `T(t0)` 与 `R'(t0)`，
再计算

\[
d_N=\frac{T(t_0)-q}{T(t_0)H_t(t_0)},\quad t_1=t_0+d_N,\quad
\widehat H_t=H_t(t_0)+H_{tt}(t_0)d_N.
\]

第二次既有遍历在 `t1` 处计算隐式分子，估计
`Jhat=-H_theta(t1)/Hhat_t`。这针对底层连续隐式根 Jacobian；
它不是四轮离散细分程序的精确自动微分。历史五轮后端同样使用隐式反传。
首个实现采用 `|d_N|<0.25 min_i|t0-p_i|` 与 `|d_N|<0.0002` 的工程保护，以及历史分母地板。
这些阈值不构成区间算术证书。

### 条件性二阶界

假定覆盖 `t0,t1,t*` 的同一不含峰值区间上：
`|T'|>=m_F>0`, `|T''|<=M_F`, `H_t>=m_H>0`,
`|H_tt|<=M_H`, `|H_ttt|<=L_H`,
`||H_theta||<=B`, `||partial_t H_theta||<=N`。
设 `e=|t0-t*|`，则由 Newton 余项
`e1=|t1-t*| <= M_F e^2/(2m_F)`。
设 `D=e+e1`，并另要求 `Hhat_t>=m_H/2`。Taylor 余项与分式差给出

\[
\|\widehat J-J_*\|\le
\frac{2N e_1}{m_H}+\frac{2B}{m_H^2}
\left(\frac{L_H D^2}{2}+M_H e_1\right)=O(e^2).
\]

证明：分子在 `t1,t*` 的差不超过 `N e1`；
分母对 `H_t(t1)` 的余项不超过 `L_H d_N^2/2`，
再加上 `|H_t(t1)-H_t(t*)|<=M_H e1`，利用两分母下界即可。
此结论不消除粗前向的损失误差：若仍乘以 `loss'(t0)`，
一般 `loss'(t0)-loss'(t*)=O(e)`，完整损失梯度仍可能只有一阶准确性。

`adjoint_math_v1.json` 的三组独立 float64 有效射线 oracle 显示二阶收敛，并拒绝指定临界例。
`adjoint_backward_v1.json`：21/24 组通过；scan55/0027 的 xyz/scale/rotation 仍失败，最大约 **1.3798%**。
`adjoint_guard_scan55_0027_v1.json` 显示该视角 123 个诊断像素中 3 个被保护条件拒绝。
最大粗 Jacobian 误差像素 (771,75) 的修正幅度为峰值距离的约 0.326 倍，
独立 float64 检查未跨峰，但被 0.25 规则拒绝；强制一阶修正后仍有约 0.1412% 的有效射线 Jacobian 误差。
这是机制诊断，不能等同于 CUDA 完整世界参数误差，也不是放宽最终质量门槛的理由。

## 4. 另存的高阶变体

`median_adjquad_backend.py` 保持普通四轮前向，用同一反向累积遍历再计算 `R''`。
设 `rho=(T-q)/T`，采用经典 Halley 步和二次分母展开：

\[
d_H=\frac{\rho}{H_t[1-\tfrac12\rho(1-H_{tt}/H_t^2)]},\qquad
\widehat R=R+R'd_H+\tfrac12R''d_H^2.
\]

保护条件为有限值、正分母、`|d_H|<0.5 min_i|t0-p_i|` 和同一绝对步长上限。
半峰值距离保证被接受的修正线段与所有峰值保持距离；不证明真实根就在该线段，也不控制浮点舍入。
这是根据失败机制开发的新变体，需完整重测；之前所有负结果和阈值保留。
在 `T` 为 `C^3`、`H` 为 `C^4`、`H_theta` 对 `t` 局部 Lipschitz、
所需导数有界且根简单并与全部峰值分离的局部假设下，Halley 根误差为 `O(e^3)`；
二次分母 Taylor 余项为 `O(|d_H|^3)`，同样分式差论证给出底层 Jacobian 的 `O(e^3)` 误差。
完整损失梯度仍受粗前向 upstream 导数的限制。
`adjquad_math_v1.json` 已通过精确分支二阶导数和 Halley 恒等式核验；三个 float64 光滑例的中间尺度阶数至少 2.996。
完整证明及三阶误差系数 `a^2-b` 已补入当前 `median_depth_note.tex`，内置编译通过。
`adjoint_loss_limit_v1.json` 还独立核验了一个单高斯平移例：底层峰值 Jacobian 被高阶修正，
但二次深度损失的完整梯度因粗 upstream 只有约 1.0000 阶。这不是对 NCC 偏差大小或方向的实证结论。

`adjquad_backward_v1.json`：24/24 组通过原有 0.001 相对 L2 门槛，
最大值 **0.000806163889161607**（0.080616%），全部有限。
这些是同一开发相机和像素的五轮对照，不是独立世界参数真导数 oracle。
`adjquad_step_timing_v1.json`：完整图像 RGB+0.01×depth 均值损失的前向+反向，
六个视角、5 轮×10 次、随机变体顺序；较五轮延迟降低 **6.503%～8.720%**。
清梯度在计时区间外，不含优化器、密化、PatchMatch、NCC，也不支持完整训练加速声明。

真实 PGSR 检查点的旧恢复顺序会使优化器绑定旧外观对象；`checkpoint_restore_v1.json` 复现了
原实现外观参数更新为 0，隔离修正更新约 0.00078618，捕获内容相同。
两臂都使用修正，并补齐滤波、绝对密化统计、RNG 与相机队列后，通过短程共同状态分叉。
完整 `experiments/adjquad55_v1` pilot 已完成：Δoverall=+0.000424087 mm，
Δs2d=+0.000908171 mm，Δ训练 PSNR=+0.066834 dB；未过原有几何门槛。
详情见 `ADJQUAD_PILOT_RESULT.md`。独立 123 射线乘积式 oracle 验证解析公式，
但发现 1 条被保护条件拒绝的射线退回粗根后仍有约 0.466% Jacobian 误差；不能只报告 122 条接受者。
详细固定方案见 `ADJOINT_TRAINING_PROTOCOL.md`；所有原负结果仍保留。

同状态五轮控制重复已结束：相对原控制平均 Δoverall=+0.001001144 mm、
Δs2d=+0.002160794 mm，大于原候选差异；单次训练对照不能隔离方法作用。
原失败门槛不变，不选择更有利的重复基线。峰值自适应恢复第五轮的 CUDA 版本
虽然通过 24 组梯度检查，六视角前向/反向却慢 0.42%～1.15%，不进入完整训练。

## 5. 科研工具溯源

本说明采用已安装 SymPy、hypothesis-generation 与 scientific-critical-thinking skills 的条件陈述、
独立检验和竞争解释工作流；未把工具说明当作方法有效性的证据。
按该技能集的溯源要求引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).
