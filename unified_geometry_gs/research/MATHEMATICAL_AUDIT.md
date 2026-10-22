# GGGS 数学与实验审查：已验证结论和待验证研究线

日期：2026-10-12。此文是研究工作记录，不是已经取得论文成果的声明。

## 当前证据

`results/evidence_audit.json` 重新读取了现有原始结果，未修改历史判定。
数据共有 15 个 DTU 扫描，其中 11 个有完整 baseline/pipeline 配对：10 个改善，1 个退化。
这 11 个扫描的平均 overall 差为 −0.0295654；排除用于选配方的三个 pilot，
其余八个配对扫描的均值为 −0.0316400。scan69/106/114/118 缺 baseline。
因此可以说“已有 11 个配对扫描支持整体配方”，不能说“15 个扫描均完成公平对照”。

在原三个 pilot 的独立消融中，no_dn 相对 baseline 的平均 overall 差为 −0.00377814，
平均训练视角 PSNR 差为 +0.04101 dB；冻结 no_dn 后改 trunc4 的平均 overall 差
为 −0.02033675。后者贡献更大。比较 JSON 与当前 results.json 曾由不同随机网格
采样得到，不应强行拼成逐位相等的加法分解。

现有图像指标来自训练视角；它们不能证明新视角泛化。cfg_args 没有保存完整优化配方；
代码 `safe_state` 固定种子 0，但这也不等于保证 GPU 逐位可复现。
原协议的 5e−4 是采纳门槛，不是统计显著性。

## 1. 稳定化平方 NCC 的精确导数

令参考和邻居的中心化图像块分别为 x、y，定义

\[
A=x^Tx,\quad B=y^Ty,\quad C=x^Ty,\quad
S=\frac{C^2}{AB+\epsilon},\qquad\epsilon=10^{-8}.
\]

这里 A、B 是平方和而非除以块大小的方差。CUDA 前向使用以上 S。
参考块固定时，精确偏导为

\[
\frac{\partial S}{\partial C}=\frac{2C}{AB+\epsilon},\qquad
\frac{\partial S}{\partial B}=-\frac{AC^2}{(AB+\epsilon)^2}
=-\frac{SA}{AB+\epsilon}.
\]

现有实现第二项为 −S/(B+ε)。在 C≠0、A,B>0 时，它与正确系数的相对差为

\[
\frac{g_{legacy}}{g_{exact}}-1
=\frac{\epsilon(1-A)}{A(B+\epsilon)}.
\]

直接对商求导即可证明；SymPy 与 Wolfram 独立得到同一结果。只有 ε→0 或 A=1
时两者一致。A、B 仅分别超过 5e−6，并不保证 AB 远大于 1e−8。
链式法则给出邻居强度梯度

\[
\nabla_y S=\frac{2C}{AB+\epsilon}x
-\frac{2SA}{AB+\epsilon}y.
\]

`ncc_exact_backend.py` 在独立 CUDA 扩展中只改第二个系数。
`--ncc_exact_gradient` 是训练的显式可选开关；默认配方不启用。
它不是新的损失，也不修改阈值、损失权重或颜色梯度路径。
原有已安装 CUDA 包不被替换。

在低对比度合成 warp 的 368 个有效块上，前向和 validity 逐位一致；
深度梯度中位绝对误差由 0.0448646 降至 9.956e−7；独立双精度参考的
中心差分中位误差为 4.885e−10。高对比度测试同样通过前向一致性。
这些是导数验证，不是网格或 PSNR 改善。

## 2. 平方 NCC 的光度假设边界

不计 ε，固定参考块，允许任意实数光度增益 a 和偏置 b 时，

\[
\min_{a,b}\frac{\|I_n-aI_r-b\mathbf1\|^2}
{\|I_n-\overline I_n\mathbf1\|^2}=1-\rho^2.
\]

证明：对 b 最小化使两块中心化；对 a 的二次函数最小化得到 a=C/A，
代回为 B−C²/A，再除以 B。如果约束 a≥0，则结果改为

\[
1-\max(\rho,0)^2.
\]

所以平方相关度把负增益也视作有效解释。它能表达“允许对比度反转的 affine
匹配”，但不足以单独表达“正增益光度一致”。固定 ε 的实现还会失去严格的
增益不变性；不能把无 ε 的恒等式直接宣称成当前实现的完全等价。

三个冻结模型各取 8 个参考视角、每视角至多 4096 个几何有效候选点，
生产 CUDA 在 τ=0.9 接受的块中，负相关比例为：scan37 1083/18630≈5.813%；
scan55 133/30884≈0.431%；scan63 744/15333≈4.852%。
总采样点为 98,304。这是图像块统计，独立实验单位仍是扫描/训练种子，
不能把每个块当作独立场景样本进行显著性检验。

负相关不等于已知错误对应：反射、遮挡、局部形变和光照都可能导致它。
正相关约束也属于已知匹配思想，单独使用不能宣称原创方法。
暂不把它加入生产损失；先验证精确梯度修正，再独立注册正增益消融。

若 S 为有限非负数，e=clamp(1−S,0,2)≤1。因此 τ=1.2 实际关闭相关度筛选，
而非一个仍排除低相关块的宽松阈值；validity 与几何 mask 仍保留。
原四点 τ 曲线可作为描述性观察，但不能据此证明偏差/方差的单调规律或总体 U 形。

## 3. 随机实体深度与固有表面的精确差距

GGGS 原论文已经明确指出表面一致性的条件：中位深度在贡献 Gaussian 的峰值之前时，
透射率才与空间 vacancy 一致。本节补充一个精确差距公式和反例，
不把论文的条件性论述曲解成无条件承诺。

对某条射线，令

\[
G_i(t)=\alpha_i\exp[-(t-p_i)^2/(2\sigma_i^2)],\quad
v_i(t)=\sqrt{1-G_i(t)},\quad 0\le\alpha_i<1,\ \sigma_i>0.
\]

采用论文的连续透射率定义

\[
T_i(t)=\begin{cases}v_i(t),&t\le p_i,\\
(1-\alpha_i)/v_i(t),&t>p_i.\end{cases}
\qquad T(t)=\prod_iT_i(t),\quad V(t)=\prod_iv_i(t).
\]

**命题：** 对任意有限 t，

\[
\Delta(t)=\log\frac{V(t)}{T(t)}
=\sum_{i:t>p_i}\log\frac{1-G_i(t)}{1-\alpha_i}\ge0.
\]

**证明：** 对未越峰值的分量，v_i/T_i=1；对已经越峰值的分量，
v_i/T_i=v_i²/(1−α_i)=(1−G_i)/(1−α_i)≥1，
因为 G_i≤α_i。相乘再取对数即得。
对正 opacity 和有限正宽度的 Gaussian，t>p_i 且 t≠p_i 时该因子严格大于 1。
所以除零 opacity 等退化分量外，T=V 的充要条件是没有越过贡献分量的峰值。

在 T(t_m)=q 的中位/分位深度处，V(t_m)=q exp(Δ(t_m))。
Δ 是一个显式的射线历史项；它不是一个天然空间标量置信度。

**局部深度误差界：** 若 [t_m,t_v] 上 log V 严格递减，V(t_v)=q，且

\[
0<m\le-\partial_t\log V\le M,
\]

则对上述恒等式使用中值定理得到

\[
\frac{\Delta(t_m)}{M}\le t_v-t_m\le\frac{\Delta(t_m)}{m}.
\]

必须同时有这样的 root 和导数下界；不能把该界应用于任意非单调 V。
它说明“搜索足够精确”与“深度对应同一个空间等值面”是两件不同的事。

**可复现反例：** 两个分离的 Gaussian：α₁=α₂=0.4，p₁=0、p₂=1，
σ₁=σ₂=0.03。T(+∞)=0.36<0.5，所以透射率中位深度存在。
然而任意 t 至少离一个峰值不小于 1/2，因此

\[
V(t)\ge\sqrt{(1-0.4)
\left(1-0.4\exp[-1/(8\cdot0.03^2)]\right)}
>0.77459>0.5.
\]

这证明 V=0.5 的等值面根本不存在，不依赖有限网格采样。
数值 root 为左向右 0.977981842；反向观察时对应世界位置为 0.022018158。
中位深度处 V≈0.833333333，Δ≈0.510825624。
`verify_stochastic_surface.py` 还验证了 200 组随机混合、20,000 个位置的恒等式，
最大数值误差约 1.07e−14。随机验证只用于检查实现，普遍结论由上面的代数证明支撑。

这是可写成理论补充的候选，尚未证明在训练后的 DTU 模型中大量发生，
也尚未完成原创性排查。下一步是测量真实 median 点的 Δ，而不是直接增加一个正则项。

## 4. 下一轮实验的事先约束

先做精确梯度修正的成对试验：相同源代码、初始点云、种子、视角顺序、
30,000 步、no_dn、λ_ncc=0.6、λ_geo=0.02、单近邻、τ=0.9，
仅切换 `--ncc_exact_gradient`；提取统一 v=0.002、trunc=4。
首先 scan37 做可运行性/方向性试点，结果标为探索性；随后按既有三个 pilot
及至少三个种子验证。不以现有不同 seed/不同版本的历史模型充当严格成对对照。

主指标沿用整体 Chamfer、d2s、s2d；图像部分分别记录训练视角和独立新视角实验。
效率报告训练耗时、峰值显存、Gaussian 数和固定场景渲染时间。
改善门沿用平均 overall≤−5e−4、s2d≤0、PSNR≥−0.2 dB，
并报告场景差异与种子波动。没有重复实验时，不使用“显著改善”的统计表述。
不同时改变正相关筛选、数值方差计算或 TSDF 提取。

理论线先检验真实 Δ 的分布及其与跨视角深度不一致的关系。
若 Δ 大多接近零，则该反例只保留为适用条件说明，放弃据此设计主方法。
若存在稳定关系，才研究带误差证书的深度/表面提取，并单独注册新协议。

三角形 TGS 的作用是外部表示参照，不能把三角形基元本身或硬件渲染效率
直接当作本项目创新。GGGS、2DGS、TGS 的公平跨表示评测另列预算和输入协议。

## 工具与出处

另见 `MEDIAN_SOLVER_THEORY.md`：连续中位深度的隐式反传还需要非零界面斜率，
单 Gaussian 在射线有效 α=0.75 时存在平方根型退化；已给出根括区证书。
CPU 随机验证通过，但当前混合 Newton 原型没有减少评估次数，因此不作加速声明。
新检索到 Gaussian Wrapping 和 TSPE-GS，已记录表示/建面/多峰深度方向的文献重叠。

2026-10-12 训练集成：scan37 两组各 50 步已完成；两后端各实际调用 50 次，
首轮有效反传块均为 48,667。完整 30k 成对实验已启动于
`research/experiments/exact37_v1`，只改变 NCC 导数，训练/提取/评估由单一 manifest 记录。
完整结果未生成前，不宣称几何或图像质量改善。

已安装官方 `jupyter-notebook`，以及 K-Dense 的 `sympy`、`scientific-critical-thinking`、
`hypothesis-generation`、`statistical-analysis`、`scientific-visualization`、`literature-review`。
K-Dense 固定到提交 `92ace75ac21efe19a620434e0ca4e356081fe807`。
本次实际应用 SymPy 与批判性审查流程，软件来源列入以下参考。

本地环境：RTX 5090 Laptop，PyTorch 2.12.0.dev20260314+cu128、CUDA 12.8、
Open3D 0.19.0、Trimesh 4.11.3、SymPy 1.14.0。
Consensus 文献检索及原始记录读取已成功；Wolfram 符号计算已成功，context 查询曾返回
内部错误；Hugging Face 的论文搜索返回工具不可用，当前不依赖该项。
未找到可直接用于此科研流水线的 Blender 插件，也未检测到标准路径下的 Blender 安装，
因此复用实际可运行的 Open3D/Trimesh 进行 3D 验证。

- Zhang et al. Geometry-Grounded Gaussian Splatting，§4.2 和附录 C/E/G：
  https://arxiv.org/html/2601.17835
- 原作者代码（上游已更名；本地历史版本保持原样）：
  https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids
- Huang et al. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields：
  https://arxiv.org/abs/2403.17888
- Held et al. Triangle Splatting for Real-Time Radiance Field Rendering：
  https://arxiv.org/abs/2505.19175
- Chen et al. PGSR：本次由 Consensus 检索并读取原始记录，后续新方法排查须读全文：
  https://consensus.app/papers/pgsr-planarbased-gaussian-splatting-for-efficient-and-chen-li/dca0faae20d3591c8a7e224ed457bed4/
- Kassis, Agarwal, He, Patel, Brueckner (2026). Scientific Agent Skills: A Library
  of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
- 技能源码：https://github.com/K-Dense-AI/scientific-agent-skills

理论推导、反例及实验脚本仍需外部审阅。上述出处不意味着这些补充命题已有独立创新性结论。
