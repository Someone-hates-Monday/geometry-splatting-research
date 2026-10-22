# 分位区间均值的正则性、梯度界与表面混合代价

2026-10-13。固定有限GGGS射线模型的理论补充与CPU原型；不是完整渲染器或质量成果。
区间分位均值本身已有统计学先例，例如[Fissler与Ziegel](https://arxiv.org/html/1902.04489v3)
第2.1节。这里不声称发明新统计量，也不把参数正则性等同于该领域的统计稳健性。

## 定理与证明

设0<alpha_i<1、sigma_i>0，C_N=product(1-alpha_i)，固定C_N<q_l<q_u<1、
h=q_u-q_l；t(q)是连续透射率T(t)=q的唯一有限根。
定义M=integral_[q_l,q_u] t(q)dq/h。
在任一参数点附近，可选局部共同固定的有限[a,b]使T(a)>q_u、T(b)<q_l。
所有根均位于其内部。则

    M = a + integral_[a,b] clip((T(t)-q_l)/h, 0, 1) dt,
    M_theta = integral_{q_l<T(t)<q_u} T_theta(t) dt / h.

证明第一式：t(q)-a是集合{t in[a,b]:q<T(t)}的长度。对q积分交换次序，
内层q区间长度正是h乘clip表达式。
T对所有内部参数C1；clip是1/h-Lipschitz。除T(t)=q_l或q_u的两个点外，
对参数微分存在且等于第二式的被积函数。局部共同紧区间上T_theta有界，
支配收敛允许对积分微分，并允许梯度随参数连续变化。故M为C1，
即使区间内部某个单独分位根为临界根，也不需要对该根应用IFT。

冻结各层有效opacity、width和其他参数，仅对峰位微分时
T_{p_i}=T f_i' >=0，sum_i T_{p_i}=-T_t。因此

    0 <= M_{p_i} <= 1,       sum_i M_{p_i}=1.

对opacity，峰前T_alpha/T=-e/[2(1-alpha e)]；峰后
T_alpha/T=-1/(1-alpha)+e/[2(1-alpha e)]，e=exp(-(t-p)^2/(2sigma²))。
两式非正且绝对值至多1/(1-alpha)，T<=1。因此

    -(b-a)/(h(1-alpha_i)) <= M_{alpha_i} <= 0.

该粗界不依赖T_t的正下界。它依赖共同bracket、h及opacity到1的距离；
不是对透明边界、无限深度、h->0的统一定理，也不约束所有三维参数。

## 薄层极限与代价

固定互异有序p_i与alpha_i，sigma_i=epsilon s_i，epsilon↓0。
记ell_i=length([q_l,q_u] intersect [C_i,C_{i-1}])。则

    M_epsilon -> sum_i p_i ell_i/h,
    (M_epsilon)_{p_i} -> ell_i/h.

除有限多个平台和内部峰值分位外，已证严格跨越深度/位置梯度极限成立。
根在所有0<epsilon<=epsilon0下有共同有限bracket；位置梯度位于[0,1]。
分别对q应用支配收敛，得到两式。积分梯度在几乎处处分位可由IFT表达，
与空间积分式通过单调变量代换一致；有限个临界分位不影响积分。

若还排除内部平台等于区间端点，即所有i<N均C_i不等于q_l、q_u，则

    lim (M_epsilon)_{alpha_l}
      = -sum_{i>=l, q_l<C_i<q_u} (p_{i+1}-p_i) C_i / [h(1-alpha_l)].

这是对空间积分梯度使用支配收敛：峰间T趋于C_i，T_alpha_l趋于
−C_i/(1-alpha_l)（i>=l），否则0。区间指示因子在端点裕量下也收敛。
所以区间完全位于一个跨越层时，深度/位置权重仍集中于该层，opacity梯度趋于0；
区间跨平台时则产生混合深度，opacity敏感性由层间距和区间宽度控制。

双层p=(-1,1)、alpha=(.5,.5)：[.45,.55]薄层均值0，位于两个表面之间；
[.40,.55]第一层占1/3、第二层占2/3，薄层均值+1/3，也不是表面。
方案的−1/3手算符号错误已在 `QUANTILE_BAND_PROTOCOL_ERRATUM.md` 明确更正，原方案未改。
梯度稳定性因此不足以支持直接替代所有深度；需真实射线诊断表面混合和计算量。

## 实际核验与失败记录

`verify_quantile_band.py` / `results/quantile_band_v2.json`：

- 三层临界族7个预定epsilon全部保留。位置梯度非负/和1、opacity界与固定1e−6
  差分通过原门槛；最大归一化差分误差2.651e−11（floor1e−8、门槛1e−5）。
- epsilon=.075时M=.000459138855，位置权重约(1.51e−36,1,1.97e−36)，
  M_alpha2=−.532260208；同族单根的t_alpha2约−1.44629e37。
  函数改变了；不能把新梯度当成原中位深度的正确导数。
- 单层alpha=.75、sigma=.2中，q=.5单根不正则，但[.45,.55]均值梯度有限：
  M_alpha=−1.568633045，位置梯度1。没有在临界单根处套用IFT。
- 两种宽度、两个区间、三个厚度，12个平台/混合案例全部记录。
  不对最薄点误差另设事后通过门槛；不隐去不为表面点的结果。
- 实空间积分 vs 分位空间逐根积分，在预定三层最厚例和单层临界例一致；
  60位独立标量product-T积分补充复核。数值误差不构成证明。
- v1双精度极小尾部差分失败，实际源码/日志/方案封存于
  `results/quantile_band_failure_v1.zip`；v2保留double观测并用60位重新积分修复。
  没有放宽门槛。结果 `all_declared_checks_pass` 限于实际数值断言，须连同勘误阅读。

没有修改生产CUDA、没有启动新训练、没有效率或重建增益结论。
与GGGS、2DGS、TGS完整方法的关系及学术新颖性仍需核验。
