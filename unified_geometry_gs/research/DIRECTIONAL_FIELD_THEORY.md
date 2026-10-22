# 方向衰减叠加与单法向标量场：点态定理和误差下界

2026-10-13。数学补充，不是完整GGGS渲染错误或原创方法声明。
已有相关理论：[GGGS](https://arxiv.org/html/2601.17835v2)、
[Gaussian Wrapping](https://arxiv.org/html/2604.07337v1)、
[Objects as Volumes](https://arxiv.org/html/2312.15406v2)。
后者允许一般法向分布；以下只检验单法向的绝对梯度投影形式。

## 固定点的精确兼容条件

令G_i(x)=a_i exp(−(x−mu_i)^T P_i(x−mu_i)/2)，0<a_i<1、P_i正定，
v_i=sqrt(1−G_i)。在固定x，

    b_i=grad log v_i = G_i/[2(1−G_i)] P_i(x−mu_i),
    K(w)=sum_i |w dot b_i|.

K来自逐基元绝对方向衰减相加，固定点时是一个齐次凸函数；它始终互易。
问是否有一个向量b，使所有单位w均满足K(w)=|w dot b|。

**定理1。** 存在这样的b，当且仅当所有非零b_i共线（零项允许）。

证明：若b=0，则K恒零，所有项必须零。否则对所有w垂直于b，有
sum_i|w dot b_i|=0；每项非负，故每个b_i都垂直于b的正交平面，即b_i与b共线。
反之写b_i=c_i u、||u||=1，取b=(sum|c_i|)u即成立。
这是点态条件，不据此宣称已经构造了一个全局可积的梯度场。

**推论1。** 若预先指定v=product_i v_i，则grad log v=sum_i b_i。
此时K(w)=|w dot grad log v|对所有w成立，当且仅当所有非零b_i同向共线。
因为定理1先要求共线，再在u方向要求sum|c_i|=|sum c_i|，正是同号条件。
因此“互易”本身不足以保证这种单法向表示，更不足以保证特定product-v场的表示。
这不否定一般法向分布的随机实体理论，也不否定GGGS已经说明的近侧等价条件。

## 与任意单个向量的不可消除差距

定义S=sum_i b_i b_i^T，lambda1>=lambda2>=lambda3>=0，

    E = inf_b sup_{||w||=1} |K(w)−|w dot b||.

**定理2。** E>=sqrt(lambda2(S))。

证明：对任意非零b，其正交平面与S前两个特征向量张成的二维空间存在非零交。
取交中的单位w，w dot b=0，且w^T S w>=lambda2。于是

    |K(w)−|w dot b||=K(w)>=sqrt(sum_i(w dot b_i)^2)>=sqrt(lambda2).

b=0时取最大特征向量即可。再对任意b取下确界，得结论。
该证明是基本矩阵变分和l1>=l2不等式；不把这些标准工具宣称为新代数。
意义在于提供本衰减模型的一个无需拟合b的、对全部方向成立的误差下界。

**尖锐例。** b1=e1、b2=e2。S=diag(1,1,0)，下界为1。
取候选b=e1，则K(w)−|w dot b|=|w2|，单位球上最大值恰为1。
因此E=1，下界不能作为普遍不等式再提高。

这能由合法Gaussian产生：取a1=a2=1/2、s=e^(−1/2)/(4−2e^(−1/2))>0、
Sigma1=Sigma2=s²I，mu1=−s e1、mu2=−s e2。在x=0恰有b1=e1、b2=e2。
若改成mu2=+s e1，则两梯度反向：任意单向量表示仍存在(b=2e1)，但product-v梯度为0，
不能表示K。这区分“任意单法向”和“指定product-v梯度”两个问题。

## 稀疏保守证书与有限精度

只用两项b1,b2即可得到一个可核验证书。令d=||b1 cross b2||²、
r=||b1||²+||b2||²，若r>0，则二项Gram的第二非零特征值为
d/lambda_max，故

    E_full >= sqrt(lambda2(S_full))
           >= sqrt(lambda2(S_pair)) >= sqrt(d/r).

第二步来自S_full−S_pair半正定；增加任意其他Gaussian不会削弱这个绝对下界。
如果d>0，证书严格正；如果d=0，证书无信息，不能反推全模型可兼容。
归一化比值需要完整分母或其可靠界，不能随意用子集分母宣称全模型相对下界。

数值脚本对原raw参数和保存的二进制浮点查询点做80位有向区间计算。
完整模型的双精度S与归一化统计只作名义测量；两项区间下界独立保存。
保存exact rational endpoints，避免把格式化十进制字符串误作保守下界。
库实现、输入解析和证书重放仍需独立核验；不是形式化证明助手的验证结果。

## 已核验与待核验

`verify_directional_field.py` / `results/directional_field_theory_v1.json`：
3个符号梯度、合法Gaussian实现、共向/反向身份式，以及64个预定随机证据方向通过。
随机检查不是sup/inf证明；上述证明本身给出所有方向的结论。

真实模型诊断遵循 `DIRECTIONAL_FIELD_PROTOCOL.md`，已经complete：24个预定查询点、
全体世界Gaussian、原PLY哈希、逐点结果及双项正证书全部保存。
`results/directional_field_world_report_v1.json`独立核验13源/archive、26产物、3PLY、
48raw基元、24精确查询点与有理数不等式、144个120位标量空间导数分量。
80位和100位区间均包含独立导数，后者嵌套于前者；两个进程均已结束。
mpmath.iv官方明确为实验性，不能把数值证据夸大成证明助手认证。
查询点不是GT表面、此世界模型不是per-view affine生产图，衰减差不是像素或重建误差。
不据此直接增加单法向正则化，因为真实边缘、多层和薄结构本就可能需要多个方向。
候选学术贡献是定量兼容性分析；与已有凸几何/随机实体工作的重叠还需核验。

文献重叠已确认一部分：K是zonotope sum[-b_i,b_i]的支持函数，单向量表示是线段。
[Siegel2023](https://arxiv.org/html/2307.15285v2) Eq1.7/1.8已有该表示及一般一致逼近问题；
[Torres2024](https://arxiv.org/html/2407.13125v1)研究相关Hausdorff逼近。
因此基础表示和一般逼近不能作为原创；此处仅补充当前Gaussian模型的条件、可达下界和证书。
新颖性未完成全面审查，尚不能视为已经达到论文贡献标准。
