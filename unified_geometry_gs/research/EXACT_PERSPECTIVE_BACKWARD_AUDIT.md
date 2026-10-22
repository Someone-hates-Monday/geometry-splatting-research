# 精确透视系数的完整后向与支持域审查

2026-10-13。静态审查当前 `submodules/diff-gaussian-rasterization/cuda_rasterizer`，
与小型独立数值核验。没有修改后端、旧实验、LaTeX或训练流程。
这里的“精确”只表示所给Gaussian二次型沿射线的(p,ρ,α)正确；
不表示离散alpha RGB合成已经变成精确的连续体积颜色积分，也不表示支持域完整。

## 1. 当前真实调用链与不兼容处

- `render_forward.cu:484–532`：affine screen power→cap .99→alpha≥1/255→test_T≥1e−4。
  越过early-stop的当前primitive本身被排除；`last_contributor`指向最后已接受位置。
  RGB、normal权重、depth初值都使用这次同一alpha和前缀。
- `render_forward.cu:599–621`：median仅在前缀内再次检查affine power/alpha，
  p=plane_x*d_x+plane_y*d_y+plane_z、ρ=plane_w；因此ρ目前是每primitive常数。
  它求的是从−∞原点定义的GGGS T(t)，不是 T(t)/T(0)。搜索下界被截在0不等于原点条件化。
- `render_forward.cu:643–664`：有限bracket线性插值，输出z-depth=t/||q||。
  slope通道是最后bracket的割线，不是精确根处解析斜率。
- `render_backward.cu:839–882`：用同一旧affine系数重算根斜率，假设T=.5，并把分母夹在1e−7。
  它不微分有限bracket算法。夹斜率是梯度修改，不能声称临界根的正确隐式导数。
- `render_backward.cu:984–1054`：p的梯度只传给plane、d；ρ只传给plane_w；
  alpha只传给affine conic和screen center。精确p、ρ、α均依赖同一μ/P/射线，旧链缺失导数。
- `render_backward.cu:1000–1015`：slope backward明确是STE。
  forward割线与解析S后向不是同一函数；还没有形成解析S(t*(θ),θ)完整的根联动导数。
- `sample_forward.cu:137–160,256–283`：evaluateTransmittance、sampleDepth/evaluateSDF各自执行同样affine alpha筛选与前缀。
  sampleDepth支持任意投影查询点；它不是render像素root的简单插值。
- `sample_backward.cu:210–224,289–327`：再次复用affine p/ρ/alpha链，
  同时用mean2D梯度取反充当查询点坐标梯度。精确透视后不再可用该平移对称替代。
- `rasterizer_impl.cu:522,557`及`1318,1347,1383`：render与sample均先local backward、后同一旧preprocess backward。
  改forward后若仍调用这些旧geometry路径，会静默生成错误梯度。

必须保持query、primitive和camera三者的坐标约定；从render输出z-depth到sample返回点、NCC输入的单位转换不能丢失。

## 2. 必须禁用／替换的旧路径

精确模式不能调用旧 `BACKWARD::render` 或 `BACKWARD::sampleDepth` 的geometry部分，
也不能把新系数upstream假扮成旧 `dL_dray_planes`／`dL_dconic2D` 后交给旧preprocess。
禁止仅替换forward并沿用原 `_RasterizeGaussians.backward` 包装器。
无已核验后向的原型应在需要梯度时明确报错；冻结模型评估可使用独立forward入口。

可复用的数学部分是alpha合成的反向递推、RGB特征梯度、同一normal归一化规则；
其输入必须来自新forward缓存的**同一系数、cap mask、membership与prefix**。
若保留SH/SG程序，其颜色对mean方向的梯度也必须保留。
最稳妥的隔离方法是独立颜色VJP或Torch SH/SG图，而非调用含旧cov/normal计算的整套preprocess，
因为把旧geometry upstream置零也未自动证明不会出现0*Inf或重复几何累计。

## 3. 可直接实现的最小完整因子空间VJP

这不要求手工微分冗长的Schur packet图。若目标是正确透视renderer，
可直接提供每primitive相机系的μ、precision因子F（P=FᵀF）和峰值amplitude A。
Torch在外部构造F/μ/A及raw参数激活，CUDA仅负责射线局部系数与合成。
若研究目标特指“复用原packet”，则需对新recover完整AD／解析VJP，不能跳过其qmean、qbar、conic、plane联动。

设o为相机原点、d为单位射线，

    δ=o−μ, u=Fδ, v=Fd, Q=v·v>0,
    p=−(u·v)/Q, ρ=√Q, w=u+p v,
    E=w·w, a0=A exp(−E/2), α=min(.99,a0).

接受区间之外的hard membership与排序当前按离散变量固定；在分支边界没有普通导数。
给定任意下游梯度(g_p,g_ρ,g_α)，设c=1[a0<.99]；远离cap边界，

    g_A=c g_α exp(−E/2), g_E=−½c g_α a0,
    g_u=−g_p v/Q+2g_E w,
    g_v=g_p(−u/Q−2p v/Q)+g_ρ v/ρ+2g_E p w,
    g_F=g_u δᵀ+g_v dᵀ,
    g_μ=−Fᵀg_u, g_o=Fᵀg_u, g_d=Fᵀg_v.

推导：w·v=0使E的dp项消失；对p的商和ρ的平方根直接求导。
式子是**精确实数**导数。有限精度w·v不完全零时，这些式子微分数学函数，
不保证逐条微分某一浮点指令序列；需要独立CUDA数值核验。

若d=q/||q||，

    g_q=(I−ddᵀ)g_d/||q||.

固定相机render的q不训练，但sampleDepth的query射线必须保留这项。
对每个共享primitive应累计所有ray的g_F、g_μ、g_A；当前按输入复制raw参数的旧检查不覆盖真实atomic汇总。

raw链需要保留：

    s_i²=exp(2ℓ_i)+filter²,
    R=rotation(normalize(raw_quaternion)), F=diag(1/s_i)Rᵀ（再转相机系），
    A=sigmoid(opacity_logit)*√Π_i[exp(2ℓ_i)/s_i²].

这对应 `scene/gaussian_model.py:151–159,189–208` 的3D filter与opacity补偿。
不要重复应用Python已处理的激活，也不能把opacity补偿从scale梯度中detach。
kernel_size>0的2D额外抗混叠不是这个精确世界Gaussian；当前候选应显式限制kernel0或另定义并验证滤波模型。

## 4. RGB、normal与根的upstream如何接入

固定顺序和已接受集合J。T_i=Π_{j<i}(1−α_j)、权重w_i=T_i α_i。
RGB=Σw_i c_i+T_end c_bg、alpha_out=1−T_end，均有标准alpha合成反向。
它们只需给局部VJP提供g_α及feature梯度；不应再把g_α经过affine exp(power)链。
cap激活时整个a0→α路径导数为0；当前旧源码未在旧affine链显式乘这个mask。
保留旧forward cap不等于必须保留这一旧backward近似。

为隔离透视系数影响，建议保留每primitive原normal定义、SH/SG颜色定义与normal归一化。
理想精确算术下，当前 `computeCov2D` 的normal可化为

    n_i=−normalize(P_cam qbar_i).

这不是median世界点处的Gaussian梯度normal，也不是每ray的−normalize(P_cam qray)。
替换成后者会额外改变normal模型，应单列消融。
即使n_i不变，α与prefix变化也会改变Σw_i n_i以及归一化结果。

此旧normal定义可独立完整微分：t=Pqbar、n=−t/||t||，给定g_n，

    g_t=−(I−nnᵀ)g_n/||t||,
    G_P=g_t qbarᵀ, g_qbar=P g_t,
    g_F += F(G_P+G_Pᵀ).

再把qbar=clip(μ_xy/μ_z)的两坐标梯度传给μ，clip内为商的导数、外为0。
边界采用的单侧／subgradient约定必须与实际前向实现一致；本次小检查避开边界。
若用Σ预计算路径或FP32非正交R，原CUDA normal与此理想化式子的逐bit一致性尚未建立。

根使用H(t)=Σ_i h_i(t)，

    x_i=(t−p_i)ρ_i, G_i=α_i exp(−x_i²/2),
    h_i=−½log(1−G_i)                      t≤p_i,
    h_i=−log(1−α_i)+½log(1−G_i)           t>p_i,
    H_t=Σ G_i |x_i|ρ_i/[2(1−G_i)].

在固定集合、有限且H_t>0的根H(t*)=log2，任意下游g_t给

    g_(α,p,ρ) += −g_t/H_t · ∂_(α,p,ρ)H(t*)（t*求偏导时固定）。

从输出z=t*/||q||传来g_t=g_z/||q||，同时
g_q += −g_z t* q/||q||³。这个显式单位转换和局部系数中的query依赖都要保留。
sampleDepth若返回相机点X=t*d，则g_t=g_X·d、g_d+=t*g_X，再走d(q)链。
原sample query点坐标的梯度不能简化成primitive mean2D梯度取反。

若改为相机原点条件化，应使用Froot=H(t*)−H(0)−log2，
parameter导数也必须减去∂θH(0)；这是新的模型变更，不能与单纯透视修正混在一个对照。
临界根、无有限根、无有效括区间必须明示mask／失败，不应把1e−7夹分母说成正确IFT。

若保留slope训练损失，最低数学一致方案是forward返回解析S=exp(−H)H_t，
并微分S(t*(θ),θ)。给root的seed需再加g_S ∂tS，给参数加g_S ∂θS。
旧forward割线配解析STE应禁用，或明确作为独立近似消融保留；不能称完整精确后向。

## 5. NCC和跨视图几何

NCC warp接收z-depth与normal，见 `research/ncc_reference.py:43–78`。
修正透视系数后应正常接收其g_z与g_n，并按上一节回传，不能detach depth或normal以绕开未完成路径。
已知NCC方差导数修正是另一变更；对照两臂使用相同NCC实现。
patch有效mask、grazing denominator、图像bilinear边界是额外离散／数值条件，局部VJP不覆盖。
跨视图 `sample_depth` 必须替换相同系数与query VJP；仅替换render不足以支持原训练loss。

## 6. 冻结几何比较中的选择域混杂

现有tile候选来自Jacobian投影协方差的3σ屏幕圆；near cull为μ_cam.z>0.2（`auxiliary.h:144`），
tile排序键是primitive中心camera距离（`rasterizer_impl.cu:92–100`），不是逐pixel精确p。
将alpha改为精确射线峰值会同时改变：

1. α≥1/255的membership；
2. T_end<1e−4前缀和其后所有输入；
3. median初值、搜索括区间有效性和finite-root比例；
4. RGB与normal权重；
5. sample/evaluateSDF各自query处的集合；
6. 如果重新排序或扩大tile，额外改变遮挡混合与几何集合。

所以至少需要分别记录“固定旧J/prefix，仅改p/ρ”“同J/prefix再改α”与“自然新选择域”的结果。
固定旧J是诊断反事实，不是完整新renderer支持域证明。
自然域比较应报告共同有效像素、两臂各自无效及仅一臂有效数量，不能只比较有利交集。
kernel、near cull、cap、clip参考、origin条件化、normal规则、SH/SG、排序、搜索精度必须逐项声明。

旧3σ圆不自动包含所有满足精确α≥1/255的射线。精确支持的严格几何表述为：
A≥τ时令L=2log(A/τ)，直线与椭球(x−μ)ᵀP(x−μ)≤L相交，当且仅当其精确峰值α≥τ。
因此可用该椭球的保守透视投影做候选范围；若椭球跨过相机平面，投影可能无界，需保守全图或单列域限制。
若μ_z>√(LΣ_zz)，所有椭球点在相机前方，可由

    (μ_x−u μ_z)²=L(Σ_xx−2uΣ_xz+u²Σ_zz)

两实根得到u=x/z的切线界；y同理。该等式是线性泛函x−uz的椭球极值条件，属于标准投影二次曲面几何。
它需要与候选所定义P一致的Σ=P^−1；仅假设舍入的rotation正交会再次引入不一致。
本次没有实现或验证完整tile域；保守bbox也不消除prefix／排序近似。

## 7. 最小可交付实施顺序与验证门槛

先做独立forward入口并在requires_grad时拒绝旧backward；固定模型对照声明其domain。
然后新入口提供缓存α/p/ρ及离散mask，组合reverse生成(g_α,g_p,g_ρ)，使用第3节共享primitive VJP。
raw激活、normal、颜色可先用Torch图，减少首次手工CUDA链的遗漏。
同时替换sample query路径；只有render或只有local primitive通过时都不能开始声称原训练loss完整可用。

进入训练前必须新增真实CUDA forward与包括cap、clip、NCC、slope、sample query、共享primitive累加的后向对照，
有限差分在固定支持域内做；边界行为另测。还要保证新median forward残差足够小，否则IFT与有限算法仍不一致。
若采用原finite bracket forward，则只能标为implicit surrogate backward，不能宣称它是该有限算法的精确导数。

`verify_exact_perspective_chain.py`仅检查上述局部解析因子+normal VJP对Torch AD，
以及六个常规root在−∞和条件原点两种定义下的IFT对独立brentq有限差分。
它不测试全renderer、GPU、真实raw共享累加、NCC image warp、GT质量或效率。
结果文件：`results/exact_perspective_chain_v1.json`。当前审查不改变任何已有生产路径。

小核验已完成：seed13018、96构造（24个cap激活），F/mean/query-ray/amplitude共1536梯度分量通过，
最大scaled error分别约(4.5315e−13,3.7230e−14,4.5570e−13,0)，门槛1e−8/floor1e−8。
六个非临界root的IFT对独立brentq中心有限差分通过，最大scaled error约1.55015e−8（门槛1e−6）。
其中−∞原点的两个root为负数；它们只检查数学定义，不能作为生产t≥0有效几何样本。
结果与源文件哈希一起保存；没有测试cap/clip边界、CUDA多ray atomic、完整NCC、tile支持域或真实质量。
