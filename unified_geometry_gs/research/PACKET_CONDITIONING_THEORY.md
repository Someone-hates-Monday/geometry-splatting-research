# Packet 条件性、有限扰动界与量化不可辨识

2026-10-13。本工作为看过旧 case2 后的事后理论与诊断；不变更任何旧阈值或失败判定。
原 `clipped_packet_precision_v1` 的 1/128 失败仍然成立。当前文件不是新的质量/效率成果。
以下是标准三角不等式、商的扰动分析、均值定理与量化双点论证在本 packet 上的具体化；
不声称一般数值分析工具、Schur 分解或量化下界的新颖性。

## 1. 定义与范围

采用 `CLIPPED_PACKET_THEORY.md` 的精确实数定义。令

    a=qray_xy−qbar_xy, m=qmean_xy−qbar_xy,
    A=1+c·a, B=1+c·m,
    d=A²+aᵀKa, C=AB+aᵀKm, M=B²+mᵀKm,
    p=z r C/d, σ=r/√(b d), E=z²b(M−C²/d), α=u exp(−E/2), r=||qray||.

假设 z,b,r,σ>0、K 为 SPD2、0≤u<1。由 Schur 身份 E≥0。
所有上波浪号变量是另一合法 packet 的**精确**恢复值；浮点 decode 运算误差另计。
不假设两组 packet 保持相同 clip 分支；但比较其合法性时需要分别核验 SPD 与正尺度。
若舍入使 K 非 SPD，以下合法 Gaussian 结论不能继续使用，也不能用小 perturbation 界掩盖。
本节允许 qmean、qbar、qray 都变化，Focal/plane/width/conic 的变化通过恢复后的 c,K,z,b,r 进入。

## 2. 可证的有限扰动界

记 e_c=||Δc||₂、e_a=||Δa||₂、e_m=||Δm||₂、e_K≥||ΔK||₂；
取 k≥||K||₂，代码用 Frobenius 范数作为可计算上界。令

    E_A=e_c||a||+||c||e_a+e_c e_a,
    E_B=e_c||m||+||c||e_m+e_c e_m,
    Q_a=e_K(||a||+e_a)²+k(2||a||e_a+e_a²),
    Q_m=e_K(||m||+e_m)²+k(2||m||e_m+e_m²),
    Q_am=e_K(||a||+e_a)(||m||+e_m)
         +k(e_a||m||+||a||e_m+e_a e_m),
    E_d=2|A|E_A+E_A²+Q_a,
    E_C=|A|E_B+|B|E_A+E_A E_B+Q_am,
    E_M=2|B|E_B+E_B²+Q_m.

则 |Δd|≤E_d、|ΔC|≤E_C、|ΔM|≤E_M。
证明：逐项展开 Δ(c·a)、Δ(aᵀKa)、Δ(aᵀKm)，对每项用 Cauchy–Schwarz。
例如 Δ(aᵀKa)=2aᵀKΔa+ΔaᵀKΔa+(a+Δa)ᵀΔK(a+Δa)，其界恰为 Q_a。

若 d_−=d−E_d>0，令 q=C/d，则

    |Δq| ≤ E_q=(E_C+|q|E_d)/d_−.

这是精确身份 Δq=(d ΔC−C Δd)/[d(d+Δd)] 的直接结果。
令 e_z=|Δz|、e_r=|Δr|，E_s=|z|e_r+|r|e_z+e_z e_r，则

    |Δp| ≤ |zr| E_q + E_s(|q|+E_q) = E_p.

这里**绝对**峰位界在 C=0 时仍有定义；除以 |p| 的相对界可能发散。
仅从 SPD 和正尺度不能推出统一的峰位相对精度保证。

宽度方面，令 e_b=|Δb|、U=bd、E_U=|b|E_d+d e_b+e_b E_d。
若 U>E_U 且 r>e_r，则

    σ_−=(r−e_r)/√(U+E_U) ≤ σ̃ ≤ (r+e_r)/√(U−E_U)=σ_+.

这同时给出 |Δσ| 与 |Δlogσ| 的有限区间界。若下界条件不满足，则界为**无结论**，不是算法失败或通过。

令 β=M−C²/d≥0，S=z²b。由相同商身份，

    |Δβ| ≤ E_β=E_M+[2|C|E_C+E_C²+C²E_d/d]/d_−,
    E_S=(|b|+e_b)(2|z|e_z+e_z²)+z²e_b,
    |ΔE| ≤ |S|E_β+E_S(|β|+E_β)=E_E.

虽然此**误差界**使用平方差身份，实际 E 的求值可继续采用原 Gram 非负分解；
两者不能混为同一种浮点算法。若 e_u=|Δu|，用合法性 Ẽ≥0 得

    α_−=max(0,u−e_u) exp[−(E+E_E)/2],
    α_+=(u+e_u) exp[−max(0,E−E_E)/2].

有 α̃∈[α_−,α_+]；因此 E_α=max(|α_−−α|,|α_+−α|) 为有限扰动界。
当能量很大、E_E 相对较小时，α_+仍指数地小；这可严格区分峰位相对失败和透射率相关性。
这不准许回头删除原失败输入。

## 3. qmean 单独变化与 cross 接近零

固定 plane、conic、width、qbar、qray、focal，qmean=(u,v,1)。记 γ=dz/z，则

    γ=−(u du+v dv)/(1+u²+v²), db=dr=da=0,
    dc=−(c−qbar_xy/||qbar||²) γ, dK=−2K γ, dm=(du,dv),
    dA=a·dc, dB=m·dc+c·dm,
    dd=2A dA+aᵀ(dK)a,
    dC=B dA+A dB+aᵀ(dK)m+aᵀK dm,
    dp=(zr/d)[dC+C(γ−dd/d)].

对 C≠0，可写 dp/p=γ+dC/C−dd/d，显式暴露 cross cancellation 的相对敏感性。
这解释为什么部分样本保留 qmean 精度有帮助，但不证明保护 qmean 普遍有效。
绝对偏导在 C=0、d>0 时仍有限；1/C 只出现在相对精度分析中。
中心参数变化引发 z,c,K 的联动，不能只把 qmean 看成峰位中的一个独立线性输入。

## 4. 从局部参数界到 logT 的全射线界

单 Gaussian 的累计光学厚度 h(t)=−log T_i(t) 是

    g=α exp[−(t−p)²/(2σ²)],
    h=−½log(1−g)                  当 t≤p,
    h=−log(1−α)+½log(1−g)         当 t>p.

两支在 t=p 处关于 α,p,logσ 的一阶导数吻合。
设 α,α̃≤ā<1、σ,σ̃≥σ_min>0，则对**所有实数 t**有

    |h̃(t)−h(t)| ≤ |Δα|/(1−ā)
      + ā |Δp|/[2√e (1−ā) σ_min]
      + ā |Δlogσ|/[e(1−ā)] = ε_i.

证明：沿 (α,p,logσ) 的线段用均值定理。使用
max_x |x|exp(−x²/2)=1/√e、max_x x²exp(−x²/2)=2/e，
且 1−g≥1−ā。跨过峰位时 C1 接合保证可用同一界。
取 ā=max(α,α_+) 和 σ_min=min(σ,σ_−) 即可由第2节得到事前上界；若 ā≥1 则当前界无结论。

有限相同基元集合的 H=Σh 满足 sup_t|H̃−H|≤Σ ε_i=ε。
如果使用相机原点条件化 H_cam(t)=H(t)−H(0)，则界为 **2ε**；不能忘记原点项。
增删基元、不同 cap 或不一致的集合，需要再加相应基元的 h 界，不能直接套用。
任意删掉单基元可取其总权重 −log(1−α) 为全实线 h 上界。

若精确基准中位数 t* 满足 H_cam(t*)=log2，在包含 t* 的区间 I 上
H_cam′≥m>0，且 t*±2ε/m∈I，则候选在这些端点跨过 log2，从而

    |t̃*−t*| ≤ 2ε/m.

对 −∞ 原点则为 ε/m。等价地可直接核验端点有大于2ε的符号余量。
必须另外满足相同合法有限集合、连续单调映射和候选有限根的前提；
端点跨越本身可建立区间内根存在。临界根 m=0、总透明度不足等情形不获此界。
本文件没有把逐 packet 数值检查升级为真实整场景中位深度证书。

## 5. 同一 FP32 输入 cell 的双点论证

仅说“提高已舍入数据的 decode 精度仍失败”不能证明任意 decoder 不可能。
更强而范围明确的结论需要两个合法输入 X1、X2 满足 round32(X1)=round32(X2)，
但正确峰位的原接受区间

    I_j=[p_j−τ max(10^−8,|p_j|), p_j+τ max(10^−8,|p_j|)], τ=10^−5

不相交。任何仅接收该相同 FP32 packet 的确定性 decoder 输出相同，不能同时属于两区间。
随机 decoder 也不能对两者都以概率1满足同一保证。
这证明**此表示在所给合法域上不能统一达到该峰位精度门槛**，不是所有 Gaussian 实现或所有下游任务不可能。
额外 metadata、保护精度、收紧输入域、或改变另立的下游相关性目标都不在排除范围。
这里的表示明确是当前八输入原型中 **qmean自身round为FP32** 的schema。
真实renderer保存UV后恢复qmean的运算次序、共享约束或其他metadata布局没有被此双点论证覆盖。

脚本选旧 case2 的舍入 cell，对 qmean 前两坐标各取相邻 binary32 中点向中心缩进2^−12的四个角。
这些角是可精确表示的 binary64 dyadic；其余 packet 量固定在合法 SPD Schur 输入，qbar仍是原 .65 clip。
用精确 Schur 逆可得到各自合法 P、μ；并非从真实数据另发现两枚基元。
使用 mpmath.iv 100位区间独立计算正尺度、SPD 与两个峰位，核验接受区间严格分离。
mpmath.iv 1.3 为实验性库，区间算术实现是明示的可信依赖；没有声称形式化机器证明。

## 6. 可复现检查与限制

命令（WSL 项目根目录）：

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/someone/miniconda3/envs/uggs/bin/python -m research.verify_packet_conditioning --out research/results/packet_conditioning_v1.json --include-real

脚本不引用候选 production recover，独立以 mpmath 100位重算原 double packet 与 float32-rounded packet；
对全部128旧构造和可选全部1992真实 selected输入检查有限扰动界，在六个 t 点抽查 logT 界。
六点抽查不能证明全域界；全域结论来自上述解析推导。
一般 bounds 的数值检查不是区间证书；只有明确标签的双点量化反例使用区间外包。
所有新结果保留 posthoc 标签。不启动训练、不改原 renderer、不作CUDA/图像/几何/速度或新颖性完成声明。

首次全部检查得到：5个符号身份通过；全部128构造+1992真实输入的有限界有结论，
0个数值界检查失败。真实输入最大的峰位相对误差上界约8.2575e−7。
旧 case2 的100位精确 decode 舍入误差约1.71395123e−5；有限峰位相对上界约4.91157837e−5，
其中 cross 相对扰动界约4.89297e−5，den 相对扰动界约1.60554e−7。
该 case2 的 nominal α_+约2.30633e−5712、全实线单基元 logT 界约5.84306e−5715；
这些是**高精度数值界评估而非区间证书**，且不构成真实全场景中位深度或可见质量证明。

双点区间核验得到两个峰位约−.028399275821260331与−.028398215071580236，
原1e−5接受区间之间仍有约4.92774771e−7的严格正间隔。
FP32输入逐字段字节相同、qbar是两均值的合法clip，正尺度和SPD均通过区间核验。
接受区间端点的最后算术也使用区间运算和精确有理数τ；结果保留精确binary端点。

一次脚本读取错误把共享focal当成逐输入数组，已修正并保留失败日志；没有改数据或门槛。
添加精确binary端点时曾因mpz不能JSON序列化而在保存前退出，已转为Python整型并保留日志。
首次成功产物作为 `packet_conditioning_v1_pre_endpoint_interval` 保留，
最终结果 `results/packet_conditioning_v1.json` 增加了接受区间端点的完整区间外包，
不会覆盖任何旧科学实验。
