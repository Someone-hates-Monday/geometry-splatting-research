# Exact perspective 冻结离群射线：表示根与数值路径

2026-10-14。此方案在观察到24幅冻结图像中的B/C离群差后制定，属于事后机制诊断。
不更改旧门槛，不把9个选择后的样本用于估计普遍发生率，不启动训练。
oracle的对象是实际保存的FP32 (alpha,peak,rho)及实际prefix交集，不能自动等同原raw世界Gaussian。

## 1. 精确质量先于任何数值求根

alpha_i为保存的有限binary32实数，0≤alpha_i<1，rho_i>0，peak_i有限。
使用 `Fraction(float(alpha_i))` 精确构造二进制有理数，令 β=∏(1−alpha_i)。
由于rho_i>0，任何非零alpha对应有限正宽度的连续GGGS层。

- β<1/2：从−∞原点定义的T(t)有唯一有限中位根。
- β=1/2：只在t→+∞趋于1/2，无有限根；属于总质量临界。
- β>1/2：总质量不足，无有限根。

这些判定不依赖mp log的精度或F32累乘，也不能被原CUDA正深度返回覆盖。
若CUDA输出正深度但同保存F32系数的β≥1/2，应报告该表示下false finite，而非强制brentq生成根。
若所有alpha为0，则β=1，亦在不足分支。

## 2. 稳定残差与真实斜率

记x_i=(t−p_i)rho_i、g_i=alpha_i exp(−x_i²/2)，近侧厚度b_i(t)=−½log1p(−g_i)。
则H=−logT，分支为t≤p_i时b_i、t>p_i时−log(1−alpha_i)−b_i。
真实解析斜率是

    H_t(t)=Σ g_i |x_i|rho_i/[2(1−g_i)], −T_t(t)=T(t)H_t(t).

报告根处及实际CUDA输出处的真实H_t、−T_t和残差；
保存输出slope是原最后bracket割线，不能把它小直接解释成true slope小。

为稳定求R(t)=H(t)−log2，对所有p_i<t的项构造精确有理数
β_left=∏_{p_i<t}(1−alpha_i)。然后

    R(t)=log1p[(1/2−β_left)/β_left]
         −Σ_{p_i<t}b_i(t)+Σ_{p_i≥t}b_i(t).

平台常数的减法先在Fraction中完成，可保留极小非零质量裕量；
β_left=1/2时常数严格为0，极薄尾部之间的比较不会被两个O(1) log差相消掩盖。
mpmath任意指数可保留极小尾部，但仍只有有限有效位数；中点残差数值等于0不视为精确根证书。
脚本用严格异号数值括区间持续收缩；若更内侧端点也无法辨号，保留precision_plateau_unresolved。
T−1/2按expm1稳定形成：exp(−R)/2−1/2=expm1(−R)/2。

MP brackets为高精度数值证据，未作向外舍入区间证明。
真实根附近的1/H_t只给局部敏感性；没有统一导数下界时，不把residual/slope写成全局误差界。
薄层已有反例说明深度极限与梯度极限不能互换；本次不能把理想薄层反例直接当真实离群原因。

## 3. 负深度、原点条件化与原搜索区间

原render搜索截在t≥0，但它乘的是T_from_minus_infinity(t)，并未除T(0)。
β<1/2只能保证全实线有限根，不保证其位于正区间，更不保证位于给定mDepthinit±.4。
必须另报告T(0)、全实线根、初始区间端点余量与原CUDAvalid状态。

相机原点条件化模型为T_cam(t)=T(t)/T(0)、t≥0。
其有限根条件是β/T(0)<1/2；由于T(0)涉及指数，这一条件本轮是MP数值判定，
不得冒称Fraction总质量证书。oracle单独计算这个不同模型作为范围诊断，不把它代替原模型。

## 4. 分解与可证明的边界

仅当B/C同保存F32系数都存在已解析的有限根时，定义

    Δobserved = Δrepresented_roots + Δremaining_numerical_path.

Δrepresented_roots区分二者系数表示导致的数学根差。
余项包含有限SPLIT8/5轮、乘积求值、fast math、FMA、插值、输出z转换和probe上下文等，
不能在缺少复刻证据时直接命名为“FP32累乘误差”。
任一臂无有限根或oracle无法辨号时，分解是undefined，必须保留这项结论。

ray-depth由保存z-depth除probe rln得到；probe rln不自动等于原kernel内部值。
脚本附带仅考虑最后z乘法舍入的条件区间，明示其依赖rln匹配假设；这不是完整转换误差证书。

## 5. 最小复刻对照

所有复刻保持同一保存coefficients、prefix和初始区间，SPLIT8、5轮：

1. MP全算术/稳定残差的有限搜索：量化有限搜索相对同表示真实根的行为。
2. double稳定log残差搜索：区分通常的精度限制；不能在数值平台上任意指定真根。
3. double顺序乘积搜索：观察乘积表示与稳定残差的差异。
4. CPU NumPy F32顺序乘积搜索：单独标记approximation，不声称复现CUDA expf/rsqrtf/FMA指令。

原搜索在网格T≥.5时总选最靠右的合格点；数值T=.5平台可能改变选段。
最后端点T相同导致0/0插值时，脚本保留rounded_flat_endpoint_values，
不猜测CUDA saturate NaN语义或把CPU NaN当作原CUDA输出。
最终原内部bracket／endpoint未被原buffer保存；capture初始化也是复制表达式probe，而非原始instruction trace。

接收格式与用法写在 `exact_outlier_root_oracle.py` 顶部。
完整capture输入必须提供actual prefix导出的accepted_mask；oracle不重新用double cutoff替代原mask。
selftests覆盖精确质量临界／不足、单层闭式根和极薄双层平台，并保留失败产物。
待真实capture落盘后，全部选择出的射线逐项运行，不追加或删除有利样本。

## 6. rebuilt capture 的继承限制

实际输入固定为 `experiments/exact_outlier_capture_v3`，原历史binary SHA无法字节重现的失败保留。
v3使用单独固定的新binary SHA；全部8受影响视角的两臂depth/slope/RGB/alpha/normal对旧存档F32逐位相同。
这支持新binary输出与历史观察等价，不证明原binary恢复、旧隐藏buffer相同或同指令执行。
本oracle读取的是新固定binary的真实buffer prefix与复制表达式probe系数／初始化，继承以上范围。
分析前核验capture全部源/archive、产物、模型、backend生成源和新binary、probe源和binary及8视角重放标记。
accepted_mask另与实际n_contrib、rank和affine eligibility交叉检查，accepted便利数组与probe列按字节核对。
真实分析默认只含packet/factor两臂（9射线、18输入）；可显式 `--include-affine` 加入独立参考，不能混淆数量。

    python -m research.exact_outlier_root_oracle --capture research/experiments/exact_outlier_capture_v3 --out research/results/exact_outlier_root_v1.json
