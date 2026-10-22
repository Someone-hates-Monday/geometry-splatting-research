# v8已实现真实渲染：局部门槛通过，整图深度门槛失败，禁止性能推广

2026-10-20，证据experiments/tgs_compensated_renderer_v8。不是观察器或训练模拟：
本轮已将tile粗排除、hi/lo Dot2、最大边紧区间、唯一active边误差接入隔离native渲染器，
同一helper供forward/backward使用。保留v5投影、原floatnormal/Jacobian、cap/cutoff、
显式RN合成、skip-current、两遍T区间分类与不确定ray重放。
原库、原始源码快照、checkpoint和当前LaTeX均未修改；没有optimizer/训练/GT评估。

## 通过的检查，以及未通过的完整门槛

四版本guard_diag/oracle_diag/guard/oracle均编译成功。SASS前向计数ADD分别6/4/0/0，
无计数版本保留同样缓冲并清零，不将静态指令数量当运行时间。
全部127保留案例、每版本70680828图像像素、四版2572个选定VJP通过冻结门槛；
counter twins的图像/路径/cache逐位一致，选定VJP差满足1e-5。
旧CPU/native在median阈值上的四个合成病例分歧继续保留，不能说所有CPU64 auxiliary一致。

5184条继承FD全部执行、支持集合稳定。独立归一化误差分位[0,50,90,99,100]%为
[3.5261e-8,7.6418e-5,.0016176,.0327395,2.7011004]；**不是全部FD精度达到1e-3**。
最差值比v5的1.3686大，未删除探针或把支持稳定当作FD数值精度通过。

六张完整相机9258/9322/9269/9221/9213，加输出前按SHA选到的新测试相机9326，
同stump7000、784056tri、SH3、1245x825。所有图RGB最大差<=4.173e-7，
贡献者/median索引、radii完全相同，median depth逐位相同，flagged端点逐位oracle。
固定RGB和weighted-depth VJP的六类数组均有限，最大组相对误差1.050625e-6，符合原1e-3门槛。

**整图仍失败：9213有14个weighted-depth像素差3.0517578125e-5，超3e-5。**
另五图最大depth差依次1.90735e-5/4.76837e-6/1.14441e-5/1.52588e-5/4.76837e-6。
14点全未触发重放，不能因为只有几个ULP、相对误差很小或新留出9326通过就豁免。
原两个早停异常回归像素9258(116,742)、9322(740,268)本轮均flag2，RGB/aux/path逐位oracle。

独立诊断审计保留all_gates=false、performance_measurement_allowed=false。
复算了全部barycentric图像、选定VJP、FD、源算术和native结果；2161个既有封存hash和
452个原源码hash未变。单独旧oracle/cache审计覆盖127病例全部747个可见边，
6real有理数抽查9789个边缓存（各1632，9213为1629），并独立检查全部可见domain/footprint。
新oracle的RGB/aux/coeff/旧10error与旧v5逐位相同；六相机旧v5 oracle VJP与新oracle
最大相对误差1.225754e-6，说明控制基线没有靠改变原数值前向来消除失败。

## 14像素因果追踪与精确误差分解

depth_failure_plan.json冻结了全部14个超差像素。独立CUDA trace使用完全相同的排序包和cache，
逐位复现14×2个guard/oracle原生端点，包括RGB、weighted-depth、alpha、median及last/median索引。
全部14点**没有early-stop**，支持/last/median均相同，不能归因于停止位置或贡献者改变。

在记录的float alpha、深度和权重上，Fraction精确证明每个端点差分解为三项之和：
1. 用记录alpha作精确实数合成的差；
2. T及blending-weight浮点舍入引入的差；
3. weighted-depth FMA累积舍入残差的差。
17692个RN递推/权重/累积检查通过，三项之和与观测端点差严格相等。
例如(772,1)：−.631426e-6−8.675802e-6−21.210350e-6=−30.517578125e-6。
这些是checkpoint深度数值单位，不擅自换算毫米或微米。
14点局部alpha最大差仅1.49e-8/2.98e-8/5.96e-8，真实alpha合成差绝对值最大约2.625e-6；
其余主要来自合成舍入。由“小的单窗口alpha差”不能推得这个未归一化AOV的固定绝对门槛。
有些点guard反而更接近其自身记录alpha的实数合成值，但这不构成GT质量提升，且不改变失败判定。

预声明七种回放全部保存，支持无变化：
- float-u改double power：14点均回到门槛内，6点depth逐位oracle，max差2.28882e-5；
- native-u转float但仍fast power：仍5点超差；
- float-u改标准powf：14点回到门槛内，4点depth逐位oracle，max差2.28882e-5；
- native-u转float再标准powf：14点回到门槛内，6点depth逐位oracle；
- 保留Dot2[p,s]、double重建phi/归一化/power：14点RGB/depth均逐位oracle。
这些是**已失败像素上的因果counterfactual**，不是完整渲染器修复、新留出或速度成果。
Trace无use_fast_math、ftz=false，显式__powf仍保留；原0/1端点逐位一致验证了这组输入的重现条件。

## 成本与后续选择

仍未计时。preflight使用double的数量明显下降，例如9258为1864/178063370，
最终render含整ray重放为376923/178132476。其余相机也保留原计数；不能据此宣布加速。
flag像素548/835/1182/1651/1844/837，比v5多，因为更多fast窗口携带非零alpha误差区间。
stable_error从10扩为48float/tri，增加152B/tri；实际geom378702912B，image32868308B。
比v5多119176448B（按有效元素152×784056=119176512B，布局对齐差64B），不是显存优化。

下一版不放宽门槛，有两个有证据支持的候选：
1. 只替换fast power为标准power，保留其他算术；需要明确调度以免fast_math重写，然后
   跑全127/5184FD/六完整相机和这14点，再加未用相机。已有六相机不再算新留出。
2. 以已测85.8%–87.5%的outside工作量为目标，FP32可证粗排除，所有未排除窗口仍使用
   原生double window/alpha，删除整图预检第二遍。alpha相同即可避免这次合成轨迹分叉；
   这是新的单遍候选，尚未实现/计时，必须用全部端点与VJP验证，不能凭设计宣布等价或更快。
后者也无需先采用暂存128/maxblend回滚方案；原生确定路径可直接提交。需实测两种设计成本后选择。

严格power区间、跨场景/GT/训练质量、学术新颖性及最终效率目标仍未完成。整体goal active。
编译71720、病例29003、binary72499、real95082、FD89810、baseline49195、总审计75781、
trace20669均已终止exit0；real工具exit0不等于门槛通过，JSON明确false。无等待作业。
