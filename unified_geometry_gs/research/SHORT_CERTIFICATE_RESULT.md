# 短级数充分认证：固定开发再加速，严格回退与未解分支保留

2026-10-14。主协议 `SHORT_CERTIFICATE_PROTOCOL.md`，权威主批 `experiments/short_certificate_v1`，独立审核 `results/short_certificate_audit_v1.json`；另有 `SHORT_FALLBACK_PROTOCOL.md` 和 `experiments/short_fallback_v1`。

## 实现与数学边界

原整数倒数表、参数预算、质量分类、候选、low/high顺序和tau0容差保持。只增加一层带有向余项界的短证书：log 9项atanh级数余项≤2z^19/[19(1−z²)]；exp 12阶余项≤3r^13/13!；|x|<1/8时log1p/expm1 8阶余项分别≤r^9/[9(1−r)]与r^9/[9!(1−r)]。域外与极小尾部处理继承旧核心。

这些是标准级数界，不主张新数学方法。只有候选两侧严格相反的区间符号才能short返回；无法确认时立即使用原24/25项严格证书，仍失败再按原逻辑尝试high；最终仍失败输出unresolved。代码分开gi/gf命名空间并显式限定函数，防止同类型参数的ADL选择长/短版本。设备未接收CPU候选、端点或参照。

## 完整主批核验

- 原7322层、110模型全保留；预算与重编control/历史逐位一致，control根与历史逐位一致。
- 774已保存CUDA scalar界均包含独立Decimal240参照，第二个MP240 elementary oracle也全部通过。
- 全106有限根认证、4无根分类正确；最终106均short，76 low / 30 high。**strict仍被调用30次**：这些是low候选的short与strict都失败后转high，不能称零回退。
- 最大根误差仍为1.361838888e−5 camera-ray单位，不能解释为world mm或GT。
- 独立MP240世界covariance逆复核106新端点对、4408使用中系数、scaled残差包围、精确Fraction宽度和4无根全部通过。16来源、8产物、23生成文件及ZIP核对完成。

主批全部属于已观测开发数据，不称独立未接触输入确认，也不能估计真实像素发生率。独立审核的 `local_adoption_gate_pass=true` 限定于本数学诊断。

## 与倒数表对照的配对效率

同进程重编原倒数表control和单一short变体，严格CUDA选项相同。预热后5对交替顺序，event计时含输出allocation/fill；排除host输入传输、完整renderer、backward、GT和optimizer。

- root/证书 event中位数：384.782196 → 245.457794 ms。
- 预算+root合并 event中位数：**385.132538 → 245.741669 ms**。
- 合并配对比几何均值：**0.638154993**，耗时减少 **36.1845%**，速度比约 **1.56702**；五对比约0.637992–0.638375。
- 所有预定数值检查成立且比≤.95，局部门槛通过。不同阶段的705/385/246ms不能当成同一运行内的三臂对照；此处明确报告同进程的385→246配对。

## 单独的平坦探针：成功回退与不可认证均实测

主批没有最终strict成功案例，因此新增单独固定数学压力族，不混入主批采用/计时：单基元mean=(0,0,1)、unit camera/rotation、amplitude=3/4、scale=10^j，j=0..7，解析median为1。只加载主批实际冻结二进制，解析根仅用于返回后的核验。

control认证4/8，另外4 unresolved；variant有3 short、1 strict、4 unresolved。全部返回证书均通过独立MP240单基元公式、scaled包围、精确宽度与解析根包含检查，无假成功。严格成功分支已覆盖。未解4例不能视为整体可认证或推广成功；完整renderer必须明确处理未解案例。

## 下一阶段

246ms仍是严格数学批次成本，未证明完整图像或训练效率。下一阶段固定完整24视角的生产选择/初始范围与成本协议，检查可扩展的缓存和射线内并行计算；独立世界VJP通过后才进入跨场景配对训练与held-out图像/几何评价。极平坦族的精度限制、临界opacity梯度问题和此前质量失败保持。原生产CUDA/训练源码未改变。
