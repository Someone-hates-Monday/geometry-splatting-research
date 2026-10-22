# 完整24视角：192层原型容量不够，图像与实际贡献项计数均核验

2026-10-14。协议 `FULL_TOPOLOGY_PROTOCOL.md`，权威结果 `experiments/full_topology_v1`，独立审核 `results/full_topology_audit_v1.json`。这次测量真实完整渲染的名单长度，未将局部证书宣称为完整认证渲染。

## 范围与隔离

原scan37/55/63 iteration30000各8个既定视角，全部 **10,834,488** 像素。新的同源control与计数插桩同进程编译；插桩仅在原RGB真正接受位置及median首次迭代真正eligibility通过位置计数，记录非ball数与原initial eligible。无名单容量上限、截断或抽样；原生产源码未改。

每视角原RGB、alpha、normal、depth、slope、radii完整逐位一致，全部有限。RGB与median首轮的实际接受数在全部initial eligible像素一致，**零不一致**。保留原invalid depth=0；未回填。native forward范围绕过高层appearance后处理，两臂相同输入，不因此声称最终图像质量改善。

独立Windows CPU审核669来源、50产物、两臂各16生成源文件、实际binary与vendor header哈希及ZIP；检查生成源仅有计数/输出扩展，重算完整保存的depth/slope位一致、计数、名单容量和所有直方图。RGB/alpha/normal/radii在运行中完整比较，其数组未为第二次离线位检查留存，该审核限制明确记录。

## 全部名单分布

初始eligible **9,165,957** 像素，最终valid **9,145,921**。两组median接受数均min=3，p50=88，p90=155，p99=234，max=461。

- 原192容量遗漏334,083个initial eligible像素（3.644824%）；其中334,072个最终valid（有效像素的 **3.652688%**）。因此原`n<=192`求根接口不能直接用于完整图像。
- 256容量仍遗漏41,536个initial eligible像素（0.453155%），不能作为通用修复。
- 512覆盖本批所有名单，但这只是已观察24开发视角的上界，不是普遍场景保证；后续采用动态名单，不把本批max写成固定截断。
- 全部initial名单有870,178,922个primitive-ray关联；最终valid为869,143,824。这是跨24视角汇总，不能当成单视角同时常驻的内存。

按场景的8视角汇总：scan37初始max444、最终valid超192为153,403；scan55初始max382、超192为33,951；scan63初始max461、超192为146,718。scan63/0041有效像素超192比例10.6250%，不能被平均比例掩盖。

## 插桩成本与限制

3次预热后5对交替完整native render，CUDA event与wall、峰值allocated/reserved/added allocated全部保存。计数插桩/control配对比按视角再取几何均值 **1.025854787**，约2.59%额外时间。这是计数开销，既不是短证书的完整渲染成本，也不是训练加速。

原native完整时序与实际输入tensor、相机、PLY、生成源和二进制身份可复核。全部为此前已使用的开发视角，不能估计未接触数据集的像素发生率或作独立场景统计推断。

本批control native event中位数按视角为8.235072–21.230368ms，instrumented为8.521280–21.300928ms；这与110数学模型的245.7ms属于不同工作量，不能直接计算方法速度比。插桩新增peak-added allocated范围6,690,304–7,950,848字节。initial eligible/最终valid像素中没有rsigma<=0贡献项，这只是该固定名单上的观察。

首个离线审核在读取UTF8 CUDA源时触发Windows默认GBK解码错误，发生于像素审核前；失败记录保留。改成明确UTF8后审核通过，未改实验源、任何输入、统计、门槛或结果。CUDA主批complete/exit0(session11252)。

## 下一项

在实际first-iteration接受位置保存动态CSR primitive-ID名单，名单长度由本次真实计数的prefix-sum分配，逐像素验证写入数与原名单数一致，任何不一致返回显式失败而非截断。然后用可共享的原参数缓存和射线内并行计算接入认证/回退，固定完整24视角的初始范围、实际成本、unresolved策略与世界VJP门槛。完整认证、真实GT质量、训练收益和学术新颖性仍未完成。
