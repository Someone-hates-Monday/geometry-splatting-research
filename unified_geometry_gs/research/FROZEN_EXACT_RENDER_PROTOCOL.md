# 冻结模型精确透视三臂：先完整图像前向，再GT

2026-10-13，写于完整图像三臂smoke/性能结果观察前。已有1992局部生产FP32输入通过，
但FP32合成失败和两个精度候选失败保持。当前不是完整训练方法或新颖性确认。

## Phase A：既有模型全部24个预定开发视角

使用原median_cpu_scan37/55/63.json记录的8个视角，完整原分辨率、全部像素、iteration30000。
先核验原PLY哈希，Scene在scratch中读模型，不改原模型元数据。
三臂为新同源JIT affine、exact_packet、exact_factor；不把已安装二进制差异归给数学改变。
单视图smoke用于编译/接口错误定位，同一方案后续完整24视角是开发筛查，不是独立场景确认。

范围严格限于median T functional under legacy affine selection：保留原RGB、alpha、normal、
初始depth、排序、affine eligibility/cap/prefix、SAMPLE_RANGE和5次求根。精确臂在既定基元列表中
只替换沿射线p/rho/alpha，B/C共用规则。没有相机原点条件化、保守exact支持域或完整法向场替换。
所有RGB/normal不变是因素隔离，不等于完整exact renderer。sample/NCC/integrate/backward明确禁用。

完整图像数值开发门槛，全部24视角都需满足：

1. 所有输出有限；depth、alpha不为负；保留depth=0作为invalid，不填补、裁正或删像素。
2. B/C相对affine的RGB/alpha/normal最大绝对差<=1e-5；记录实际值，不只给是否通过。
3. exact_packet与exact_factor的有效深度mask不一致比例<=1e-4（0.01个百分点）。
   共同有效像素depth绝对差p99<=1e-4、max<=1e-3，均为模型世界长度单位；
   这不是相对GT误差或严格逐像素证书，临界根与有限求根误差仍可能影响结果。
4. 报告每臂对affine的有效覆盖、共同有效深度位移全部分位数、mask翻转位置/数量。
   报告B/C数值失败时停止GT改善解释；不能把支持域/初始化问题归给精确代数。

时序采用seed13018随机化，每视角每臂5次预热，3轮各10次render；GPU event与同步wall均记录。
记录每臂峰值allocated/reserved显存、生成源/二进制哈希；编译与Scene加载另计，不混作kernel收益。
每轮同一GPU串行，禁止多个agent同时计时。FP64 packet decode与F32 direct factor精度差异明示。
重复次数/像素不是独立场景样本，不作统计显著性。完整render含预处理、排序和求根，但不含训练backward。
效率路线只在B/C输出通过时考虑：packet/factor端到端render GPU时间几何均值<=.90，
且任一视角比<=1.05；该门槛是本次开发采纳条件，不证明训练加速。

首次smoke完成后、PhaseA完整24视角执行前，追加斜率通道差异的描述性汇总：
smoke已观察到B/C割线斜率存在差异，不能把深度接近写成梯度等价。
这项描述指标并非观察前登记，不改变原深度/有限性/时间门槛，也不另声称斜率通过。

## Phase B：仅Phase A数值通过后预定完整GT对照

同scan37/55/63已有模型全部训练视角，同原TSDF参数(voxel .002、trunc4、depthmax8、cluster1)，
相机ID配准到官方DTU标定；统一采样种子和官方几何度量。每臂保留有效深度覆盖与全部mesh产物。
主开发门槛沿用平均overall至少改善1%、s2d不增；任一场景overall恶化>1%或覆盖率下降>.1个百分点
则不进入训练。质量路线完整forward额外成本最多10%；若只有质量改善，不称效率改善。
这一阶段尚未执行；旧affine支持域下的对照只能隔离系数效应，后续保守exact支持域另立方案。
若数值、GT或成本均不能支持优势，停止扩大packet优化训练，保留负结果，优先考察薄层理论的真实机制。

## 反向与完整目标

本版本不满足完整反向、共享基元梯度或NCC一致性，不能用于训练。要进入训练，需另行实现
per-ray(alpha,p,rho) VJP、几何raw参数链、root梯度与sample/NCC的一致定义，并按独立参照检查。
真实跨场景训练、消融、留出RGB质量、独立重复及原论文新颖性核查仍是长期目标的未完成项。
