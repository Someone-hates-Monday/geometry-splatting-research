# 实际GaussianModel接入：共享梯度与短步控制通过，质量与完整训练未证

2026-10-15。上一阶段warp优化为实质进展；本阶段已完成actual query/AMP/CSR GPU采集、
sampled Float64 raw图、回写原共享Float32参数，以及三个原场景内存中36次fresh-Adam更新。
不是完整NCC/normal训练、不是已恢复原训练optimizer、不是GT或留出图像评价。

三个30k原PLY场景37/55/63各camera0000：public renderer与private query/CSR的RGB/alpha/normal/
depth/slope/radii逐位一致，probe源/raw/ID/sections逐位对上旧输入。采样数据保持GPU，仅诊断产物另拷贝保存。
正则mean对原参数的shared-ID Float32梯度与独立MP损失梯度聚合相对L2分别
4.682502111901933e-8、2.561310135244713e-8、3.844083955840648e-8，各块最大≤7.392e-8，
均过原整射线1e-5/块1e-4门槛；非目标appearance/color/SG参数没有正则梯度。
第一版检查停止于原模型非有限参数，第二版误取median-depth而非history-loss参考，均未进入optimizer更新。
失败sourceZIP/manifest/traceback保留，V3正确选择独立loss参考并核对映射，不改核或梯度门槛。

原PLY输入全量核验发现opacity分别3431/1908/2443个+inf，无opacity NaN/-inf，无xyz/scale/quat非有限。
V3不替换/删除/修复这些原值，显式保持为冻结opaque坐标：sigmoid(+inf)=1，梯度必须严格0，mask和值不能变。
其余参数、所有梯度、Adam状态全有限。初版all-finite参数gate确实失败，没有改称通过；
这是兼容真实基线的明确修订，不是在无穷logit处主张可微，也未证明这些原值形成原因或导致旧质量失败。

128独立Generator像素采样联合原probe，最终164/162/167射线，这三批实际empty均0。
scan37另有真实全空ray，loss=1.9218120556728056并保持零梯度。补充saved-real-input replay fixture把36非空加1空：
mixed loss分母37正确（误差5.551e-17），梯度按36/37缩放（误差8.283e-8），全空梯度精确0。
fixture不是新live场景；另外两图没有真实empty可测试，不能说每个场景都现场验证了空分支。

两次disabled、computed-zero、lambda.01，每臂3步/场景，总36次更新；固定拓扑/filter、相同camera/采样，
每臂前恢复相同实际parameter对象和fresh-Adam state，binding始终核验，结束原内存参数逐位恢复、原PLY SHA不变。
adapter不消耗Python/NumPy/Torch CPU/CUDA的训练RNG。零控制按预设max(1e-5,5×control-repeat偏差)工程界通过；
不能称bitwise或统计等价。scan37 scaling梯度零对照差1.0542%，disabled重复本身.6782%；
其余报告全部组差异，未把重复波动归因新方法。三场景zero参数max相对差2.783e-6/8.425e-6/4.261e-6。

positive前三次更新前reg观察值：37 .163931→.150108→.138692，55 .244935→.233780→.224625，
63 .123730→.113413→.104279。最后RGB混合损失positive为.02437155/.01616110/.01104335，
disabled为.02435276/.01614098/.01102864，positive略高。只能说明正则生效及局部tradeoff，不能声称重建/图像改善。

同步diagnostic update wall中位（ms）disabled重复70.71/47.25/31.30，computed-zero128.20/87.92/69.77，
positive111.31/85.66/64.38。positive完整capture42.32/30.51/24.20，prepare/pack/loss6.36/5.37/5.92。
包括所有private count/query/CSR scan/capture/parse、同步与finite/binding检查；不含产物CPU拷贝/写盘及各臂状态恢复。
最初Adam冷观测2459ms原样保存。不能把之前loss组件11.99ms说成真实训练开销。

主V1 session89078 exit1，V2 session30049 exit1，V3 session22299 exit0；empty fixture97796 exit0；
独立verify_history_gap_adapter session45504 exit0，20来源/35产物、全部回放/MP loss scatter/分母/mask/controls/steps核验。
cfg/输入PNG/标定/实际public binary附加哈希在独立报告记录，postrun读取且mtime早于run；原model文件hash始末核对。
见results/history_gap_adapter_audit_v1.json；原LaTeX原位编译，生产源码/原模型没有修改。

下一步接入原完整train.py loss，使用先前42/7严格划分的common6999 checkpoint+runtime sidecar，
先核验完整状态恢复/实际训练camera与采样序列、disabled/computed-zero控制及短步，随后预设权重和预算做配对pilot。
当前30k PLY用过49视图，不得拿它们的微调图像当真正held-out结果；也不因本smoke损失下降直接采用为学术质量成功。
原147-view冻结GT失败结论和所有先前负结果保持；总体目标仍未完成。
