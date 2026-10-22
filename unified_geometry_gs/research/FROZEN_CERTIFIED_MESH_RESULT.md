# 完整49视角冻结模型质量对照：数值通过，预注册质量门槛失败

2026-10-15 Asia/Shanghai。主批frozen_certified_mesh_v1 session33905 exit0；独立审核session29600 exit0，results/frozen_certified_mesh_audit_v1.json。原模型PLY、生产训练/renderer源码不改。

scan37/55/63各完整49视角，147视角、66361239像素、5346733664原接受关联。逐视角与原Gaussian renderer对比RGB/alpha/normal/radii/depth/slope全部位一致；CSR完整write/check通过，cache状态有效。原有效56594143像素，已有充分证书52932318；其余3661825中认证修正3659782、数学无根2043、最终未解决0。应用同一gt_mask后两支有效像素都24152107；这些无根失效点没有改变掩码后覆盖数，不能把质量差异归因于减少可见评估像素。

两支同RGB白背景、同mask、intrinsics/extrinsics、视角顺序、TSDF(.002/1/8/trunc4/block50000)、largest cluster1。完整0000--0048按官方相机索引对齐，两mesh共享同一transform。每scan每支seed0/1/2官方评估，共18次，属于同模型评估抽样重复，不是独立训练或置信区间。

三个seed的overall均值（mm）：scan37原0.4605930275、修正0.4605502386；scan55原0.3139134531、修正0.3138697150；scan63原0.7369190101、修正0.7369960982。三scene平均候选/对照=0.9999937549058923，仅约0.0006245%下降，未达到<=.99的预注册1%门槛；三scan的s2d均值delta分别+0.000023822981、+0.000033254043、+0.000505425888mm，也全未过<=0门槛。不得将微小数值变化称为显著改进。

独立审核：97来源，实际原/私有binary，所有147数组、原输出位、掩码/保留像素/修正/无根代码与证书端点计数、6套mesh文件SHA及18评估源/结果全部核对。完整源关联计数不等于53亿逐节点MP真值核验；既有库/固定探针保证和原IEEE算术假设范围仍保留。主源zip不内嵌官方GT数据，以SHA引用；模型/mesh/全量图像在工作目录另存。

三scene wall约515.86/299.24/369.96s，包含render/capture/修正、存储、CPU TSDF、extract与评估等待；所有view wall之和309.54s。全部GPU event/CPU成本留在每view JSON，未用单kernel时序冒充整体加速。没有训练/图像质量/新颖性提升结论。

结论：该后处理保留作数值参照与诊断，当前不作为已验证的重建质量优化采纳，也不据此启动训练。目标尚未完成。后续需研究能改变学习目标/几何一致性的方案；新数学候选与反例/核验计划单列HISTORY_GAP_REGULARIZER_CANDIDATE.md，尚未证明或训练，不与本已完成失败试验混同。
