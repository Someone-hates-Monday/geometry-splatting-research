# 固定30k平方历史差pilot：完成，未通过采用门槛

2026-10-16。两组从严格42-view common6999完整状态重新起跑到30000，各23000次Adam，
共46000次；最终30000有反传/report/save，无Adam。16个队列阶段全部exit0，
包含两训练、两mesh/对齐、六官方GT采样评价、四TRAIN/TEST图像评价。
源代码、权重、相机划分与预设判据均未中途修改。没有训练仍在等待。

独立审核 `results/history_gap_pilot_audit_v1.json` 复核290来源、319输入、来源ZIP、
完整46000行状态/损失/ID记录、两final checkpoint+runtime sidecar的canonical digest、
模型PLY实际基元数、原始评价均值和全部门槛。audit通过不等于pilot通过。
两组所有23000个主camera序列完全相同；candidate独立Generator的每个128像素ID复算一致。
首次query/CSR/public bit对照通过；三private binary和训练control binary身份一致。
Adam/binding/finite检查每步执行，审计不声称独立重放了46000个浮点更新。

实际结果（control → weight.01）：

- 三个GT采样seed平均overall：0.3307163221 → 0.3315905070 mm，**恶化0.26433%**；
  d2s ratio1.00209755、s2d ratio1.00294454，三项几何门槛均失败。
- TEST7 global-RGB-MSE PSNR：28.07059016 → 28.07511820 dB，差+0.00452805；
  SSIM .899857896 → .900708420，差+.000850524。
  这两项通过预设非劣门槛，不是显著图像改善。不同PSNR定义不能混用：
  repository channel-mean TEST反而28.40820667 → 28.40738623 dB。
- TRAIN42 PSNR：35.54241317 → 35.46370216 dB，差-.07871101；
  SSIM .964257352 → .963615193。
- 基元928975 → 931140，ratio1.00233053；容量守护通过。
- 全训练进程wall2387.70 → 3151.18秒（约39.8 → 52.5分钟），ratio1.31975615；
  通过预设≤2倍资源上限，但约32%额外成本，没有训练加速。

预设需要**所有**门槛通过；最终pilot_gate_pass=false。
单一共享warmup配对无法断言普遍有害或统计显著；三个evaluator seed不是训练重复。
但它明确没有达到事先要求的≥1%几何改善，不把微小RGB变化当作成功。
按原协议停止这条正则的自动扩大训练和权重搜索；数学C1/数值实现结果单独保留。

## 只读事后检查：优化目标确实下降，但GT没有跟进

在所有42个train camera上每图512个相同uniform像素，两模型各21504射线，
同一实际query/CSR/loss链，只做no-grad，不修改模型，不使用新质量判据。
这是事后诊断，协议在该诊断前写入HISTORY_GAP_POSTHOC_PROTOCOL.md；不是预注册确认实验。

共同非空18344射线的平均D²：.1776675410 → .0931691978，下降47.5598%。
全采样均值.4284622530 → .3498538008；空射线比例14.2904% → 13.9927%。
全均值中空ray常数部分.2746339494 → .2689142706，非空部分（仍以全采样分母）
.1538283036 → .0809395302，两部分加和逐位数值复算。条件非空均值
.1794760914 → .0941077944；空→非空151、非空→空87、共同空2922。
native code2只称tail，不能当无根证书。

84份每camera NPZ的pixels/counts/empty常数/D²及所有均值已独立复算通过，
见results/history_gap_posthoc_audit_v1.json。实际输入是裁剪图，分辨率从真实PNG读取，
不是名义1600x1200；早先审计的硬编码尺寸假设已纠正，不改数据/模型/质量门槛。
这支持该pair中“目标下降与GT收益脱节”，**不识别因果机制**，也不证明改mask/权重会有效。

审计开发记录：初版错误地用Float64重构Float32训练loss，及把NCC调用次数等同23001反传；
已按实际Float32逐算复核。真实9001两组NCC/geo同时为0，与原PatchMatch empty-mask提前返回
规则及实际23000个NCC调用相容，mask本身未另存，不能伪称独立回放。candidate总objective
仍是原一元素Float32 tensor，加入零维Double正则后依PyTorch规则回到Float32；已精确复算。
初稿保留verify_history_gap_pilot_initial_f64_audit.py，固定实验门槛没有放宽。

## 下一项独立问题

先行文献定向核验见HISTORY_GAP_RELATED_WORK_2026_10_16.md。
不继续修饰本负结果。下一候选转向TGS自适应细分的光学量保持问题：先固定官方实现版本，
区分面积/支持域保持、alpha场保持、训练概率保持，检查是否存在真实图像扰动；
先做原公式及实际renderer复现，通过可证伪前置门槛后才考虑守恒细分方法或新的训练预算。
目前这仅是待检查问题，没有新TGS定理、新颖性或质量结论。
