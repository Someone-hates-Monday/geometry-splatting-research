# 研究脉络、证据与未解问题

2026-10-22 收尾快照；先读 [交接入口](../HANDOFF.md)。以下是接手路线，不替代原协议或封存输出。没有在本轮重新训练、重跑 CUDA 或逐一复核全部历史结论。

## 1. GGGS 消融与证据清账

[MATHEMATICAL_AUDIT.md](../unified_geometry_gs/research/MATHEMATICAL_AUDIT.md) 与 [原始审计 JSON 副本](evidence_receipts/unified_geometry_gs/research/results/evidence_audit.json) 是入口。15 个 DTU 扫描只有 11 组成对结果，缺 scan69／106／114／118 基线；成对 10 好 1 差，mean overall 差 -0.0295654011，排除原 3 个 pilot 后其余 8 个约 -0.03164。原 3 个 pilot：去 depth-normal 项 mean 差 -0.00377814、训练 PSNR +0.04101 dB；再使用 trunc4 的几何差 -0.02033675，说明提取配置也贡献了变化。

这些记录是已有实验的审计，不是重新做齐的严密 15 场景消融。训练视角 PSNR 不能当留出泛化；配置、容量和提取条件必须随结论报告。`unified_geometry_gs/docs/RESEARCH_PROTOCOL.md` 保留 P1–P5、τ 曲线等早期流程；`output_compare/metrics` 保存早期失败探索的指标。部分历史模型在此前已经清理，本轮又有明确范围的中间检查点裁剪，见 [CLEANUP.md](CLEANUP.md)。

## 2. 中位深度、NCC、方向场与数值路径

[贡献审计](../unified_geometry_gs/research/CONTRIBUTION_AUDIT_2026_10_13.md) 指出与 GGGS／2DGS 已有思想的重叠，不能把重述公式当作新贡献。`median_depth_note.tex` 是原位理论笔记，本轮没有改写或编译。顶层 `*_RESULT.md`、对应协议、`check_*.py`、`.cu` 及 `results/` 组成详细路径，见 [全索引](RECORDS_INDEX.md)。

主要线索包括：稳定化 NCC 的精确导数、median／quantile 方程与隐函数导数、薄极限条件、方向场及 world-affine ray、有限透视与裁剪包、根区间证书和 guarded precision。数学假设、实数恒等式、FP32 实现误差、几何质量、可发表新颖性是不同问题。

[NCC pilot](../unified_geometry_gs/research/NCC_PILOT_RESULT.md)：导数修正通过核验，但 scan37 overall 约退化 0.01119 mm，训练 PSNR +0.155 dB；训练轨迹不同，不能严格归因。

[冻结完整渲染](../unified_geometry_gs/research/FROZEN_EXACT_RENDER_RESULT.md)：24 视角仍有 9 像素误差大于 1e-3，最大约 0.17964；packet/factor GPU 时间比约 2.60018，没有达到质量／速度采用条件。端点证书或局部求解通过不等于全图一致或完整渲染加速。

## 3. 完整训练的负结果

[factor pilot](../unified_geometry_gs/research/FACTOR_PILOT_RESULT.md)：scan55 从共同 6999 步到 30000 步，两分支各 23000 Adam 步；14 阶段 exit0。overall 0.315025878 → 0.313129859 mm，改善 0.601861%，未达预设 1%。训练 PSNR +0.532053 dB、SSIM +0.003128499，容量 +2.50636%，没有重复种子或留出证明。

[factor 留出实验](../unified_geometry_gs/research/FACTOR_HOLDOUT_RESULT.md)：新划分 42 train／7 test；TEST PSNR +0.090075357 dB、SSIM -0.000304571，未过 +0.30 dB 且 SSIM 不降门槛。

[history-gap 最终结果](../unified_geometry_gs/research/HISTORY_GAP_PILOT_RESULT.md) 才是最终状态；`HISTORY_GAP_FULL_TRAIN_RESULT.md` 是更早的启动快照。2026-10-16 两个 30k 训练共 46k Adam 步、16 阶段 exit0；overall 0.3307163221 → 0.3315905070 mm，退化 0.26433%；TEST global-MSE PSNR +0.00452805 dB、SSIM +0.000850524，仅非劣口径，仓库 channel-mean PSNR 还略降。耗时 2387.70 → 3151.18 秒（+31.98%）。条件非空样本的目标 D² 虽下降 47.56%，GT 未改善；不支持自动继续扫 λ。

[训练停滞诊断](../unified_geometry_gs/research/TRAINING_STALL_STATUS.md) 区分旧父进程 SIGSTOP、工具／消息流断连和 CUDA 计算。没有证据支持“每次训练结束 GPU 都会死锁”。遇终端超时应先查原句柄、退出码、日志和进程，不应重复启动覆盖同一输出目录。

## 4. TGS 固定输入研究：以 v29 为截止点

v1–v25 的校验、拓扑／顺序／局部窗口／补偿／backward 探针等均保留在索引。不要将早期小例子的梯度或吞吐结论推广至完整训练。当前讨论的 v21 原生控制入口是 forward-only；其 `backward_supported:false` 不表示整个项目从来不能训练，而是这一诊断接口没有训练能力。

[v26 共享损失](../unified_geometry_gs/research/TGS_SHARED_LOSS_RESULT_2026_10_21.md)：32 案例共享 FP64 损失与非共享定义逐位一致。热缓存 8 候选查询 A/D 时间比 6.20、11.96、12.45、21.47；D 约 2.90–5.74 ms，缓存 74.9／116.4 MB 每视角。给定 RGB、无 any-cache 的冷成本算术比 1.78–2.55 是条件成本推算，不能写成真实 cold-start 计时或训练加速。

[v27 联合编辑](../unified_geometry_gs/research/TGS_JOINT_EDIT_RESULT_2026_10_21.md)：32 原生参考、24 成对案例；所有交互仅舍入量级，尚未示出实质真实交互。支持分离的拼接条件得到案例验证。平均 opacity 的实际单编辑权重通常不能直接相加为联合变化；符号 toy 不能冒充真实训练。

[v28 视角迁移](../unified_geometry_gs/research/TGS_POLICY_TRANSFER_RESULT_2026_10_21.md)：flowers 主规则新测试变化为 0；room 平均 PSNR +0.000030997 dB 但 SSIM 下降，没有有意义质量收益。首次全体重算颜色曾失败：room 中 790593 个颜色值出现至多 4.76837e-7 差异，约 102 万像素在局部 ROI 外发生变化。改用固定外部 FP32 颜色、仅新增子项独立 SH 后通过当前契约，不能宣称原始全流水线等价。原失败与 TLS 错误回执保留。

[v29 等预算联合策略](../unified_geometry_gs/research/TGS_JOINT_POLICY_RESULT_2026_10_21.md)：每场景 256 候选、5 规则各 8 次细分、净增 24 三角形；52 原生前向、40 完整联合参考。CPU／GPU 分数最大差 3.978301995e-18。主规则 flowers 留出混合损失均值 +2.986238722e-6、PSNR -0.0001455154 dB；room +4.292517101e-8、PSNR -0.00000343436 dB，均退化。数值核验成功与策略泛化失败同时成立。

v29 数学补充：固定 k 时 `r_i^(k)=3*lambda*(o_i-mu)/(N+3*k)` 可将联合平均 opacity 正则差正确分配给候选；实数公式用真实 binary32 输入的精确有理数核验，另报 FP32 mean 舍入差。双校准方向下独立三阶 SH 每通道 16 个参数、秩至多 2、零空间维数至少 14；80 个真实父／子实例的 float64 方向矩阵数值秩为 2。这不是实际渲染 Jacobian 秩测量，也没有拟合或泛化保证。

[v29 文献定位修正](../unified_geometry_gs/research/TGS_RESEARCH_POSITION_UPDATE_V29.md) 优先于 v26 旧定位：CF-Splat III-F 已包含三角加密；细分、外观重拟合、方向正则的组合本身不建立新颖性。此处转述既有文献审查，不是本轮开展的新文献检索。

## 5. 下一项有区分力的工作

遵循 [v29 后续提案](../unified_geometry_gs/research/TGS_BOUNDED_APPEARANCE_NEXT_PLAN_V29.md)：先捕获原生 accepted alpha/transmittance 的颜色响应系数，保留全部贡献，不用固定 top-K 近似。用零偏移和预先固定正负 DC 扰动，对完整原生图像检查一致性／有限误差界，明确事件列表固定、颜色 clamp 与 SH 参数语义。**目前没有 v30 实现或已通过的门槛。**

若准备阶段通过，再让人类审批具体有界拟合协议：原几何 24 参数、子项 tied 24 参数、子项独立 96 参数，均冻结几何与其他外观；固定最大步数、Adam 参数、耗时、内存和停止条件。旧测试视角已成为开发数据，未来验证需先冻结新留出选择。若收益只能由额外参数解释、仅校准改善或留出不改善，应接受反证，不能改视角／阈值追结果。

## 封存与失败证据的优先级

优先级：冻结协议和实际输出／manifest > 对应最终结果说明 > 阶段性 README／聊天总结。阶段失败必须连同修复保存：v22 首次计时 954 预期 vs 982 实际，CPU 恢复原始计时且未重跑 GPU；v25 SM8040 遥测异常不能作稳定时钟证明；v28 全体颜色重算失败不能被成功重试抹去。

最近 manifest 的摘要（2026-10-21 封存；本轮只复制并核对小回执，不重新审计全部依赖）：

- v26：`e0b6f942acc75cd33bc60db31774b40ed00f6020e791508c00c27d39c3acb40a`
- v28：`b59e5ca48840896b9c5b11b608d71437ac0f47b851fe4f0e8416ebb535564f3a`
- v29：`cfae10b954bf971a9b3e3c18acadeff10e5764bdefaf4226451d0c64933d63c2`，293 文件、268864684 字节、394 依赖。原 manifest 可在[交接副本](evidence_receipts/unified_geometry_gs/research/experiments/tgs_joint_policy_v29/evidence_manifest.json)查看；文件和依赖实体需随外部资产移交。
