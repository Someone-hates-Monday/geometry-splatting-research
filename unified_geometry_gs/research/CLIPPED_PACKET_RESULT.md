# 裁剪扩展前向通过；完整FP32开发门槛仍失败

2026-10-13。目标active，无质量/效率/原创性完成声明。原生产及所有旧负结果保持。

`clipped_packet_v1` complete、exit0：6项一般符号身份、128个固定SPD构造通过，
其中112个clip；同24射线全部1992 selected输入通过，包含原46个域外输入，无fallback。
`results/clipped_packet_report_v1.json`核验180源/archive、50产物、3PLY和全部ID分类，
全部1992原PLY独立80位forward核验，最大scaled errors(p,width,alpha)=
(4.7152e−15,1.5177e−15,2.6097e−12)；46clip子集最大alpha误差7.08794e−14。
旧unclipped v1仍保留其原域限制；不声称新成功修复了旧表达。

数学与实现见 `CLIPPED_PACKET_THEORY.md` / `clipped_packet.py`。
已有conic、两个plane系数和width编码参考基底中的完整P之Schur参数，
由此精确恢复ray peak/width/amplitude；Gram非负分解避免直接能量平方差。
仍要求kernel0和P/Sigma精确互逆；浮点与真实预处理不自动满足。

## 原参数反向与FP32检查

遵循 `CLIPPED_PACKET_PRECISION_PROTOCOL.md`，`clipped_packet_precision_v1` complete、exit0：

- 全1992真实FP32 packet舍入检查通过。max peak相对3.12954e−7、width相对2.70682e−7、
  alpha绝对5.58863e−6，分别低于原1e−5/1e−4/1e−4门槛。
- 全1992 raw FP64 packet图与直接world factor图的forward通过。
  mean3/logscale3/quaternion4/opacity1/filter1共12变量，3输出，共71712梯度分量全通过，
  max scaled errors按输出=(4.18977e−7,1.22574e−7,2.40034e−7)，原门槛1e−4/floor1e−8。
  clip边界<=1e−10的输入0。
- **128合成FP32中1例失败，因此development_gate_pass=false。** 不放宽阈值或删例。

两图共享标准raw world参数变换和Torch AD库；这是代数图交叉核验，不是独立解析VJP。
按输入复制的raw参数也不测试真实多射线共享基元的atomic累加。
FP32阶段只round double packet，不含实际CUDA/FP32 cov/conic/plane预处理误差。
所以不能据上述局部通过宣称完整渲染器反向或GPU稳定性通过。

## 唯一失败例与事后误差分离

固定case2：clip=true，真峰位约−.02839817738，FP32候选−.02839967981，
相对误差5.29059e−5，超过预设1e−5。宽度/alpha筛查未失败。
`results/clipped_packet_precision_report_v1.json`核验233源/archive、51产物、
全部1992行和71712梯度算术，保留整体失败，并用独立80位公式重算所有失败例。
这是已经看过失败case和double重算后的**事后诊断**，不是新盲预注册。

原double packet的80位decode峰位误差约1.85142e−12；
先round为FP32再用80位decode，峰位误差仍1.71395e−5，继续超过原门槛。
因此仅提高已经round过的packet的求值精度，在此例不足以恢复原精度；
不由单例推导“任何decoder不可能”的定理。
该构造的真peak在相机后方，80位amplitude约2.3005e−5712（double中underflow为0），
这是下游相关性限制，**不用于把原失败重新判通过**。

下一步需要研究输入舍入敏感性与精度保护，区分metadata误差与实际可见性/透射率误差。
任何新门槛或方法必须另立方案并保留此原失败；暂不训练。
实际生产FP32 preprocessing、CUDA forward/backward、资源/计时与GT质量尚未验证。

运行中曾有一次自动审批403，命令未执行；原样重试审批后已执行。
这是服务失败证据，不是训练/GPU故障证据。所有研究数值主任务已终态，进程状态需实时复核。
