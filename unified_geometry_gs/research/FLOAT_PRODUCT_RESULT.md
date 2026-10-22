# 原公共F32深度的直接乘积证书：全量覆盖93.85%，固定完整射线核验通过

2026-10-15 Asia/Shanghai。主批session26752 exit0；独立MP核验session10530 exit0；探索原因诊断exit0。生产训练/renderer源码、模型不改。原LaTeX原位补充公共F32输出舍入的证书命题。

实现、协议、结果：float_product.cuh / float_product_cuda.py / FLOAT_PRODUCT_PROTOCOL.md / experiments/float_product_v1。F32有向exp8及显式余项，低模式double有向几何保持；每warp完整CSR射线，不截断节点，紧凑12double输出全部永久保存。只认证原公共F32z对应真实qnorm深度±tau端点严格符号，不自动修复未认证像素。

24开发视角全部10,834,488像素、870,178,922关联闭合。初始eligible9,165,957；原有效深度9,145,921，其中认证8,583,586（93.851521%），未认证562,335（6.148479%）；其余1,688,567属于无原候选/不eligible，不是数学无根。全部bad标记0。每view初次和3次重复输出位一致。

独立审核results/float_product_audit_v1.json：25来源、76产物、actual binary/生成源/ZIP通过；旧几何正文只修改alpha exp。4104固定scalar输入和精确ln2/尾部/reciprocal/tau算术校验通过。固定每view64等距eligible像素+原9异常共1545完整射线、145859世界源节点，MP80/140独立逆世界covariance；583436项p/rho/E/alpha系数包络与所有有效探针端点T通过，1410认证探针严格符号与公共深度舍入条件通过。没有把1545探针视作全部870M逐条MP证明。

探索性原因诊断FLOAT_PRODUCT_FAILURE_PROTOCOL.md / results/float_product_failure_analysis_v1.json：在上述固定探针中检查全部132条未认证射线，以公共z×真实qnorm±精确1/1000的MP80/140符号分为根在左侧18、右侧114；无界宽单独导致拒绝的例子、无无根/不确定。原9异常全部包含，2左7右。这是按未认证状态选择的完整子集，不能推算562335像素总体错误率，更不是GT几何误差。

诊断kernel+compact allocation/fill的每view3次event中位数57.713535–134.755325ms，cache build/CSR/query驻留另计，全部时序原始值保存。相对上轮0.60–2.16s双路径double诊断降低了开销，但仍慢于原renderer8.235–21.230ms；两个诊断职责/输出不同，不据此声称方法端到端加速。

下一步先实现动态长度、射线内并行的选择性strict根修正，保留贡献集合和实际源参数；诊断未认证根修正是否可在受控完整成本下完成，再核验世界VJP与GT/训练。当前成果是可复核的充分条件与失败定位，学术新颖性/显著完整效率或质量提升尚未成立。当前无训练/GPU作业待结束。
