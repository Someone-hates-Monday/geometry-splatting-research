# 动态CSR选择性严格求根：先固定协议

2026-10-15。float_product_v1全部562335个未认证、原有效像素是主评估集合，24原开发视角全部执行，原8583586已认证像素保持原公共值。原 contributor ID/实际query/AMP/模型/相机保持，不限制192节点。原无候选/不eligible者不在此阶段自动填补。

每节点先解码旧shared_cache低模式，存26double；动态int64分段，每warp一射线。名义double候选使用源AMP/energy稳定前缀位移和缩放signed-log尾部，二分最多200轮/48次扩展，数值零只可返回候选。AMP高于cap时转换名义有效alpha，不把AMP>=1视作非法参数。先通过总最终透射率的严格区间判定有限根存在、无根或不确定，任何分母/分支/包络不确定都保留。

候选t转换为公共float32 z，之后用实际query长度的double有向区间计算真实中心。端点在中心区间±tau内，tau与旧F32协议相同严格小于1/1000。依次尝试F32乘积端点、旧short_interval有向scaled-log端点、旧strict_interval scaled-log端点，只有有效严格相反符号才accept。任一低模式失败，重新解码高模式并重新提候选/认证；最终失败为unresolved。prefix峰侧不确定不随意选分支。高模式和strict回退失败也必须保存，不能当成功深度。

全量per-ray输出、association计数、模式/证书/无根/失败分布与event/wall/驻留保存。每view主运行后两次重复，输出位一致作为可重现检查；compile/build/decode/root/CPUcompact等成本分别记录，旧full-renderer/capture/cache构建成本另计。不把这一分段diagnostic叫完整方法提速。

独立核验：原固定1545探针中的全部132未认证射线，加每view32等距未认证像素（结果生成前固定，非随机/非留出），取集合；完整源节点MP80/140参考，在最终报告公共z中心与两端检查严格符号/误差带/系数包络。所有失败及原9异常保留，任何误认证停止采用。逐条MP探针不是全部562335射线的逐节点证明。

全量根替换的图像/几何GT、世界VJP、训练、新颖性仍待此数值门槛之后核验。固定已观测开发视角不能证明泛化。生产源码保持，结果不含CPU参考根作为GPU输入。
