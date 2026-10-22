# 下一质量门槛：固定模型完整视角重建

2026-10-15，结果生成前固定。模型output_compare/ggs_pipeline/scan37,55,63的iteration30000与已冻结PLY SHA保持；每个模型完整49原训练视角，不只使用先前24开发视角。全部是此前使用过的研究数据，不称作独立留出场景/新训练结果。

对照原Gaussian renderer的median_depth；候选基于相同真实query/AMP与原完整接受集合，保留原已通过充分条件的公共深度、对未认证原有效像素动态修正。严格无根与最终未解决分别记数，候选对应深度置0；原无候选/不eligible不擅自填补。RGB保持原值，gt_mask策略完全相同。完整像素数/名单/位一致与当前numerical部分的来源检查必须保留。

沿用mesh_extract.py的TSDF配置：voxel=.002、depth_scale=1、depth_max=8、trunc_voxel_multiplier=4、block_count=50000、largest cluster1；原RGB白背景、mask、intrinsic/extrinsic/视角顺序相同。不调参数挑最好候选。两mesh采用同一官方相机中心对齐，确认完整视角名0000--0048与官方索引匹配，不用错误firstN映射处理非连续相机。模型/源/GT文件SHA及全部运行状态保存，原模型文件不写。

每scan两支各用固定seed0/1/2的官方DTU几何评估副本。它们是评估抽样重复，不是训练独立重复；相同seed不意味着两不同mesh拥有相同采样点。预先质量门槛：三scan平均overall候选/对照<=.99，各scan平均s2d delta<=0，RGB逐像素位不变；同时报告每scan/d2s/s2d/coverage/无根/未解决，不拿训练PSNR替代。任何source/node/位核验失败必须保留并停止采用。

完整耗时含原forward、query/count、CSR scan/allocation/capture/check、cache、certificate/correction、CPU/TSDF/eval，不把单内核快或参考证明当整体加速。此试验仅推断固定模型推理重建变化；即使质量门槛通过也须另核验完整GPU后向/同状态训练与新颖性。未执行阶段保持pending，不能写成已通过。
