# 动态实际名单：全部24视角和8.70亿关联完整核验

2026-10-14。协议 `CSR_CAPTURE_PROTOCOL.md`，主批 `experiments/csr_capture_v1`，独立CPU审核 `results/csr_capture_audit_v1.json`。本阶段消除名单接口的固定192容量限制，没有将其写成完整认证渲染成果。

## 实现与全图正确性

先用已冻结的计数二进制读取实际首轮接受数，用int64 prefix sum分配offsets[H*W+1]和int32 records[L,2]。在原FIRST真实接受位置存primitive ID与原tile point-list位置；分配只依实际数量，不按192/256/512截断。写入先检查该pixel边界与总容量，越界置bit1，实写数与expected不等置bit2。原生产源码不变，新ABI和源树只用于私有native诊断。

原scan37/55/63各8视角、全部10,834,488像素，捕获 **870,178,922** 条关联，与上一阶段全部initial名单总数精确相同。全部original RGB/alpha/normal/depth/counters/radii在运行中逐位一致，depth/counters与先前存档完整位一致。所有slot已填、ID有效、write_status=0。

GPU checker逐条核验position严格递增、落在真实n_contrib前缀和tile range、ID等于原point_list对应位置；全图checker_status=0，没有抽样。

独立Windows CPU检查全部offset差等于pixel计数、每条ID范围、同pixel顺序、真实前缀/tile边界与point-list ID匹配，**全部870,178,922条通过**。核对36来源、54产物、18生成源、实际二进制与vendor headers/ZIP。原RGB/alpha/normal/radii在运行中完整比较，其数组未留存第二次离线位审核；完整depth/counters/CSR及原point-list已保存并独立复核。

## 错误分配保护

scan37/0000第一个非零pixel（flat index0）两项预定探针：

- 少分配1项：仅目标pixel status=3（越写尝试与数量不符），无实际越界写入，所有分配slot填满；原图像/计数不变。
- 多分配1项：仅目标pixel status=2（数量不符），恰好1个sentinel未填；原图像/计数不变。

两项通过。它们是故意错误输入，单独保存，不能计作主批成功名单；后续认证必须拒绝任何非零状态。

## 实际成本

每视角records本体157,311,432–383,197,936字节（约157–383MB），offsets/status另计；6.96GB记录是跨24视角汇总，不是单视角同时常驻。逐视角存储完整NPZ后释放，实际源与结果有归档。

3次预热后5对交替count/capture，capture计时重用分配好的CSR但含records fill/status清零；capture/count配对比的视角几何均值 **1.074843640**（约7.48%增加）。这只度量捕获阶段相对原计数前向的开销。

另外完整count→prefix scan/总数读取/allocation→第二次capture→checker流程，每视角一次event为 **23.679647–50.505154ms**，同步wall为23.814754–49.580105ms。该event包含host同步/发射间隙，两个计时保留实际值；不是5对完整流程性能采用试验，也没有方法加速判定。CPU传输/压缩存储、编译与Scene加载单独计，不混成GPU收益。

名单准备还需融合/优化，当前不代表最终认证算法成本、完整训练吞吐或GT提升。

## 下一项

接入可共享的原参数缓存，使同一个primitive的世界因子与误差预算不随每条关联反复构建；再用射线内并行和动态名单求候选与证书。保持原物理根、原贡献项集合与初始范围语义，明确unresolved及公共FP32深度输出的额外舍入。世界VJP、完整成本、跨场景训练、held-out图像/几何及新颖性验证仍未完成。

主批session19995和独立CPU session95619均正常exit0；本阶段没有新训练。
