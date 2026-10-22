# 完整实际名单：动态CSR捕获与数量/顺序验证

2026-10-14，运行前固定。full_topology_v1全部24视角发现192容量不足334072个valid像素，max461。本阶段不设新的名单截断，全部24视角、全部原像素和原primitive-ray关联保留。

## 数据与接口

加载已冻结计数二进制control，核验源/二进制、原PLY、实际activated tensor和相机输入与full_topology_v1。读取GPU首轮median接受数（它与真实RGB接受数已逐像素匹配），用int64 prefix-sum分配offsets[H*W+1]和records[L,2] int32，L为精确总接受数。每项存primitive ID以及在原tile point-list中的位置，不按192/256/512筛选。

新私有native forward ABI只增加offsets/records/write_status三个tensor。记录在原FIRST实际接受位置、数值计算之后、depth count++之前；仅当inside且slot落在该pixel的已分配区间内才写。超写置bit1并跳过该写入，实际数不等expected置bit2；失败不算通过，不能传给后续认证。首次分配records=-1，status=0。

## 全24核验

原RGB/alpha/normal/depth/slope/radii与计数control全图逐位一致，新增计数/initial eligible逐位一致，且与原full_topology_v1保存图逐位一致。每pixel实际写入数等于prefix span，所有status=0、所有slot已填、ID属于[0,P)。独立GPU checker逐项确认position严格递增、落在真实n_contrib前缀和tile range内、ID等于point_list[range.start+position]；无抽样。保存完整CSR及pixel计数、状态/检查输出，独立CPU逐项重算offsets/长度/ID范围/同pixel顺序与完整metadata/hash。

另在预定scan37/0000第一个非零pixel做两项保护探针：expected count−1和+1，其他输入相同。分别要求目标pixel write_status=3和2（其他pixel=0），过分配恰好留下1个sentinel；原图像/计数仍相同。这两项故意错误输入独立保存，不混入主批成功或计时。

## 时间与采用边界

记录count forward、GPU prefix scan/总数读取/allocation、capture forward、checker、CPU transfer/storage各阶段event/wall；3次预热后5对交替control计数/正确capture（capture含records fill和status清零），固定原offsets/records重用。完整两次forward+CSR分配+checker单独计时，不能只拿capture阶段宣称方法速度。记录GPU allocated/reserved峰值和CSR字节。计时是捕获成本，不是认证/训练加速；本阶段没有性能采用门槛。

保存完整lists需要约7GB未压缩记录跨24视角，逐视角写NPZ并释放，检查当前磁盘余量但不删除原数据。若IO/观察超时，继续同一运行，不因轮询超时重启。编译/运行失败和所有未通过图像保留。

成功条件为全部24视角/原像素/无截断、完整数量与顺序检查及两项失败保护通过。仍不是完整认证renderer、世界VJP、GT改善、训练收益或新颖性；下一项共享源参数缓存与射线内并行认证/回退。
