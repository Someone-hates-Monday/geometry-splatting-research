# 实际query/AMP的全量CSR解码：数值与探针通过，诊断成本不适合直接采用

运行2026-10-14 UTC，独立核验完成2026-10-15 Asia/Shanghai。协议 `CSR_DECODE_PROTOCOL.md`，主批 `experiments/csr_decode_v1`，独立审核 `results/csr_decode_audit_v1.json`。

## 来源与完整数据

私有query捕获源只扩展两个输出通道，在原内核写出实际pixnf.x/y；第三方向分量是原常数1。幅值读取实际geometry buffer的conic_opacity.w，没有把它替换成输入opacity。所有原RGB/alpha/normal/depth/slope/counters/radii完整位一致，新增query与先前9异常射线的copied query位也相同。

原PLY/activated参数/相机身份与旧CSR核对，逐pixel接受数与存档相等，原CSR ID/offset全部保留。以int64 upper-bound搜索定位每条record的pixel，源参数cache重建后通过冻结query正文解码。没有删失败输入或固定名单容量。

24视角 **870,178,922** 条关联全部处理，两个mode都status=0、有效输出全有限，状态/processed计数精确闭合。固定batch65536，最大临时输出 **27,262,976B**，避免全量约362GB double诊断常驻。实际cache、CSR、query/AMP仍各有驻留成本，该批大小不是全部GPU内存。

保存每batch状态、完整输出SHA、event/wall、transfer/hash/statistics wall；没有把全量double输出永久存下。离线可核对批覆盖/状态/hash清单与代码，不能重新离线计算未留存输出的全部SHA；限制保留。

## 预定独立数学探针

每view48个等距record及此前9异常射线的全部关联，共 **1,924** 个不同探针。缓存/stream与旧直接preprocess的全26double×2数组位一致。独立MP80与MP140分别从原世界covariance逆计算uncapped energy及capped alpha，两精度一致；两mode共 **15,392** 项peak/rho/E/alpha区间检查零失败。

39来源、74产物、生成query/strict源、实际binary和ZIP核验；query捕获的源变化仅新通道，strict shared_cache.cuh正文与旧存档相同。探针没有按结果选择，但它们不是对870M全部系数的逐条MP证明，也不是每像素根证书。全部仍是此前观察过的开发视角。

## 成本与下溢

各view双路径decode event总和 **597.528577–2164.430415ms**（约0.60–2.16秒），包含临时输出allocation/fill与kernel，不含CPU传输/hash/statistics；这些CPU wall及capture/cache build另存。当前26double×2、全部double有向算术和双路径全解码是昂贵诊断，**不作为完整方法速度或训练采用结果**。后续需要选择性回退、紧凑结果与射线内并行，且必须保留安全检查。

两mode各发现 **21,577** 条关联的名义AMP*exp(-E_nom/2) double计算为0；这不是数学AMP=0，也未删除关联。严格源区间仍status=0；标准math下溢不允许作为严格尾部/根/梯度保证。此处统计的是未转F32的名义double alpha，尚未计量所有F32候选信息损失。

## 下一项

用这批真实输入固定普通像素的便宜充分条件，严格缓存/解码保留为回退参照，明确候选参数信息损失、公共F32输出舍入与unresolved处理；实现动态、射线内并行root时保留原贡献集合与初始范围语义。根/端点证书、世界VJP、完整GT、训练/留出图像/几何和新颖性仍未完成。原生产源码未改；主批session35967与独立审核均exit0。
