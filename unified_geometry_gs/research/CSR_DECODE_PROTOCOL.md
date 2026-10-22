# 真实CSR分批解码：实际query/AMP、全量状态与固定独立探针

2026-10-14，观察输出前固定。共享缓存224B诊断构造与7322位一致已过；本轮接入全部24视角870178922条实际关联，不截断、不挑选成功像素。

## 输入来源与核验

从full_topology_v1冻结instrumented源生成私有query捕获版本：只扩大mdepth到8通道，在原kernel写实际pixnf.x/y到新通道6/7。原RGB/alpha/normal/depth/slope/counters/radii要求全图位一致，6个旧通道与存档一致。第三query分量为原字面常数1。AMP从该实际native geometry buffer conic_opacity.w取得，不假定它等于输入opacity。

每view重建旧共享cache，核对原PLY/activated输入/相机身份。加载已独立全量审核CSR records/offsets，与当前count/query capture的每pixel计数再次精确匹配，保留全部primitive ID与offsets。使用冻结shared_cache.cuh解码正文；新stream kernel只做ID/pixel定位与query/AMP填写。

## 全量流式与门槛

batch固定65536，最后批为余数；用int64上界搜索在offsets中恢复每条record所属pixel。ID/pixel非法显式状态6。每mode仍输出26double临时诊断；逐批记录所有status数量、status0非有限量、hash，以及名义alpha候选的下溢/precision loss情况，不能把失败状态丢弃。要求实际processed=870178922，所有source/query/CSR身份检查过，mode高路径没有非法/非有限记录；低路径失败留档作回退需求。

不永久保存约362GB全量double诊断：保存每批输出SHA、输入/point/query身份、状态与固定探针完整source/output，以及逐view的query map/AMP/counters，全部统计覆盖所有record。固定探针为每view48个等距record索引（不按结果选择）；另加此前9异常射线的全部CSR关联。探针设备cached与旧直接preprocess全数组要求bitwise相同，独立MP80/140世界covariance逆计算uncapped energy及capped alpha，核对peak/rho/E/alpha区间。异常q与旧实际捕获记录比较，差异必须报告而非假定。

没有逐条MP全870M验证；probe性质与覆盖范围明确，不扩大为每像素根证书。未运行候选/root/WorldVJP/GT/训练。

## 成本与归档

逐批event/wall包含output allocation/fill与对应kernel；CPU transfer/hash/状态扫描wall另计，cache/query捕获/CSR加载分别计。记录批峰值GPU内存和每view总event/wall。没有本阶段性能采用门槛，不能把几何cache重用的17%局部收益转写为全图速度。若状态失败，保存全部view及失败探针，不改batch/样本/阈值后挑成功。

保留源码/生成源/actual binaries引用/hash、原inputs引用、全部batch统计/probes，更新现有LaTeX并编译。后续才实现动态/射线内并行root及证书，保持原目标/初始范围语义和显式unresolved/F32输出舍入。
