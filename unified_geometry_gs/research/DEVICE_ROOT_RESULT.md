# 原参数到认证中位根：固定 CUDA 开发通过，效率与训练尚待验证

2026-10-14。协议 `DEVICE_BUDGET_PROTOCOL.md` 与 `DEVICE_ROOT_PROTOCOL.md` 固定；原物理根目标和旧失败保留。本阶段将原参数预算、存在性分类、候选求根、严格认证与条件回退全部放到设备。

## 实现与数据边界

`guarded_preprocess.cuh` 用有向矩阵逆构造真正F=diag(1/s)R^-1 A^-1，设备two-pass modified Gram-Schmidt给名义Q/U；polar缺陷与residual预算覆盖QR及缓存误差。低6F32 U+3F64 mean，高路径F64 U；预算向上舍入F32。camera mean与offset有向计算，只有source offset严格为零才证明energy=0。

`guarded_root.cuh` 以名义p/rho/alpha(F32低)或原amplitude/energy和F64名义p/rho(高)提出候选。标准math函数只用于候选，不能认证。认证使用已验证的严格基本算术/级数/log-tail核心。低候选未认证再做高候选；仍不确定返回unresolved。tau0为严格小于1/1000的binary64数，端点向内舍入，避免首版浮点宽度问题。

输入为7322个原参数层项和110模型start/count；设备预算/求根kernel均未接收CPU候选、bracket或MP参照。后验检验读取参照，不参与选择或回填。真实9射线保持实际renderer accepted prefix；数学根使用该固定集合全局bracket，而不是生产5步初始化/范围策略。

## 完整固定结果

- `device_budget_v1` complete/exit0(session54256)：旧7010与新seed13019已保存312层，共7322；两路径58576项peak/rho/energy/alpha参照包围检查零失败。110模型×2路径mass分类全过；原CPU括区下低认证79、高认证106，属于端点组件。
- `device_root_v1` complete/exit0(session60003)：GPU自主产生候选与括区，全部106有限根返回认证，4无根（2critical/2insufficient）分类正确。低76有限根、高30有限根；无unresolved/遗漏/回填。
- 最大根误差 **1.361838888e−5 camera-ray单位**，原门槛1e−3不改。不是world mm或GT误差。
- `results/device_root_audit_v1.json` 独立MP240世界covariance逆重算106对新端点，4408使用中系数、严格符号/scaled包围、精确有理宽度及4无根均过；两组56+14 source/archive、9+7产物、10+11 generated校验完成。

同一固定新seed已经用于此前CPU和组件开发，本轮属于完整设备开发检验，不再称独立未观测确认或真实像素发生率。

## 成本与采用状态

预算组件7322项的双路径event中位数0.394208ms，mass0.093536ms，双路径212端点200.871521ms。
自主求根阶段7322项预算0.441504ms，110模型求根含证书705.173584ms，合并705.568237ms；五批event，包含输出allocation/fill和对应kernel，未包含host传输或完整renderer。

这些是两个独立诊断运行的局部计时，不是同一规模/工作流的完整效率对照。严格认证当前成本明显，**未证明加速，不直接移入生产训练**。诊断26double/模式的输出不是56B拟定cache对齐或内存占用实测。

## 下一步

优先给健康像素的便宜充分认证条件，使严格核心保留为难点参照；保持同一物理根与所有临界输入。之后在完整24视角的forward路径衡量实际成本/选择与初始范围，再独立核验backward。只有通过这些门槛才进入跨场景配对训练、留出图像与几何评价。

尚无完整renderer/culling/order/init-range、全部世界VJP、GT质量或新颖性完成证据。根值正确也不证明梯度可用：已有薄层临界族的真实opacity导数可远超浮点范围，此限制仍保留。

严格结论信任IEEE intrinsic、编译器和有向基本运算，未做proof-assistant或全部指令形式验证。原生产CUDA/训练实现未改变。
