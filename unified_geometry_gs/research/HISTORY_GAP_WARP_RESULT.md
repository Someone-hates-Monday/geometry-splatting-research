# 并行损失核：正确性全过，GPU完整loss批次约6.60倍加速

2026-10-15。上一goal turn完成GPU正确性，本轮为实质进展：独立warp核、完整同进程对照和独立审核。
不改原串行CUDA/graphV2/生产renderer/模型，没有启动训练。

每条射线一个32-lane warp，strided读取全部节点，用local补偿求和、double-double prefix乘积和固定warp归约树。
数学loss、早退出、GPU独立根、cap和density伴随不改；没有隐藏串行回退。
原891射线/51524节点、9异常及3零loss全部保留；串行与warp各891/891过原loss<=1e-9、
整射线相对L2<=1e-5、各raw块<=1e-4门槛。两版本max整射线9.143033722839749e-11，
max块1.7200367458184842e-10。原64合成全部过，两个真实缺根更新全部过。

原10系数特殊案例（奇异、质量边界、strict cap、薄间隙等）全部过；新增513/1025节点射线全列表MP参照/seed通过，
report节点数也完整。独立exact Fraction确认：去掉最后基元的prefix最终T>1/2，全列表最终T<1/2，
避免最后项没处理却碰巧正确的弱测试。invalidrho0明确负状态/NaN拒绝。
所有12个特殊case完整input/reference/actual/info数组保存，未改变原零块门槛。

同一进程/设备/graph/stream，两臂每轮交替先后顺序，10轮，前2轮每臂warmup原样保留，后8次全用于中位数：
完整geometry+loss+raw-backward串行79.11893081665039ms、warp11.993087768554688ms，
warp/serial=.15158303638287754，loss组件加速6.5970442594522245倍（耗时约下降84.84%）。
仅loss/seeds串行72.75982284545898ms、warp5.913071870803833ms。
allocation计入，setup/host transfer/compile不混入这些事件。
这是本新loss对串行参照的GPU批次优化，不是原3DGS/GGGS renderer或完整训练加速，不是显著性/独立训练重复证据。
原预设correctness全过且full-ratio<=.95采用门槛通过，可用于下一research adapter。

首次runner session99775编译并跑完timings，汇总dict重复gate_pass而exit1，未保存数值结论；
原manifest/traceback/sourceZIP和native保留。只修runner字典及增加逐步timing/特殊数组持久化，
第二批session72993 exit0复用相同source-checked binary，输入/精度/数学/门槛不变。
第二批summary字段compile_wall_seconds=.00884s实际为已编译native加载时间，不是CUDA编译耗时；
首次编译已完成但其耗时未保存，不能把.00884s称为编译加速。

独立verify_history_gap_warp session33454 exit0：111来源、17产物、sourceZIP/实际binary、
全部loss/梯度/原MP引用/12特殊数组、exact长射线质量证据与全部20×2交替时序统计复算通过。
results/history_gap_warp_audit_v1.json；实际native SHA
b95d0172b690670318456986daf3defdecac151386e7a408a313e0c777b27aa1，
保存在experiments/history_gap_warp_v1/generated，虽然首次runner失败，该已编译核不属于数值失败。

下一步从真正GPU采样/capture源构建训练regularizer adapter：actual shared Gaussian IDs，source-preserving raw链，
原render输出保留，固定camera/采样序列，先零权重同状态梯度/optimizer对照，再非零权重短步与完整迭代成本。
不能每步做CPU capture/transfer/冷启动，不能在数值失败/空射线上默默奖励0或传播NaN。
通过后再预注册配对训练、GT/留出图像/容量评价；整体科研目标仍未完成，旧147-view GT失败结论不变。
