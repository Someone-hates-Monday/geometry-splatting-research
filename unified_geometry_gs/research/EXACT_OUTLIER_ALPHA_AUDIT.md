# Alpha/能量的实际舍入路径与候选修复

2026-10-14。只读当前源码与`EXACT_OUTLIER_COEFFICIENT_PROTOCOL.md`；未运行GPU、未扩检索。8mix/raw结果尚未在本审计中读取，以下是候选预测，非已确认归因、新颖性或质量结论。

后续勘误/范围：本页形成于8mix/raw计算前。现已完成的结果见`EXACT_OUTLIER_COEFFICIENT_RESULT.md`：有效alpha-only预测得到本批支持，但尚未唯一分离能量损失中的具体算子。下文kernel0的coef=1是实数公式判断；实际CUDA保存的co.w与输入filtered opacity在96/772 accepted项末位不同，两臂仍共享同一个co.w。第三固定共同amplitude参照已经控制这一因素，不能把数学上=1写成实际逐位=1。

## 已定位的信息丢失点

- `exact_perspective_ray.inc`两臂共享`co.w`；kernel0下其预处理系数为1。对同accepted项，alpha差主要需要检查各自能量与最终F32输出，不能归给独立opacity缓存。两臂都在double求能量/exp，随后将alpha转F32并cap .99；提高之后求根精度不能恢复这次转换前的信息。
- 生产`render_forward.cu`的`computeCov2D`用`S*R*W*J`的Gram构造投影cov，再在349–354行以F32行列式求逆成conic；plane/rho则来自`S_inv*R*W`的precision Gram。R与W只近似正交，两条矩阵链一般不严格互逆。packet将前者conic与后者plane混合解码，已有实数Schur证明的互逆前提可能在实际输入上破坏；这比最后一次det相减更早。
- 相机均值也走两条链：packet从F32 `point_image`反算mx/my，并用F32 `plane.z`范数反推z；原投影330–332行分母额外加1e−7，且用full projection矩阵。factor直接缓存F32 view变换后的三维均值。即使矩阵数学上对应，这些实际算术路径也不逐位等价；当前packet反投影未补该epsilon。
- packet的b由已rounded rho平方得到，c由已rounded plane得到，K由已rounded conic得到；double只能精确处理这些保存值。最后`det=kx*ky-kxy*kxy`及`v^T K v`仍有相消。factor避免conic逆，但F9和mean3先舍入F32，再求`o-p*u`残差，接近平行时也可能相消。不能预先把factor当raw世界真值。

## 一个值得等待8mix结果的稳定构造

若alpha替换主导数学根差且raw参照指向packet能量链，可从**同一precision factor F**构造b、c及K，避免把cov逆与precision混用。令u=F*qbar，vi=F*ei，则

    b=||u||²,
    Kij=((u×vi)·(u×vj))/b².

这是已有Gram身份的实现安排，不是新的Schur证明。再把K保存为三角因子L（三个标量），用`||L*v||²+(det L)²*wedge²`计算当前clip-aware能量的分子；三角det无需大乘积相减。叉积本身仍可能相消，需要FMA/更高精度或稳健正交变换，不能宣称普遍稳定。

最小缓存预算须区分目标：仅替换能量所用K且保留旧plane，可新增3个L标量，但不能修复plane、中心与b的舍入。要保留原RGB/affine selection字段，同时提供一致几何packet，需要独立的b、c两项、L三项共6标量，另缓存实际camera mean3，共9个F32=36字节/基元；当前完整factor缓存为12个F32=48字节。也可用F的QR三角因子6项加mean3得到同样9项预算。QR/因子舍入、预处理成本、反向与速度均未验证，不能从字节数直接宣称加速。

若允许替换已有字段，预算可不同；但当前实验必须保留原affine选择，不能一边改变conic选择一边把收益归给能量。单个额外alpha只服务一个像素，无法补全所有方向的能量。仅对已rounded packet再做Gram/FP64解码，也无法知道之前各字段的舍入残差；本轮没有构造“相同packet、不同raw能量”的具体双输入证书，故不声称已证明信息论下界。

## 可证伪预测与执行限制

若根间是真正的宽平台，小alpha差可能使平台两侧累计质量跨过.5，产生跨层根切换。对同一前侧集合，精确有界关系为

    |Delta log C| <= sum_i |Delta alpha_i| /
                           [1-max(alpha_i_packet,alpha_i_factor)].

这只界定累计质量扰动，不给无裕量时的统一深度界。预测应由全部8mix与raw参照检验：alpha-only是否复现大根差，peak/rho-only是否较小，混合模型是否跨平台阈值，以及哪个表示更接近各自明确定义的世界参照。根处斜率不能代替整个根间区间下界。若该预测失败，不实现上述能量修复；若成立，也先锁定原输入和新构造做局部门槛，保留完整packet慢2.60倍及24视角失败，不自动训练。

来源：`exact_perspective_ray.inc`、`exact_perspective_factor.inc`、`exact_outlier_capture_backend.py`、生产`render_forward.cu`。capture是复制表达式probe；它定位输入系数，不是原指令trace。本说明没有把未归档探索stdout作为正式数据。
