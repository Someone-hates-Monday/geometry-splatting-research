# 设备原参数预算与固定射线检验

2026-10-14。接续严格尾部端点组件通过。原GUARDED_CUDA_PROTOCOL目标不改，继续未舍入saved-activated世界模型与共同amplitude。

设备输入为固定原mean/scales/quaternion或R、camera affine、query、amplitude。interval inverse给真正F=diag(1/s)R^-1 A^-1；名义F取区间中点，设备two-pass modified Gram-Schmidt生成3x3Q/U，正对角。nominal仅用于缓存和候选，不替代真实F。polar defect/residual上界形成eta_U，向上舍入F32；相机mean区间给eta_m。低6F32U+3F64mean/高F64U共用真实原参数区间。基本操作用IEEE有向double intrinsic，sqrt也有向；不使用近似正交transpose或未证明单位四元数。

空间offset先计算，名义与区间残差给p/rho/energy，利用已有u·r=0误差界。只有原camera mean/query区间证明offset严格为0，才覆盖energy=0及source peak。rho<=eta/逆不可分离/QR失败/非有限值为unresolved，不回填MP。

局部固定旧7010加新seed13019已保存312层，共7322；全部low/high p/rho/energy/alpha区间须覆盖独立MP参照，status失败保留并计为失败。记录预算、Q defect、缓存及完整区间，诊断输出的26个double不是56B生产缓存的内存实测。

全部110模型两路径terminal mass在设备分类，106finite/4no-root保持原判定。使用已有CPU候选括区的212端点，分别检验low/high设备系数区间的严格符号，失败和unresolved均保留。该端点检验尚不证明设备求根完成；之后由设备自己产生候选并认证，原1e−3误差与2e−3括区要求不变。

条件端点组件门槛为每个finite模型的同一路径两端都认证：low未认证则必须high两端都认证；不以单边路径混合回填。低路径未认证属于预定回退触发，数量和全部项保留，不改为通过。局部预算覆盖与mass分类仍要求全部两路径通过。

全部输入、source与generated、MP核验和五批event timing存盘；预算/分类/端点分别计时，不将诊断吞吐量或固定集合fallback比例冒充完整renderer/真实像素发生率。CPU预算、Q/U、MP根都不输入设备预算函数。

当局部构造/分类通过，再实现设备候选与条件回退、独立核验固定模型根；完整门槛通过后才扩张完整图像/backward及跨场景质量实验。保留全部旧失败，暂不启动训练。
