# 原参数误差预算与对数尾部：固定 CUDA 开发检验

2026-10-14。上一阶段 `guarded_precision_v2` 已完成 CPU 数值证书，不是 GPU/训练成果。本轮不改门槛、旧输入或新 seed13019 生成数据。

## 单一实现

独立 CUDA 扩展，从原 mean/scales/quaternion 或 rotation、camera affine、query、amplitude 构造真实 inverse precision factor 的区间。基本运算用 CUDA double 有向舍入 intrinsic；禁止 fast_math，禁止隐式 multiply-add contraction。名义 3x3 QR 在设备计算，预算由区间 residual 与 polar defect 给出，向上舍入到 F32。

低路径缓存六个 F32 U、三个 F64 camera mean；高路径用 F64 U。相机 mean/横向偏移也以区间计算，只有原参数区间严格证明共线时才用 energy=0。名义结果不替代证明预算；MP 参照不回填预算、求根结果或选择。

根残差的尾部保留 log magnitude，不把小于 double 最小数的 tails 置零。对每层 b=-log1p(-exp(logg))/2：logg<-40 时以 logg-log2 及解析相对修正界包围；否则用有误差扩张的 CUDA 双精度数学函数。正负项通过共同尺度求和，对被缩放后小于 exp(-700) 的项保存显式上界。正负区间分离才能返回符号。

编写后原始文档核验发现 CUDA12.8 明说 standard function 的误差边界来自非穷尽测试、非保证界。因此严格证书不使用 exp/log/log1p/expm1 库函数及猜测 ULP 扩张；改用 range reduction、24项级数与显式余项、有向基本运算。ln2 的二进制区间由精确有理 atanh 级数独立证明。这里只信任 IEEE 基本有向运算，不宣称 proof-assistant 证明。未覆盖/非有限/无法分离必须返回 unresolved。

终端 mass 区间判定 finite/insufficient/critical/unresolved。严格符号括区经设备二分/区间细分缩至 2e−3 内，返回 midpoint；低路径不能认证时在高路径重新求解。不能认证不能当通过。若先做端点/预算组件诊断，必须明确未完成 GPU 求根，不能称完整算法通过。

## 全输入与判定

局部参数全覆盖：7010 个旧输入，以及锁定 fresh_inputs.json 的 312 个层项（共7322）。全部 peak/rho/energy 区间应覆盖独立世界参照，低/高失败均记录。

根：保持18旧+88新有限模型，共106，另4无有限根模型。有限根须返回原世界参照以内的 1e−3 误差及严格符号/宽度证书；4无根分类正确。unresolved 记为失败；不删极小尾部或临界例。

保存编译源、完整输入、设备返回的区间/预算/分类及失败，检查源码 archive/hash；独立 MP 世界 covariance 参照核验。前向/预算/求根分别计 CUDA event 和墙钟，包括准备与高路径工作范围，诊断计时不冒充全 renderer 或 training 效率。

仅在本开发整体过门槛后，评估完整24视角与反向链、跨场景配对训练/留出图像及几何指标。CPU/局部GPU通过均不证明GT收益或新颖性。
