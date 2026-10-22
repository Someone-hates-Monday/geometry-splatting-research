# 共享数值核心：固定 host 核验，CUDA 执行未完成

2026-10-14。WSL 只读查询两次因自动审批超时未执行，未据此断言 GPU 或训练故障。继续完成 Windows C++ 可核验的 shared host/device 数值核心；这是 GPU 实现的一个组件，不替代整个 GUARDED_CUDA_PROTOCOL 门槛。

CUDA12.8 原始文档明确标准数学库误差边界来自非穷尽测试、非保证界，故严格路径不使用 exp/log/log1p/expm1 libdevice。以基本有向算术实现 log 的 atanh 级数（25个奇次项）、exp 的 range reduction 和24项 Taylor、log1p/expm1 的24项级数与余项界。ln2 的两相邻 binary64 端点用精确 Fraction 级数独立证明。host 用 MSVC /fp:strict 与动态 fenv 舍入；device 用 CUDA intrinsic。不能用 host 结果声称 CUDA 已验证。

同一 header 用 signed scaled interval 求残差符号：指数尾部保留 log magnitude，缩放后低于 exp(-700) 的值保留明确上界 2^-1009；不是置零。logg<-40 的 log-tail 解析修正区间为 [0,2^-56]，由 ln2 上界及 geometric-series 不等式证明。普通 scaled bounds 可以只有0下界，仍保存正的上界；区间重叠必须 unresolved。

锁定 scalar generator seed13023：256个 positive double log 输入（含subnormal和最大指数族），256个 exp 输入（范围[-60000,700]及range-reduction/threshold端点），128个 log1p 输入（[-.124999,.124999]及最小subnormal正负数），128个 expm1 输入（同small+negative大指数），6个 IEEE 舍入诊断。共774项，输入先存盘。

全部106个已有 finite 模型两侧端点（212查询），直接读取 guarded_precision_v2 的已保存low/high系数区间及 candidate bracket；不以MP真根生成端点。验证正负符号和scaled区间，保留全部旧临界/微小offset/真实模型。使用独立MP240世界端点残差记录作参照；标明它是先前独立参照的复用，不是新的数据或新的世界求根。fresh种子不改，不选子集。

门槛：scalar输出区间覆盖Windows Decimal独立高精度参照，全部212端点返回有认证严格符号且scaled区间包围参照（Decimal80+独立既有MP240记录精度范围内）。任一 unresolved 或遗漏均失败。保存host编译源、参数、结果和失败；固定ln2/截断常量的精确有理证明随结果输出。

仍待真正CUDA编译与执行、设备误差预算、相机offset、完整GPU求根/分类、成本、完整renderer/后向与GT训练验证。此组件通过也不能宣布总目标或GPU开发完成。
