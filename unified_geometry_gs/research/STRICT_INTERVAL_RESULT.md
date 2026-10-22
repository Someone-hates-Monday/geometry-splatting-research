# 严格数值核心：host 与 CUDA 端点组件通过，完整设备求根未完成

2026-10-14。上一轮 CPU 条件精度证书通过后，本轮实现可共享的有向区间与对数尾部核心。`STRICT_INTERVAL_PROTOCOL.md` 锁定输入；完整 GPU 目标继续见 `GUARDED_CUDA_PROTOCOL.md`，尚未达成。

## 证据改变与实现

CUDA12.8 原始文档 13.1 明确：standard mathematical functions 的误差边界由非穷尽测试生成，不是保证界。不能将某个假设 ULP 扩张包装成严格全输入证书。这里不用 standard exp/log/log1p/expm1；以基本 IEEE 有向运算、位级 range reduction、级数及余项上界实现。host 使用 MSVC /fp:strict 与动态舍入；device 使用 CUDA double intrinsic，关闭 fast_math、FMA contraction 和FTZ。

`guarded_interval.cuh` 中 log 用25个 atanh 奇次项；exp/log1p/expm1 用24项级数及明确余项。ln2 相邻 binary64 端点由50项精确有理 atanh 级数和余项独立核验。一般级数/区间与range reduction不是新的数学方法。

设 g=exp(ell)，b=-log(1-g)/2，ell<-40，则

`log(b) in [ell-log2, ell-log2+2^-56]`。

因为 57*ln2_upper<40，g<2^-57；而 `1 <= -log(1-g)/g <= 1/(1-g)`，故 log修正不超过g/(1-g)<2^-56。它保留尾部的 log magnitude，即使 b 无法由 IEEE binary64 标量表示。

对残差中的正负项，先选共同 log scale 再有向求和；缩放值的区间符号就是原残差符号。小于 exp(-700) 的项保留正上界2^-1009，因为1009*ln2_upper<700；不置零。不确定、峰位区间交叉或无严格符号仍返回 unresolved。

## 完整固定检验

- `strict_interval_host_v1` complete/exit0：seed13023 的774个scalar检查全部覆盖独立 Windows Decimal 高精度参照；全部106有限模型的212端点符号/区间通过。
- `strict_interval_cuda_v1` complete/exit0(session25852)：RTX5090 Laptop、sm120、CUDA12.8、gcc/g++11，全部212端点通过。compiled核心与host为同一header，原代码/数据归档保存。
- `results/strict_interval_audit_v1.json` 独立重新计算212个scaled参照、精确有理常量并核验两组archive/产物/编译产物hash。
- 输入是 `guarded_precision_v2` 已保存的low/high系数区间与CPU选择的bracket。复用之前独立世界covariance MP240端点残差；没有由MP真根生成新端点、改变seed或删失败样本。
- CUDA212端点批量的五次event计时中位数 **222.9916504ms**，含输出分配/fill及kernel，不含源参数预算、transfer、选择和求根。它是局部参考组件计时，未与完整renderer比较，不宣称效率收益。

host run后仅给未执行的CUDA binding追加same-device guard，数值header、host cpp和exe未改。host archive保存旧binding版本；CUDA archive记录新版本。此已知源漂移不冒充当前全部源等于历史源。

## 执行中断的实际原因

先前两次只读WSL查询都被自动审批在deadline处终止，没有执行、没有训练进程handle；工具明确不能据此认定动作不安全。用户随后明确允许本地WSL/CUDA诊断，实际编译与查询完成。Windows读到GPU显存当时0MiB；没有GPU死锁证据。当前已启动的host/CUDA诊断均终态，没有新训练启动。

## 下一步与限制

**这不是完整GPU算法通过。** 尚待设备预算、原相机mean/offset、终端mass分类、条件选择/回退和完整求根；尚无forward/backward整图、GT质量或新颖性结论。现有参考核心开销必须实测并优化，不能仅凭符号通过直接移入训练。

下一步从设备预算与便宜的充分认证规则入手，严格核心保留作难点验证参照；对每个失败/不确定像素走可验证回退，再测完整成本与所有固定输入。通过整体开发后才继续24视角、梯度、跨场景配对训练/留出图像/几何评价。任何简化不能改变原物理根目标或删去临界例。

严格区间结论依赖编译器与IEEE基本intrinsic语义，未做proof-assistant或全CUDA指令形式验证。device未单独跑全部774 scalar输入，仅在上述212端点上验证组合路径。

原始来源：[CUDA12.8 Mathematical Functions](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#mathematical-functions)，13.1与double intrinsic表。文档只支持相关精度边界说明，不支持本项目GT或效率结论。
