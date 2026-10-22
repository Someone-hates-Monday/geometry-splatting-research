# 高阶反向精化的共同检查点 pilot

2026-10-12，开发性实验；未作外部预注册。不改变 `VALIDATION_PROTOCOL.md` 的原有质量采纳门槛。
场景 scan55 因冻结检查暴露过较大梯度偏差而被选择，是开发场景，不能称为留出验证。

## 进入完整训练之前

- `adjquad_math_v1.json`：精确分支导数、Halley 恒等式、三个 float64 光滑例和一个拒绝的临界例。
- `adjquad_backward_v1.json`：与同源五轮对照的 24 组参数梯度全部有限且相对 L2 漂移≤0.001。
  最大值 0.000806163889161607；不是独立真导数 oracle，也不是全像素保证。
- `adjquad_step_timing_v1.json`：三场景各两视角，完整 RGB+深度前向/反向，5 轮×10 次，随机变体顺序。
  高阶四轮较五轮延迟降低 6.5%～8.7%；不含优化器、密化、PatchMatch 或 NCC。
- `checkpoint_restore_v1.json`：真实 PGSR 检查点复现旧优化器绑定导致外观参数不更新；
  research 修正先恢复外观对象再构造优化器，同一捕获状态保持一致且外观参数可以更新。
- `experiments/adjquad_smoke55_v1/manifest.json`：预热 12 步，在第 13 步提前启用正则，
  两臂分别训练到 32 步。完整状态哈希相同、首个相机 0043、每臂 20 次深度/NCC 调用。
  修改了正则时序，仅用于集成，不能说明完整配方质量。

## 完整 pilot 的固定配方

`run_shared_solver_pair.py`：30k、resolution=2、PGSR 外观、no_dn，NCC=0.6、geo=0.02、
阈值 0.9、随机近邻 1 视图，7000 步开始正则。共同预热到 **6999**，优化器仍按完整 30k 配置。
只运行一次共同预热，随后五轮控制与高阶四轮各从同一份检查点续训到 30k。
常规 tuple 之外补充保存 `filter_3D`、`xyz_gradient_accum_abs`、`percent_dense`、完整相机顺序、
剩余相机队列、Python/NumPy/Torch CPU/CUDA RNG。两臂恢复后、第一次选择相机之前，
对模型、优化器和这些附加状态作内容哈希，必须与共同检查点完全相同；同时验证优化器参数对象绑定。
这消除初始预热差异，不能强制后续 CUDA atomic 操作确定性。

恢复 hook 仅在研究 wrapper 内，针对 CPython 3.12 的固定源码锚点；不覆盖原 `GaussianModel.restore` 文件。
两臂都使用此修正。旧 NCC 对照从头训练，未使用该恢复路径，因此不能把它归因于恢复缺陷。
暂不支持 M1 router 等额外有状态模块，遇到这些配置直接报错。

## 终点与解释

两模型统一使用历史五轮提取器，voxel=0.002、trunc=4、depth_max=8、num_cluster=1；
相同官方相机中心配准，固定 evaluator seeds **0/1/2**。报告两臂 overall、d2s、s2d 原值及逐种子差值。
评估三种子仅检验抽样敏感性，不能增加训练重复数。完整 49 个训练视角报告 PSNR/SSIM，
保存相机列表与约定；不是新视角测试。源文件、隔离后端、模型和网格哈希与原始日志保留。

沿用现有质量采纳要求：平均 Δoverall≤−0.0005 mm、Δs2d≤0、Δ训练视角 PSNR≥−0.2 dB。
当前一个场景/一个训练种子即使通过，也不能宣称跨场景学术成果。
顺序固定为 control、adjquad4，完整训练墙钟只记录资源，不用于加速主张。
若质量失败，保留全套结果，不调整门槛制造通过；冻结渲染提速和条件性数学结论单独保留。
若质量通过，再进入其他场景及独立训练种子；新视角需从头按固定留出集重训。

## 数学限制

高阶反向精化估计的是底层连续根 Jacobian，不是四轮离散程序的精确梯度。
光滑、简单根、导数有界等条件下可给出二阶/三阶局部误差界；峰值接近、临界点、窄核和高 alpha
不能由此得到一致常数。粗前向的 upstream 损失导数仍可能产生一阶完整损失梯度误差。
见 `IQI_AND_ADJOINT_REFINEMENT.md` 和当前 `median_depth_note.tex`。

本研究使用 SymPy、hypothesis-generation 与 scientific-critical-thinking skills 的明确假设和证据边界流程。
工具溯源：Kassis et al. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。
