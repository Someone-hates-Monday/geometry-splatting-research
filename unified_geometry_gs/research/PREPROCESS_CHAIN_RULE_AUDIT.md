# GGGS 世界预处理反传：独立核验与三个局部修正

2026-10-12。已完成符号推导、CUDA 单基元探针、独立 float64 自动微分和方向有限差分。
生产源码未改；新变体在 `corrected_preprocess_backend.py` 中隔离构建。

## 已建立的数学结论

- 单基元法向 n=c/||c|| 的 VJP 应除以原 c 的模长。此前已有正定反例和精确仿射法向恒等式，
  见 `NORMAL_PREPROCESS_VJP.md`；本轮 CUDA 世界参数测试进一步核验了它。
- 对称正定精度 P、r=(u,v,1)、L=rᵀr、h=Pr、κ=√(rᵀPr/L)：
  ∂κ/∂u=h₁/(κL)−κu/L，v 同理。现有反传缺少第一个分子项，这会影响三维中心的导数。
  P=I、u=.5、v=0 时 κ 恒为 1，真导数 0，现有局部式给出 −.4。
- η=√(D₀/D₁) 的 ∂η/∂D₁=−(∂η/∂D₀)η²，现有式使用 −η。
  结论针对未激活 determinant floor 的光滑正定分支；原 epsilon 和 floor 暂保留。

新增两条公式及反例通过 `verify_preprocess_chain_rules.py` 的三个符号恒等式和
64 个 float64 自动微分例。法向证明与仿射恒等式仍在原文档和原脚本中。
2026-10-12 再读[官方 backward main](https://raw.githubusercontent.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/render_backward.cu)
发现相同模式；[官方 forward](https://raw.githubusercontent.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu)
确认相应前向公式。未固定上游 commit、未遍历历史，也不推测作者意图。

## 核验范围和实际结果

另有数学补充 `verify_ray_precision_integrability.py` / `results/ray_precision_integrability_v1.json`：
固定 P、κ 上游为 1、其他输出上游为 0，且不激活 FoV clamp，旧 viewport 场
g=(−κu/L,−κv/L) 的二维旋度为 (u h₂−v h₁)/(κL²)。
P=diag(1,2,3)、u=v=.5 时旋度 √10/45≠0，因此这一路径不可能是任何 C² 局部标量函数的梯度。
这是固定上游分量的可积性结论，不声称完整训练损失不存在替代目标。

P=λI 时 κ=√λ 对中心方向恒定，但旧相机中心场恰好等于 ∇[κ log cosθ]，
其模长为 κ tanθ/||t||。该人工角度项的实际优化方向还取决于上游符号。
64 个独立 AD 例及闭合回路线积分核验通过；收缩正方形上的旧场环流/面积趋于 .0702728369，
修正场最大绝对环流约 3.47e-18。不能据此把历史质量失败归因于此项，或宣称普遍场景旋转不变性。

`world_preprocess_synthetic_v1` 保留仅法向修正的首次探针结果；v2 追加组合修正。
48 个合成基元涵盖内 FoV 和 x clamp 激活两种区域、非零 image kernel=.3。
七输出块分别固定上游，检查中心、log-scale、原四元数和 opacity 四参数块。
CUDA 原局部函数与独立 float64 图比较，不复制其解析反传作为参照。

`world_preprocess_real_v1`：scan37/55/63 每场景固定相机 0013，随机种子 89，
每个场景预选 16 个过滤后轴比≤10、8 个轴比>10 的投影范围内基元，共 72。
投影范围不等于可见性；这些是已使用的开发模型，非独立场景泛化验证。
五输出块、四参数块共 1440 检查，含结构零。

两个审计使用相同误差规则 ||g_CUDA−g_AD||₂/max(||g_AD||₂,1e-5)≤.001：

- 组合 FP64 在合成与真实参数审计均全部通过；真实最大值 0.000119672。
- 只改法向的 FP32 在这 72 基元的法向检查通过，最大 0.000583839；
  射线精度问题仍然存在。增大精度但不改公式也不能修复法向和射线中心导数。
- 组合 FP32 未通过相同门槛，保留失败。真实 opacity_coef 有真导数为零的组，
  其 .08095 等数值是使用绝对下限后的误差，不能写成“真梯度相对误差 8%”。
- 真实模型全部 kernel_size=0；η=1，平方因子修正理论上不产生效果。
  此处 FP32 的非零残差主要是一个数值诊断，不能据此宣称该项在实际训练中的收益。
- 真实 forward 最大归一化差 3.865e-6，三步长有限差分中最小步的最大归一化差 1.175e-6。
  有限差分检查的是 float64 参照，不是对有硬分支/舍入噪声的完整 float32 渲染器作差分。

原始结果、生成源码和源归档位于相应 `research/experiments/world_preprocess_*` 目录；
`world_preprocess_probe.py` 构建探针，`audit_world_preprocess*.py` 运行独立参照。
完整后端的 computeCov3D / computeCov2DCUDA 函数须与探针对应变体逐字节一致后才训练。
精度提升与公式修正是不同干预，质量比较使用 legacy64 对 corrected64。

尚不能声称完整渲染导数正确、真实重建变好或方法有新颖性。主动 clamp、极端轴比、
协方差预计算/特征分解 fallback、不同 kernel 和完整损失仍有未覆盖区域。
质量试验遵循 `CORRECTED_PREPROCESS_PROTOCOL.md`，首个筛选门槛在训练前写定。

本工作使用 SymPy 与 scientific-critical-thinking skills。工作流来源：Kassis et al. (2026),
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。
