# 单基元相机法向归一化：局部反传缺失一个长度因子

2026-10-12。本地 shader 的 forward 将未归一化的相机法向 c 变为 n=c/‖c‖。
backward 却在得到 n 后，用 n 的倒数模长构造归一化 VJP；精确算术下这个倒数恒为 1。
同样模式也见于本次读取的
[GGGS 官方 backward 源码](https://raw.githubusercontent.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/render_backward.cu)，
其[官方 forward](https://raw.githubusercontent.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu)
也归一化相机法向。此次仅核验 main 页面，不声称已遍历上游历史或确认作者意图。

对于任意非零 c 和上游 g，标准欧氏归一化的正确 VJP 为

\[
\nabla_c L=\frac{g-n(n^\top g)}{\|c\|}.
\]

证明：dn=(I−nnᵀ)dc/‖c‖，将其代入 dL=gᵀdn 即得。用 ‖n‖ 替代 ‖c‖
会使这一路径的导数多一个 ‖c‖ 因子；仅在 ‖c‖=1 或对应切向上游为零时一致。
这不同于渲染后对混合像素法向作归一化的路径，当前检查针对单基元预处理。

## 正定、非奇异反例

相机坐标 (0,0,1)，精度矩阵 P=[[2,0,x],[0,1,0],[x,0,1]]，在 x=1 邻域正定。
GGGS 的仿射预处理给出 c=(−x,0,−1)。对损失 L=n_x：

- 正确导数：−(1+x²)^(−3/2)，x=1 时为 **−1/(2√2)≈−0.35355339**。
- 现有局部式：−(1+x²)^(−1)，x=1 时为 **−1/2**。

该精度矩阵有正尺度、正交旋转的谱分解，不依赖退化基元。精确符号恒等式确认两式
比例为 √(1+x²)；48 个 float64 随机非零向量例还通过自动微分及方向有限差分。
脚本和实际结果分别为 `verify_normal_preprocess_vjp.py`、
`results/normal_preprocess_vjp_v1.json`，其中记录本地源码 SHA256 和官方核验 URL。

## 仿射相机法向的更简单等价式

令 P 为相机坐标中的正定精度矩阵，d 为基元中心对应的单位视线方向。
把现有 nJ、plane 和 factor_normal 代数展开，可以得到

\[
c=-\frac{Pd}{d^\top Pd},\qquad n=-\frac{Pd}{\|Pd\|},\qquad
\|c\|=\frac1{|n^\top d|}\ge1.
\]

证明：设 r=(u,v,1)、ρ=√(1+u²+v²)、h=Pr、b=rᵀh、d=r/ρ。
代码中的 plane 前两分量满足 plane_i=ρ²h_i/b−r_i。
将其与 factor_normal=z/ρ 代入相机变换 nJ，就有 c=−ρh/b=−Pd/(dᵀPd)。
正定性保证分母正；归一化即得 n。最后由 Cauchy–Schwarz 及 ‖d‖=1 得长度恒等式与下界。
因此该缺失因子恰好是视线与法向夹角余弦的绝对值，而不是无关的常数。
这只是当前仿射基元模型内的法向恒等式，不等于真实场景表面法向的视角一致性。

其世界参数微分可以直接写成

\[
dn=-\frac{(I-nn^\top)(dP\,d+P\,dd)}{\|Pd\|}.
\]

正尺度、正交旋转、相机 z>0、正定分支下成立；若 FoV clamp 激活，必须继续链式求导
该 clamp。P 的整体正数缩放不改变 n，因此单独法向损失对三个统一 log-scale 平移的
导数和为零；这一不变量可用于未来回归检查，不能施加到完整 RGB/几何损失上。

`verify_affine_normal_identity.py` / `results/affine_normal_identity_v1.json` 已验证三个精确
分量恒等式，以及 64 个 float64 例中现有 nJ 计算图与简化计算图对中心、尺度、
规范化四元数的 VJP 一致。这仍不是 CUDA 世界参数导数参照，也没有运行时或训练结论。

## 结论边界

这是相对 forward 的局部导数不一致，尚不是完整渲染导数审计、重建改进或新方法。
几何损失本身的法向上游为零，因此它不解释本轮发现的几何尺度梯度重复波动。
高精度中间量诊断也没有修正这个因子；数值重复性和解析正确性必须分别验证。
本轮已完成独立的 CUDA 世界预处理 VJP 参照：组合 FP64 修正在 48 合成和 72 真实模型基元上
通过既定门槛；原式法向和射线精度中心导数的问题不因提高精度而消失。
具体范围及新发现的两条链式法则见 `PREPROCESS_CHAIN_RULE_AUDIT.md`。
质量对照尚须完成，遵循 `CORRECTED_PREPROCESS_PROTOCOL.md`。
原始生产源码保持未改。

本核验使用 SymPy 与 scientific-critical-thinking skills。工具溯源：Kassis et al. (2026),
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。
