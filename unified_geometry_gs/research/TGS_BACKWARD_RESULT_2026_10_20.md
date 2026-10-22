# TGS 几何反传核验：两个缺项与隔离修正

本轮在四个光滑 RGB 分支固定输入中，确认 pinned native 几何 VJP 不等于其前向的导数。
两个原因分别是像素坐标 Jacobian 缺少 W/2、H/2，以及窗口内切半径归一化的链式导数缺失。
隔离修正后与独立解析导数的最大相对误差为 **7.424e-7**。它是基础实现验证结果，
尚未建立真实训练质量收益、效率优势或论文级新颖性。整体科研目标仍未完成。

## 1. 输入、版本与对照

核验基于作者原生仓库的固定提交
[`6d61f4c0a571c635e37ffb740250706fa30a5541`](https://github.com/trianglesplatting/diff-triangle-rasterization/tree/6d61f4c0a571c635e37ffb740250706fa30a5541)。
452 个归档来源文件重新核验 SHA256；原始归档、GGGS 安装、模型和 LaTeX 均未修改。

`shfix_control` 保留几何反传原实现，只采用已独立证实的 SH 构造修正。
本轮用预计算常量颜色，所以 SH 分支根本不参与该对照。新增两个独立模块：

- `scale_only`：只补像素 Jacobian。
- `geometry_fixed`：再补完整的内切半径导数；前向和 `scale_only`、`shfix_control` 相同。

四场景：正绕序、反绕序、320×192 非正方形图像、不等顶点深度及旋转平移相机。
每场景一三角形、四个固定整数像素，sigma=1.3、opacity=0.65，带正负权重的 RGB 线性损失。
它们是 activated 参数，尚未单独检查模型内部 exp/sigmoid 参数化与优化器。
基准 alpha 范围约 0.1394–0.3483，最大边与次大边的距离差至少 7px，内切半径至少 35.34px。

九个位置分量及 sigma、opacity 各用 h=0.01、0.003、0.001、0.0003 中心差分。
分母使用实际 float32 正负扰动间距；三版本共 **528 个参数/步长对照**，全部保留。
每个正负扰动均通过 active edge 不变、远离 alpha 下限/饱和、native 三角形可见检查。
这不能保证所有未取样像素都不发生分支切换；它们的损失权重均为零。

第一轮矩形场景的一个测试点未满足事先设定的 active-edge gap>2px，程序在运行该场景前
中止；已保留 `initial_fixture_attempt.md` 与 `initial_partial_progress.json`。
将该点在所有场景统一移至更安全位置后，完整重跑。没有根据梯度是否匹配来筛除案例。

## 2. 数值结果

几何误差定义为 ||g_native-g_exact||2 / ||g_exact||2；不是图像误差或质量提升百分比。

- 原几何反传：四场景误差 **0.989203–0.990077**。
- 只补像素 Jacobian：误差 **0.441332–0.493156**。
- 两项完整修正：误差 **3.286e-7–7.424e-7**。

仅缩放修正后，即便事后选择最优单一倍率去拟合正确梯度，仍残留 **29.88%–31.11%**
的相对误差，梯度余弦约 0.9504–0.9543。这说明余下偏差涉及方向，不能解释成单纯学习率。
此最优倍率仅用于诊断，未被用作算法或超参搜索。

sigma/opacity 两分量联合相对误差在全部版本/场景中不超过 **5.413e-7**。
所有版本的四张完整前向 RGB 图像均逐像素一致，最大差 **0**。
完整修正版重复反传最大逐分量差 5.961e-8，不能声称无原子累积不确定性。

独立依据：

1. CPU float64 的完整投影、内心、边法线窗口公式，用 Torch autograd 得到参考 VJP。
2. CUDA 前向自身的中心差分。在 h=0.01 时与参考完整梯度的联合相对误差为
   2.288e-5–3.269e-5；保留更小步长的浮点噪声增大，最小步长约可到 1.009e-3。
   不能把解析比较的 7.424e-7 当成有限差分本身具有同等精度。
3. Windows NumPy 审计直接使用面积/周长及边距离的手写解析导数，**不依赖 Torch/CUDA**。
   与 CPU autograd 的最大相对差 9.361e-16，重算全部保存的 CPU 差分/标量损失并通过。
4. 原 VJP 与“冻结半径且移除像素 Jacobian”的错误公式匹配误差至多 6.762e-7；
   只补缩放的 VJP 与“仅冻结半径”公式匹配误差至多 7.576e-7。

这同时支持前向公式、偏差定位及修正，不只是把测试期望值改成 native 的输出。

## 3. 数学依据与实现

完整推导见 `TGS_BACKWARD_DERIVATION.md`。对未饱和 alpha=o*(-phi/r)^sigma，
缺失项为 -(sum_pixels dL/dalpha * sigma * alpha) * grad log(r)。
用 r=abs(二倍有向面积)/周长计算导数，避免显式展开内心坐标的长链式公式。
像素坐标梯度再分别乘 W/2、H/2，随后通过原针孔 Jacobian 回传到三维顶点。

还证明了单像素同心缩放的不变性：phi、r 同比变化，正确 alpha 方向导数为零；
冻结 r 会产生非零 sigma*alpha。此零空间变化也排除了普遍用单一倍率补偿的可能。

只使用原来已分配但未使用的 P×1 `dL_dsigma_factor` 临时 buffer 累积 dL/d log(r)，
每个参与反传像素新增一次 atomicAdd。没有测量真实场景运行时间，不能宣称无开销。

模块 SHA256：

- control：`4fa625a41b94545010a579e9c98eb68fb557023c2429be890eaf20d750b00ef1`
- scale_only：`a410e69d32d6667a4a48be9d47ffa775395cbba0c7583b00879349cd3a2273f1`
- geometry_fixed：`955dd08a6082063253e188cd7504ab78cbbad8b60411c0b5c1e26d56cbf662c2`

环境：WSL Ubuntu、Torch 2.12.0.dev20260314+cu128、CUDA12.8、gcc11，RTX5090Laptop sm120。
两次编译及两次完整 CUDA 核验已结束，scale_only session27868、geometry_fixed session22131
均 exit0。不存在待完成训练，不应继续轮询或重新启动这些句柄。

## 4. 可复现入口

研究目录下：

```
python probe_tgs_backward.py
python probe_tgs_backward.py --scale-only
python probe_tgs_backward.py --fixed
python audit_tgs_backward.py
```

前三条用已配好的 WSL `uggs` 环境；后一条也可在 Windows NumPy 环境运行。
前向输入、RGB/aux 输出、全部 VJP 为 `experiments/tgs_backward_v1/<variant>/*.npz`，
四步长的全部正负损失、样本及边界间隔为各版本 `results.json`；独立汇总为
`summary_audit.json`。原 control 执行脚本快照也保存，后来的变化仅为增加报告字段和 CLI。

## 5. 来源核查及结论边界

本轮另读取了作者仓库公开 [main backward](https://raw.githubusercontent.com/trianglesplatting/diff-triangle-rasterization/main/cuda_rasterizer/backward.cu)，
页面仍显示相关公式；没有据此建立 main commit 的逐字节身份。
[native PR 列表](https://github.com/trianglesplatting/diff-triangle-rasterization/pulls) 展示三个硬件兼容相关标题，
没有据标题排除 PR 内容或全部先行工作的覆盖；main repo issues 只读了首页。
检索 `"trianglesplatting" "gradient" "inradius"` 和 `"diff-triangle-rasterization" "gradient"`
没有取得可直接支持新颖性的原始论文。未使用搜索所得第三方自动代码解释作技术证据。
因此结论严格限于固定提交和已测分支，不能宣称首次发现或所有当前 TGS 版本都存在该问题。

未覆盖：多三角形遮挡合成、背景不为零、SH 对几何的混合梯度、alpha cap、max ties、
早停/可见性/排序切换，以及 depth、normal、median、distortion 辅助输出。
尚无训练、真实数据损失改进或效能测量。这也不能直接解释 GGGS 先前负结果，二者代码路径不同。

下一步仍在已授权的小诊断范围：先扩展固定深度排序的多三角形合成与 SH 混合梯度，再用冻结
真实 checkpoint 的确定子集检查 VJP；把饱和/边界约定单独列出，不掺入光滑分支误差。
上述门槛通过后才能设计冻结训练对照。完整训练仍不在本轮授权范围，失败的 train_maximin4 v1 不重启。
