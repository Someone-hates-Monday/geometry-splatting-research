# 稳定窗口 v2：固定采样通过，整图边界反例使推广暂缓

截至 2026-10-20，CUDA 编译与固定输入前后向正常结束。未启动优化器、训练或性能计时。
本阶段的结论是**局部坐标和保守分配解决了已知压力案例，但尚未得到可推广的整个渲染修正**。
不得只引用已通过的 79 案例而省略随后发现的边界反例。原 v1 整体失败结论和证据保持不变。

冻结协议 `TGS_STABLE_WINDOW_PROTOCOL_V2.md` SHA256
`697f6be78d7e7cc2a993a6035b46dc95edafc8c1f391c2e34417c6b90c2aaa55`；
输入计划 `experiments/tgs_stable_window_v2/input_plan.json` SHA256
`6b74f15226c3a613a3db22926faf1b61114f6999bbe5756c95553caec3c5329a`。

## 实现及核验范围

隔离生成两个模块，共用新前向：双精度投影、整数 anchor、局部 float32 顶点/边方程、
面积/周长内切圆半径；按 alpha>=1/255 的同心缩小三角形分配 tile，使用有符号 floor、
exclusive upper=floor(max/B)+1，并加一像素外扩。外扩只是经验数值保护，未证明普遍 IEEE 覆盖界。
保留原 opacity<.01、视角、radius/extent 等 primitive 剔除规则。直接 Gamma 与边余切重建 Gamma
两版共享此新前向，不再要求与旧前向整图相同；旧整图差异全部记录。

两个模块 SHA256：direct `eb0b40a5d9c910707e84faaadf3d05487c3068ae1db9502398c9b9a623195450`；
contraction `c55b352ed3b02a290a18e88e3587dbecf2f803160cbf0a212a96172f33bcf6d0`。
训练为 false；PyTorch 2.12.0.dev20260314+cu128，CUDA12.8，sm120。
原始 452 源文件及前阶段封存 293 文件的哈希检查通过。

## 冻结采样的结果

全部 18 旧案例、48 原压力案例、1 有理反例及 12 新旋转/裁剪/不同正顶点深度案例均保留。
902 组 VJP，5184 条原生有限差分探针步长记录；两模块的 79 份整幅 RGB、aux、radii 逐位相同。
所有冻结采样的前向/VJP/支持稳定门槛通过，未调整阈值。最大采样前向误差 9.682787e-7；
最大 geometry VJP 相对误差 1.285837e-4（aux distortion）；scalar 6.33336e-5，SH 3.41525e-7。
48 压力案例前向最大误差由 .0387783 降至 1.223397e-7，geometry 最大相对误差约1.90e-6。
v1 的 stress02/15/29/30/40/42 在新版本均通过，旧失败记录没有覆盖。

每次前向从 GPU GeometryState 末尾读取 rect_min/max，并用 tile 实例数校准结构解析。
5184 条 FD 均检查 base/plus/minus 的逐像素逐 primitive tile 成员关系、像素 alpha 支持和
CPU 光滑分支。新 12 案例的全部 2986 个理想有效整数采样，tile 和 alpha 支持均无漏点。
这不等于所有像素/所有输入的普遍保证。

原生 FD 数值误差没有冻结精度阈值，不能把分支稳定称为 FD 全部达到1e-3。
direct 2592 条 FD 归一化误差中位 1.17127e-4，90分位 .00243669，99分位 .0457225，最大2.70110；
contraction 共用同一前向。最差相对例 holdout11：预期 .000366405，native FD .00135610，
CPU FD .000366405；端点前向数值差被除以小步长放大。
最大绝对例 holdout06：预期 -426.82365，native FD -462.11168，CPU FD -462.17097，
其中 CPU有限步长/可表示探针路径项约 -35.34732，native-minus-CPU约+.05929。
`finite_difference_decomposition.json` 保存每条有符号分解，未删任何探针。

## 扩展整图诊断发现的问题

冻结门槛审计后，检查了 18 旧案例的每一个 RGB 像素，属于事后诊断，不冒充新的留出确认。
其中16案例新整图对独立CPU64参考的 max error<=1.71e-7；但 alpha_saturated 和
early_stop_skips_fourth 各有39像素误差>3e-5，最大分别 .442398 和 .403816。
因此 `independent_audit.json` 的 protocol gate=true 保留，后续 `promotion_decision.json`
明确 performance_measurement_allowed=false。没有以狭窄采样通过为由开始计时。

alpha_saturated 的 sigma 实际为 .019999999552965164，opacity .9998999834060669。
在像素(176,44)、(188,35)，从原 float32 世界点和名义 EPS=1e-7 出发，精确有理计算得到
phi=-377/200000010，u=377/8000000000；80位 alpha=.71354536876009056…，两点相同且均严格在内。
GPU tile 均存在，native alpha 却是0和 .7236766815，排除了漏 tile 作为这两个像素的解释。

独立有理审计还确认：两个像素在存储的局部 float32 顶点三角形中也都严格在内；
存储 normal/offset 若按精确实数求值，两点 max phi 都是 -45/16777216。
所以**顶点量化不能单独解释零 alpha**；缓存系数舍入和逐像素乘加求值都需区分。
原诊断证书的“vertex quantization loses margin”描述较窄，以
`small_sigma_independent_audit.json` 的这项分解为准。
同一缓存直线上的精确切向不变性被逐像素浮点求值破坏，这是可复查的反例。

## 数学含义与下一步

`TGS_SMALL_SIGMA_STABILITY_NOTE.md` 给出小指数 Hölder 界、局部条件数及选择性高精度准则。
它们是标准数学事实的应用和本地实现反例，不宣称已证明方法新颖。
小 sigma 下边界 phi 的微小数值误差会被 u^sigma 放大；一味减小FD步长或只修tile不能解决。

下一版本应保留全部79案例，加入这两张完整图及精确边界证书，先尝试选择性高精度边方程求值：
把“原投影误差”“float缓存系数误差”“像素乘加误差”分别隔离；必须从足够准确的几何/系数
求值，不能假设只把已有错误 float 系数转 double 就恢复原窗口。
再建立 alpha cutoff、cap、early-stop 的区间/分支证书，所有不确定像素应保留并报告。
不得直接夹 sigma>=1、删边界像素或改模型来伪装数值修复。
通过扩展正确性门槛后，才比较同一可靠前向下 direct/contraction 的实际开销。
图像质量、几何质量、训练收益和学术新颖性均尚未证明；本轮不更新 LaTeX、原始库或模型。

编译会话88211、固定输入88151、独立审计77012及单三角形证书命令均exit0，无待等训练/编译。
