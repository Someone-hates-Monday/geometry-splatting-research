# TGS 细分静态审查：积分保持与局部重分布

2026-10-16，Asia/Shanghai。状态：固定源码阅读与条件性数学分析完成；没有编译、调用 renderer、数值实验或训练。遵守用户“先不进行尝试”的限制。本文不是新颖性证明，也不提供真实图像/几何收益证据。

## 1. 来源与版本

官方主库固定为 `685ff270a8273afc3a2b094ae3148dd0b98e31e9`；主库 gitlink 指定 native 子模块为 `6d61f4c0a571c635e37ffb740250706fa30a5541`。本次只读查询的 main 与旧 2026-10-13 审查记录一致。旧记录确实已固定提交，2026-10-16 续接入口中的“main 未 pin”不完整：针对细分的新阅读没有沿用旧 pin，而非项目从未 pin。

`sources/tgs_subdivision_static_v1/manifest.json` 保存 19 个源码/说明/许可证文件的原始字节 SHA256、Git blob SHA1、精确下载 URL 和行数，总计 195687 字节，另有两库的 commit/tree 元数据。这是选定来源归档，不是完整可构建 checkout。主库对子模块的 gitlink 已核对；没有修改工作中的 GGGS 环境。

两库均有 Apache-2.0 的 `LICENSE.md` 和原 Gaussian-Splatting 的 `LICENSE_GS.md`；源码头区分修改与原代码的许可证。这里只记录文件，不把整个派生库概括成单一许可证。

来源：

- [TGS v1，§3.1–3.3 与附录 A](https://arxiv.org/html/2505.19175v1)。
- [固定主库](https://github.com/trianglesplatting/triangle-splatting/tree/685ff270a8273afc3a2b094ae3148dd0b98e31e9)。
- [固定 CUDA 子模块](https://github.com/trianglesplatting/diff-triangle-rasterization/tree/6d61f4c0a571c635e37ffb740250706fa30a5541)。

归档过程记录：首次 sandbox 网络请求失败；审批后的查询成功。初版脚本用 commit SHA 请求 trees 后错误地假定响应顶层 sha 为实际 tree SHA，触发校验失败；已改为从 commit 元数据读取真实 tree SHA 后查询，未放宽校验。一次重试在自动审批服务额度不足时未执行；人类随后要求继续，同样审批路径恢复。之后下载 auxiliary.h 遇 IncompleteRead，进程 60404 以 exit1 结束；文件未写入。后续进程 27807 复用并重新验证完整文件，仅补缺，最终 exit0。没有重启任何训练。

## 2. 固定实现所做的事情

下列行号均指归档的固定提交，不使用旧 web 缓存 main 行号。

- `triangle-splatting/scene/triangle_model.py:655–679`：大三角形用三条边中点划成四个，复制 SH、raw opacity 和 raw sigma，新 mask 设为 1。
- 同文件 `623–652`：小三角形分支保留一份并加一份平面噪声副本，两个 opacity 变为 `1-sqrt(1-o)`；这与四分分支应分别研究。
- 同文件 `726–804`：按 opacity 或 inverse-sigma 抽样，`image_size` 与 `split_size` 决定分支；增加子基元、重绑 optimizer，最后删除选中父基元和 dead 基元。不能把增密事件当作纯光学参数变换。
- `triangle_renderer/__init__.py:96–97`：opacity 还乘以带直通梯度的二值 mask。父 mask 不活跃与子 mask 重置的情况需独立处理。
- `cuda_rasterizer/forward.cu:141–146,225–227,355–359`：有 opacity、掠射角和投影尺寸剔除；尺寸分支包含最大顶点半径小于 1 像素或到边的内心距离绝对值小于 1 像素。细分后可能触发不同剔除分支，发生率未测量。
- 同文件 `279–348,388–396`：按 opacity/sigma 收紧 tile 范围并处理退化/空范围。此选择与逐像素 alpha 阈值是不同环节。
- 同文件 `458,543–573`：像素点采样，边界外跳过，归一化边距离幂函数；alpha cap 为 .99，alpha 小于 1/255 跳过，`test_T<1e-4` 时终止且当前项不累加。此函数未计算像素面积积分。
- 同文件 `33–48,410,418`：SH 的方向取相机至三角形中心；深度取中心的 view-z。复制 SH 系数不保证子基元 RGB 一致；共面也不保证中心深度相同。
- `cuda_rasterizer/rasterizer_impl.cu:105–115`：tile key 含该中心深度。分裂可能改变与第三方基元的排序，幅度和影响未测量。
- `train.py:171,196–232,252–254`：backward 后进行增密，然后 optimizer.step。`scene/triangle_model.py:529–577` 创建新的 Parameter 并处理 moments。将来比较恢复步数必须控制参数重绑、梯度去向、Adam 状态和父基元选择。

两项旧疑问已解决：固定版本 `utils/loss_utils.py:30–41` 确有 `equilateral_regularizer`，实际返回三角形面积；不能据旧缓存未找到此函数认定代码缺失。`arguments/__init__.py:79,104` 的类默认值分别是 depth_ratio=1、set_sigma=1.16；实际运行配置仍可能覆盖它们。

## 3. 一个改变选题的解析结论

这是基于论文窗口的推导，不是实际 CUDA 输出，也不声称首次提出。

令 K 为任意非退化二维三角形，面积 A，内心 s、内切圆半径 r>0。内部点到三条边的最小距离记作 d(x)，并令 u(x)=d(x)/r。论文窗口为 I(x)=u(x)^sigma，sigma>0；外部为零。

对 0<=t<=1，三角形内平行收缩满足

    {x in K : u(x)>=t} = s + (1-t)(K-s),
    area{u>=t} = A(1-t)^2.

证明：若各外法向单位边函数为 L_i，K={L_i<=0} 且 L_i(s)=-r，则 x=s+(1-t)(y-s) 满足 L_i(x)=(1-t)L_i(y)-tr。因而 L_i(x)<=-tr 等价于 y 属于 K（t=1 单独为内心）。面积按比例平方缩放。

所以均匀面积测度下 u 的密度为 2(1-u)，从而

    integral_K o I(x) dx
      = 2 o A integral_0^1 u^sigma(1-u) du
      = 2 o A / ((sigma+1)(sigma+2)).

该系数与三角形形状无关。若有限个非退化子三角形 K_j 的内部不相交、并集为 K，且复制同一 o、sigma，则几乎处处最多一个子三角形有非零影响，因此 alpha 合成退化为求和，得到

    integral_K [1-product_j(1-o I_j(x))] dx
      = sum_j 2 o area(K_j)/((sigma+1)(sigma+2))
      = integral_K o I(x) dx.

**因此理想连续模型中的四分已保持总 alpha 积分；局部窗口重分布不等于总量损失。** 不需要子三角形相似，故理想透视投影下，只要整个平面三角形位于同一有效投影域、没有跨投影奇点且投影仍为分区，上述结论也成立。不能假定一般透视投影是均匀缩放；本推导不使用该假定。

更一般地，任意相同且可积的点态响应 g(u) 都满足 integral_K g(u)=2 A integral_0^1 g(u)(1-u)du。因此同一 opacity/sigma 下单独施加 cap 或 alpha cutoff，仍不会破坏这个连续、完整区域的积分恒等式。它们会改变局部分布；实际采样、尺寸剔除、tile 筛选、mask、外部可见性与早停不能由这个恒等式消除。

局部保持则不成立：取父三角形严格内部的一段子三角形公共边。所有子窗口在该边为零，而父窗口为正（o>0、sigma>0）。固定有限参数时，由连续性，该差异延伸到某个正面积邻域，不能仅以“边界是零测集”排除。但是邻域宽度和积分误差可能随 sigma 趋零而趋零；这不是统一的误差下界，更不是图像质量损害证明。

在孤立三角形、恒定相同颜色和恒定背景条件下，积分 alpha 保持也意味着 RGB 空间积分保持；不意味着逐像素 RGB、L1/L2 误差、图像质量或梯度保持。SH 随中心变化和遮挡排序都不满足这个简化条件。

## 4. 克隆分支和先行工作限定

零位移、无 cap/cutoff 的两个完全重合子基元，令 a=1-sqrt(1-o)，共享窗口 I，则

    alpha_children - alpha_parent
      = [2a I-a^2 I^2] - o I = a^2 I(1-I) >= 0.

所以 opacity 的平方根规则只在 I=0 或 I=1 等情形点态相等；软窗口内部会改变覆盖。这一现象与已知的中心校正克隆问题一致，并不是新的贡献；实际小三角形分支还加入了位移，不能直接套用此式作实测预测。

[3DGS-MCMC v3 §3.4、附录 A](https://arxiv.org/html/2404.09591v3) 已讨论保持渲染的近似 relocation、峰值 opacity 校正及穿过中心的 1D 切片积分匹配，并明确实现是近似的，也讨论 Adam moments。这里的三角形二维面积积分恒等式与其高斯切片构造不同，但“降低增密扰动”和“中心 opacity 校正不足”本身已有先行工作。不能把它声称成精确全图不变，更不能仅由渲染差异推断某未定义 Markov 链不平稳。

[TGS+ v1 §3.1–3.4](https://arxiv.org/html/2509.25122v1) 与 [MeshSplatting v1 §3.2–3.5](https://arxiv.org/html/2512.06818v1) 已有共享顶点、不透明化策略及中点细分；两者有关联的研究脉络，不视作两次独立验证。其完整官方 CUDA 尚未在本审查固定或运行。

[Efficient Confidence-Aware Densification](https://arxiv.org/abs/2610.13541) 的摘要已提出固定预算内按误差/置信度分配基元。这里只核验摘要范围，未完成全文/源码审查；普通误差驱动分配不能据此声称新颖。

## 5. 据此修订策略

候选目标应从“补偿细分丢失的总透明度”收窄为“在既有积分保持的基础上，控制对任务有影响的空间重分布及离散实现变化”。这仍然是假设，未选定新算法。

未来若用户恢复尝试，先分开四类问题：连续窗口重分布；离散采样/尺寸剔除；中心 SH 与遮挡排序；optimizer 状态变化。首个机制诊断应以实际基线配置为准，同时保留恒定颜色、固定排序、无尺寸剔除的简化参照以区分原因。精确公式的小例子只能检查实现，不能替代真实事件影响。不能直接做 opacity 放大或把“裂缝”命名成已观测缺陷。

只有真实事件显示可测且持续的负担，才值得比较受约束初始化/窗口修改。需要区分保留有益探索与减少恢复成本；所有 arm 固定相同父基元、预算、camera、optimizer 策略，且报告最终 GT/留出图像和完整时间。若影响很小或原扰动有益，则停止此方法线，保留理论诊断。

当前用户限制仍是分析阶段；没有任何 TGS 实验启动。原 history-gap 30k 负结果、所有既有协议和生产代码未改；研究总目标未完成。

## 方法与检索范围

本次采用 literature-review skill 做已知相关原文的单人定向审查，结合固定源码；不是系统综述、独立同行评审或新颖性认证。没有向外部服务提交本地模型、数据或未公开公式。

程序知识来源：[Kassis, Agarwal, He, Patel, Brueckner (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)，元数据于本次检索核对为 v2。引用该工具方法不为本文数学或科学结论背书。
