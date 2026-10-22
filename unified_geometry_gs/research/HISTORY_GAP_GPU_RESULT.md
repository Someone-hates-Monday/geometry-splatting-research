# 延拓平方历史差：GPU正确性实现完成，效率与训练待验证

2026-10-15。上一goal turn是原参数链、真实共享更新和无根C1延拓的实质进展；
本轮实现了独立GPU loss/伴随和完整source-preserving raw链，不启动训练，不改生产源码/模型。

CUDA核在GPU从原始p/rho/E/AMP独立判断前/尾/中间分支；前根loss0、尾部精确用(log4-Omega)^2，
包括无有限根的情况；其余在[minp,maxp]独立二分。用补偿prefix乘积/和、缩放尾项、
峰前/峰后density分开计算伴随，不输入CPU/MP根或梯度，也不先形成巨大median Jacobian。
Float64符号是数值判断，不是区间认证；tail分支本身不声称数学无根。

PyTorch Float64因子图含实际camera逆、非严格正交R的真实逆、过滤尺度、raw quaternion归一化、
3D/screen opacity补偿及cap，CUDA seeds再反传到11原参数。完整原891射线/51524节点、9旧异常和3零loss保留。
初版history_gap_gpu_v1全部891过，max整射线相对L2=9.967356673415652e-11。
随后seed13018新合成32仅28过原block门槛：4个on-axis且kernel0案例理论为零的几何块出现微小残差，
不能因为整射线误差约1e-14就删除块门槛。原失败及数组/源码保留。

V2不换数学目标或门槛：用通用叉积恒等式计算横向能量
E=||S R^T(v cross d)||²/[(detS detR)² ||Fd||²]，F=S^-1 R^-1，
避免先投影再相减；screen kernel0直接采用gamma及其导数恒0。恒等式不依赖R正交，也不是新颖性主张。
V2全891过，max整射线9.143033722839749e-11、各块max1.7200367458184842e-10；
877中间二分、11尾部公式、3零loss。原32作为开发回归全部过，另固定seed13019的新32也全过，
共64/240节点，非零screen kernel0/.3/1均覆盖；64 max整射线4.582832397500839e-14。
这64是合成fixture，不是64个新真实场景。

两个原真实无根更新经GPU尾部延拓后值误差≤3.109e-15、raw方向导数相对误差≤1.416e-14。
另10个系数测试覆盖单/双层奇异点、质量边界、无根、前根、严格cap与rho5/10/13.333/20薄间隙，
loss/seed全过MP；无效rho0明确负状态/NaN拒绝，未默默奖励为零。薄间隙梯度指数消失局限仍成立。

实际RTX5090Laptop、Torch2.12.0.dev20260314+cu128、CUDA12.8。
V2 warm三次几何+loss+raw反传为79.163742/79.067230/79.066498ms，中位79.067230；
第一次cold1045.542725ms保留，compile37.3053s、准备/transfer .83956s单列。
process peak allocated153932288B，非训练增量内存。
初版阶段拆分warm median为fields5.53574ms、loss/seeds72.69913ms、raw-backward10.97907ms；
表明当前串行per-ray核为主要成本。不能把V1阶段拆分说成V2阶段时序，或声称整体训练加速。

主V1 session34235 exit0，V2 session14507 exit0，最终64确认session66333 exit0，profile exit0。
confirm_v1因NumPy整数JSON失败；confirm_v2计算/保存完成后最终console JSON再次失败；
confirm_v3从v2已保存数组/summary恢复并审核28/32，不重跑计算；confirm_v4是正确图的64确认。
原失败manifest/traceback/源码包和数值失败均未覆盖。
独立verify_history_gap_gpu全来源ZIP/实际二进制/所有梯度loss/MP引用映射审核通过，
见results/history_gap_gpu_audit_v1.json。V2实际binary SHA
406b06756bb62361cb4a57777bbba3b2815c5ef85661bee933c6dfe2c15008e4。

下一步优先warp并行化当前约72.7ms的loss/seed核，以同一输入和独立MP全门槛验证，保留串行正确性参照。
之后把GPU loss接到固定采样的训练adapter，先零权重同状态对照、可重复小步和完整成本，
再进行配对训练及GT/留出图像/容量门槛。当前只有正确性证据，没有重建、图像、整体效率或新颖性成功结论。
