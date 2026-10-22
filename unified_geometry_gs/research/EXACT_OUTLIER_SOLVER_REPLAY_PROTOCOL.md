# 固定捕获系数的CUDA求根路径替换

2026-10-14，明确事后诊断。只使用全9异常射线的捕获FP32系数和复制初值，不改旧后端。
原捕获/全图门槛与失败保持。先落盘再运行这些路径替换，不用它们重新选择像素。

新独立小CUDA kernel保存5轮完整9端点T、bracket和最终插值，三条路径：

0. FP32 per-item factor、FP32乘积、FP32 bracket/插值，复制原迭代语句；
1. 保持FP32 per-item factor，乘积及由T驱动的插值比改为FP64，bracket和输出仍FP32；
2. 同保存系数，以FP64 log1p光学厚度求T和插值比，bracket和输出仍FP32。

三条路径的prefix、系数、初值/初始range、SPLIT8、5轮、>=规则和endpoint复用都固定。
路径0须逐位比较原输出z-depth及slope，记录全部匹配/不匹配，不以近似匹配冒充原路径复现。
独立编译/内联、fast-math/FMA及捕获probe与原kernel上下文有差别，未匹配时限制因果结论。
路径1不能消除FP32 exp/rsqrt/factor舍入；路径2仍有FP64及固定bracket误差，不是高精度oracle。
因此分别对照独立oracle，而不把“某替换更好”直接叫完整修复或新方法。

保存全部每点/臂/路径结果和trace，包括无效根、零输出、非有限内部量。
高精度总质量与root存在由独立oracle认定，任何路径给有限输出不自行证明真根存在。
不运行GT/训练或新的性能基准；新代码只能用于这项受控数值机制诊断。

执行勘误：v1 complete且18/18原z-depth/slope逐位匹配，但复制初始gate曾写.5，
真实config为MIN_TRANSMITTANCE=.45；9个捕获affineT最大.39382255，两gate在这些输入上等价。
这不证明v1对其它输入忠实。保留v1源码归档/结果，改为直接include原config的v2独立重放，
其它三路径及门槛不变。生产SAMPLE_RANGE=.4，初始化和bracket仍读取probe原F32字节。
