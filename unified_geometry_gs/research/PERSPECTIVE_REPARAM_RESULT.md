# 条件性透视重参数：证明、原型与真实原参数复核完成

2026-10-13，目标active。不是完整新渲染器、加速或质量改善声明。
预案 `PERSPECTIVE_REPARAM_PROTOCOL.md`，完整证明 `PERSPECTIVE_REPARAM_THEORY.md`。

在center未clip、screen kernel=0、P与Sigma精确互逆的条件下，
现有GGGS affine plane线性峰位项、中心depth、width、conic能量与相机几何
可以代数恢复同一世界Gaussian的精确ray peak/width/peak amplitude。
原型 `perspective_reparam.py` 已实现这一表达；未增加学习参数，
但尚未核验GPU额外算术/带宽/反向成本，不能声称更快或零额外存储。
HTGS与3DGUT已有透视正确Gaussian评估，因此一般精确射线概念不是原创。
此packet表达是否有独立方法价值仍待实现比较和更全面新颖性审查。

核验完成：

- 9个一般符号身份（Schur、Taylor、峰位、宽度²、能量等）通过。
- seed13015的64个预定SPD构造通过原1e−8/floor1e−12门槛；
  最大scaled errors(p,width,alpha)=(9.1404e−14,4.3393e−14,8.3847e−13)。
- 原24射线共1992个selected基元—射线输入，全部保存；
  **1946适用输入全部通过，46个center clipping域外输入单列保留**，不算通过。
  kernel/坏输入域排除0。主double对原world restriction最大误差
  (4.2671e−15,9.6150e−16,2.4206e−12)。
- `results/perspective_reparam_report_v1.json`：124实际源/archive、50产物、3PLY、
  全部1992输入/域分类核验；原PLY参数、规范化四元数、filter后的尺度/opacity，
  独立80位因子空间公式重算**全部1946个适用输入**，最大误差
  (4.9905e−15,1.6558e−15,2.6328e−12)，原1e−8门槛未改。
  Oracle使用保存的二进制世界方向，不擅自重新规范化相机标定。

保存全部point/packet JSON、NPZ、sourcezip、模型SHA及每个独立高精度结果。
主实验和独立重算均已exit0。高精度不是严格区间或证明助手认证。
原affine参数在同适用输入上的最大差较大，但只是对世界参数的差，
不能将alpha小分母下的1222倍scaled差误作像素质量提升倍率。

尚未完成的关键项：

- 46个clip域外输入需要一般化推导/实现，不能用忽略它们来宣称全域方法。
- FP32/极端条件数、CUDA forward和全部参数VJP尚未核验。
- packet表达可能继承现有conic与P/Sigma浮点不一致；当前double通过不解决此风险。
- 几何形状等价不会修正tile/alpha/prefix选择、排序、可见性或单法向限制。
- 真实图像/几何、运行效率、重复场景和新颖性未证实；不启动大规模训练。

后续优先考虑clip-aware packet一般化，再完成FP32/反向与GPU实际开销；
若实现门槛通过，再做固定模型几何/图像对照，最后决定是否训练。
原始方法文献：[HTGS](https://arxiv.org/html/2410.08129v2)、
[3DGUT](https://arxiv.org/html/2412.12507v2)。
