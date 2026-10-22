# 完整图像否定首版精确packet的数值与效率采纳

2026-10-14。`exact_perspective_smoke_v1` 与 `frozen_exact_render_v1` 均complete、exit0；
单视图smoke原门槛通过，但全24视角的预定数值和效率门槛失败，不能由smoke替代全图结论。
没有启动训练或GT改善评估；按原协议停在数值/效率筛查。

## 完成范围

三场景scan37/55/63各8个既定开发视角，合计10,834,488个图像像素，全部保存。
新独立后端三臂affine、exact_packet、exact_factor，仅在median T评估中改变(p,rho,alpha)。
RGB、normal、alpha合成、eligibility/cap、prefix、排序、根初始化及五次subdivision均保持原规则。
packet与factor均使用FP64标量系数求值、FP32输入缓存与原FP32 solver；factor多48B/基元及预处理。
backend显式拒绝grad、sample、integrate/SDF和backward，原生产文件及安装二进制不变。
这不是完整exact renderer、训练方法或新的法向场。

## 全图数值结果

- 全部保存数组有限；RGB/alpha/normal在全部24视角逐位保持。
- packet/factor正深度mask在全部24视角完全一致。
- 共同有效像素差值p99各视角均<=1.90735e-6模型长度单位。
- 但8/24视角的最大depth差超过预定1e-3；合计9个像素超阈值，最大0.1796405315。
  此项是预定门槛的一部分，不能因像素稀少或p99小改判整体通过。
- 最大例scan55/cam0000/pixel(161,215)：affine=2.687563658、packet=2.502209425、
  factor=2.681849957。保存割线斜率分别2.1102941、.004854369、.004901961。
  割线值不是精确根导数；尚未独立重建此像素完整Gaussian系数或归因于某一机制。
- B/C slope通道也存在差异，首次smoke已见最大9.54464，完整图像有更大值。
  斜率描述指标是在smoke后添加，原深度/时间门槛未变；无梯度等价声明。

## 完整forward成本

每视角每臂5次预热，seed13018随机化顺序，3轮各10次render，GPU串行。
packet/factor的GPU时间比几何均值2.6001822，最大2.6573840；同步wall比几何均值2.5999135。
预定效率门槛为几何均值<=.90且任一视角<=1.05，因此失败。
包含完整render的预处理、排序、RGB与深度，但没有训练backward或数据加载/编译；不作训练速度结论。
当前首版重复在求根轮次内求值、FP64成本与缓存安排可能影响性能，结果不排除未来其它实现。
不能用未实现的优化设想改变该版本失败判定，也不把参考factor的成本当已知最优下界。

## 独立复核和研究决定

`results/frozen_exact_render_report_v1.json` 核验17源/archive、50产物，重放全部图像、门槛与计时算术；
`frozen_exact_assets_v1.json` 在WSL核验3二进制、51生成文件、652原源文件及3原PLY哈希。
这不是独立rerender或逐像素高精度根参照；原射线1992局部通过没有覆盖这些新异常像素。

完整反向审查 `EXACT_PERSPECTIVE_BACKWARD_AUDIT.md` 已给出局部因子VJP，96构造/1536分量与
六个非临界数学根通过独立检查，但不等于CUDA全renderer backward。Triangle原论文和固定源码审计
已补齐 `TRIANGLE_THIN_LAYER_AUDIT.md`；其sigma是屏幕边缘锐度，不能等同沿射线薄化厚度。

当前决定：不扩大这版packet到GT质量或训练。下一项是有界的失败机制重建：保存并检查全部9个超阈值像素，
读取实际生产候选前缀与packet/factor系数，对比稳定高精度根及FP32求根响应；
先区别表示误差、真实小斜率、乘积舍入和有限bracket行为，再决定薄层理论是否解释可观察效应。
此为事后失败诊断，不能用选出的最坏例估计总体频率或证明训练因果关系。
