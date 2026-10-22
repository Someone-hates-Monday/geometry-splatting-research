# 全9异常点：系数组合与原始世界参照

2026-10-14。在读过保存系数差和根差后制定，明确事后机制分析，不是独立泛化确认。
本轮不编译CUDA、不训练、不改旧门槛。输入固定为capture_v3全部9点/772 accepted项，保留实际ID/前缀/顺序。

## 固定系数8组合

以factor为mask0，packet为mask7；bit0替换alpha，bit1替换peak，bit2替换rho。
每点全部8个混合数学模型，无进一步样本选择。先精确二进制有理数质量判定，再稳定double与120位根，180位端点符号复查。
报告有限/不足/临界/无法辨号全部状态。端点mask0/7须与旧独立oracle根相符，否则停止解释。
六种替换顺序的Shapley平均只作有限代数分配，记录交互、方向和抵消；不是统计或因果贡献率。
报告alpha单独替换、peak/rho单独与联合替换及全部根，不把比例相加规则冒充线性物理。

## 世界参照：先明确数值对象

从原PLY逐ID核验raw参数字节、3D filter、摄像机和capture哈希，避免以方便数组取代原模型。
至少输出两个明确区分的高精度对象：

1. saved-activated：用保存的F32 activated尺度、opacity、quaternion多项式构造世界covariance，再在高精度中求逆。
   quaternion保持保存字节，不默认精确正交。相机用保存F32 view仿射映射的精确逆；query_xy/ray norm约定明确固定。
2. raw-physical：从原PLY的log尺度、logit opacity、raw quaternion和filter重新高精度激活/规范化，构造物理covariance。
   相机/射线约定与前者相同，避免另引入相机标定变化。区别于完全double/F32生产计算。

若实现需要另列直接精度因子参照，可单独报告，不能把R^T当非正交R的逆。
给出一般沿线quadratic系数a=d^T P d、b=d^T P(mu-o)、c=(mu-o)^T P(mu-o)，
peak=b/a、rho=sqrt(a)、alpha=filtered_opacity*exp[-(c-b²/a)/2]，并检查a>0及Schur能量>=0。
t是单位camera-ray参数；F32 view的3×3仅近似正交，故不把它严格命名为单位世界弧长。
保存完整精度数字和系数/根误差，包括哪个参考下哪条更近。不得把raw-world参考等同GT表面。

root oracle目前强制F32输入；高精度世界参照不得round成F32后再宣称是原始真根。
可以使用独立高精度残差/root实现，保留非有限/不足/无法辨号并给精度复查。
预期区分：alpha表示差、peak/rho表示差、非线性交互、F32激活/变换差、有限求根差；观察前不判哪类主导。

## 记录和后续决定

所有新路径使用唯一目录/JSON，保存源码、输入及产物哈希，保留失败。结果只覆盖这些被选择的异常。
如指向可修复的预处理能量/opacity差，先提出一个有数学依据的稳定构造，并在原输入及新的锁定构造上核验。
已有packet完整forward慢2.60倍的失败不变。没有完整后向、成本及GT质量证据，不扩大该版本训练。
