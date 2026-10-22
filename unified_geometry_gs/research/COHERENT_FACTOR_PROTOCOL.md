# 一致精度因子与紧凑QR：固定开发检验

2026-10-14。候选来自已完成的有效alpha/能量分析，属于事后开发，不是新颖性确认或独立泛化。
主候选在观测前固定为QR六个F32三角项+三个F64 camera mean（额外48B/基元）。
全F32九项36B、全F64九项72B仅作预定精度对照，不因结果好而改主候选或门槛。
不改原生产CUDA或已有backend，不编译GPU、不训练。

## 精确对象与表示

保存activated尺度s、quaternion q的多项式R和相机仿射(A,v)作为精确二进制输入。
用真正R^{-1}和A^{-1}构造F=diag(1/s) R^{-1} A^{-1}，camera mean m=A*world_mean+v。
对于quaternion多项式，k=||q||²−1、h=||q_vec||²、D=1+4kh，
R^{-1}=(R^T+4k*q_vec*q_vec^T)/D，D!=0；这是待符号核验的标准逆表达，不宣称新代数。
非quaternion的既有构造R使用一般逆。QR得F=Q*U，三角U保正对角；G=F^T F=U^T U。
同一U给a=||U*d||²、b=(U*d)·(U*m)、p=b/a、rho=sqrt(a)。
能量用||cross(U*m,U*d)||²/a，非混合conic/plane。QR、缓存舍入及后续double求值仍有误差。
query norm由保存qray重新求取，t严格为单位camera-ray参数。amplitude固定生产共同co.w（如有）。
输出系数先用double保存，再另存F32(p,rho,alpha cap .99f)，根比较明确区分两种对象。

## 输入与门槛

完整覆盖以下固定输入，无按结果删除：
- 原生产24点/1992项，使用保存的activated参数、相机与query；含全部clip标记，不以clip排除。
- 已定位9异常射线/772 accepted项，目标是saved-activated geometry+共同co.w的MP参照。
- 原44极端基元，每项固定3个查询：中心、mean方向、mean方向x偏移+.1；这是132项压力诊断。
- 既有seed13017的4096几何构造；已被观测过，仍为开发集，不能冒充独立确认。
- 双层临界fixture：epsilon取F32 .002/.005/.01，第一层z=.5、第二层z=1.5，amp=.75/.5。
  第一层y取F32 epsilon*sqrt(2log1.5)及相邻上下两个F32，第二层y=0；尺度各向同性epsilon、query=(0,0,1)、相机I。
  这9组/18项均来自合法正定F32世界输入。使用未round参考alpha的MP根，保留所有状态。

独立MP80显式covariance逆用于全项peak/sigma/alpha参考；peak相对floor1e−8门槛1e−5、sigma相对floor1e−8门槛1e−4、alpha绝对1e−4。
所有9真实射线及9临界fixture，主候选F32系数数学根须存在，并与未round世界MP根相差≤1e−3 camera-ray单位；无解或无法辨号亦不通过。
并列诊断double系数的数学根，不以它替代F32主门槛。精确二进制质量用于F32模型，高精度根不是有向区间证书。
全F64通过也只是数值参照，不建立效率、完整CUDA/backward、GT质量或采用结论。

如主候选不通过，不进入整图计时/训练、不搜索另一缓存组合。保留负结果，优先分析量化裕量/条件精度或薄层理论。
只有通过才锁定主候选，用新seed13019产生同分布4096及额外临界输入确认；本轮开发输入不能当确认集。
源码、原输入、所有数组与失败目录均用唯一版本和哈希保存。

执行勘误：v1在最后一组对齐的临界查询上，MP80参照的c−b²/a相减得到微小负能量而触发断言，未形成完整结果。
此前四组及源码archive保留；不clamp负数或删样本，改用等价的perpendicular^T P perpendicular参照计算，v2新目录重跑。
候选、输入、门槛不变。MP参照是数值交叉检查，不因80位运算就宣称无舍入误差。
