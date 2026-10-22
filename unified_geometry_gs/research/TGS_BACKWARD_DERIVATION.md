# TGS 内切半径归一化窗口的光滑分支导数

范围：固定相机、固定可见性/排序/像素贡献集合、唯一最大边、无 alpha 饱和，
仅 RGB 损失的几何、activated sigma 和 activated opacity。这里不对离散切换求导。
本推导是链式法则补全，不能仅据此宣称方法新颖或训练收益。

## 1. 窗口

屏幕三角形顶点为 q0,q1,q2，单位外法线 ni、偏置 bi，内半径 r>0。
固定内部像素 x，令 phi=max_i(ni dot x+bi)<0，u=-phi/r。
未饱和 alpha=o u^sigma，其中 o>0、sigma>0。相应偏导为

```
d alpha / d phi   = sigma alpha / phi
d alpha / d r     = -sigma alpha / r
d alpha / d sigma = alpha log(u)
d alpha / d o     = u^sigma
```

前三式分别来自幂函数和指数函数的普通求导。在唯一 active edge 保持不变的邻域，
phi 是该边有向距离的光滑函数。令 gx=dL/d alpha_x，屏幕几何梯度为

```
dL/dq = sum_x gx sigma alpha_x / phi_x * d phi_x/dq
        - (sum_x gx sigma alpha_x) * d log(r)/dq.
```

第二项在 pinned backward 中没有传播。已有 dL_dsigma_factor 是零初始化的 P×1 临时
buffer，原实现仅传递指针、没有读写；隔离实现用它积累 dL/d log(r)。这不增加新的逐像素
缓存，只增加每个参与反传像素一次 atomicAdd，及每个可见三角形一次半径导数计算。
尚无耗时基准，不能据操作量宣称速度或显存收益。

## 2. 不必对内心坐标逐项求导

记带符号二倍面积 D=cross(q1-q0,q2-q0)，周长 P=sum_i |q(i+1)-qi|。
非退化三角形的 r=|D|/P，这是面积=半周长×内半径的等式。
因此在 D 不变号的邻域

```
grad_qi log(r) = grad_qi D / D - grad_qi P / P
grad_qi D = (qnext.y-qprev.y, qprev.x-qnext.x)
grad_qi P = (qi-qnext)/|qi-qnext| + (qi-qprev)/|qi-qprev|
```

next=(i+1)%3、prev=(i+2)%3。第一式对顺、逆时针均成立，因为 d log|D|=dD/D。
它与 forward 通过最后一条单位外法线和内心算出的 r 完全等价（实数算术）。
有限精度中两种半径表达式的舍入差异仍存在，验证中保留这一限制。

## 3. 从像素到世界坐标

若 hom=P_camera [X,1]、v=hom.w+epsilon，
q_x=W/2*(hom.x/v+1)-1/2，q_y=H/2*(hom.y/v+1)-1/2，
则

```
d q_x/d X_j = W/2 * (P_xj/v - hom.x * P_wj/v^2)
d q_y/d X_j = H/2 * (P_yj/v - hom.y * P_wj/v^2)
```

epsilon 为 native forward 的 1e-7。已有投影反传有括号中的导数，但没有 W/2、H/2。
这两个因子应乘到几何屏幕坐标梯度，不能乘到另外累加的 depth、normal、SH 世界梯度。
非正方形图像中两个因子不同，单一学习率缩放无法恢复这个 Jacobian。

## 4. 一个可直接检验的不变性

固定单个内部像素 x，对所有屏幕顶点作同心正尺度变换
qi(t)=x+exp(t)(qi(0)-x)。在分支不切换的邻域，单位边法线不变，
phi(t)=exp(t)phi(0)、r(t)=exp(t)r(0)，所以 alpha(t)=alpha(0)。
精确方向导数必须为零。若反传只冻结 r，则该方向导数变成 sigma alpha，通常非零。
半径链式项恰好抵消此项。这说明缺项改变导数的零空间；单一非零学习率因子不能普遍修复。
这是窗口公式的数学性质，不是整幅图像或多视角训练损失沿同一方向都不变的声明。

## 5. 如何证伪

`probe_tgs_backward.py` 保留以下独立对照：

1. CPU float64 显式前向，PyTorch 对完整内心/边长公式自动微分；
2. CPU 和 CUDA 各自的中心差分，四步长、按实际 float32 参数间距作分母；
3. 冻结半径及移除像素 Jacobian 的解释性错误公式，检查能否重现原 native VJP；
4. 只补像素 Jacobian 与完整半径链式法则的两个独立编译模块。

保存正反绕序、非正方形图像、不等深度及旋转平移相机；所有正负扰动均检查 active edge、
alpha 区间和 native 预处理可见性。测试没有覆盖 alpha cap、max ties、早停、排序变化、
辅助几何输出或 SH 对几何的混合梯度。它也不能证明所有训练损失实现正确。
