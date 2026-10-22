# 方向衰减的单法向兼容性：数学与真实参数诊断方案

2026-10-13，写于本轮数值检查前。此前薄层/区间和质量负结果保持。
不训练，不改生产CUDA或原CPU参考；不重新声称Gaussian Wrapping已有的多视图问题。

## 限定模型与待证结论

对真正的世界Gaussian G_i(x)=a_i exp(−(x−mu_i)^T P_i(x−mu_i)/2)，
v_i=sqrt(1−G_i)，b_i=grad log v_i，考虑逐基元衰减相加
K_x(w)=sum_i |w dot b_i|。不使用per-view affine footprint代替该世界函数。

仅考察单法向/绝对梯度形式 K_x(w)=|w dot grad log v(x)|，不是
Objects as Volumes允许的一切法向分布或随机实体。候选结论：

1. 在一个固定点，存在任意向量b使K(w)=|w dot b|对所有单位w成立，
   当且仅当所有非零b_i共线。此点态条件不保证邻域中的可积性。
2. 若固定v=product v_i，则其梯度b=sum b_i；上述等价成立当且仅当
   非零b_i同向共线。互易K(−w)=K(w)本身不能保证这个单法向形式。
3. S=sum b_i b_i^T，lambda_2为第二大特征值，
   inf_b sup_||w||=1 |K(w)−|w dot b|| >= sqrt(lambda_2(S))。
   拟证明由b的正交平面与S前两个特征向量张成空间的交得到。
   b1=e1、b2=e2的例子拟达到该下界；需给出合法Gaussian实现。
4. 对任意两项，det=||b1 cross b2||²、trace=||b1||²+||b2||²，
   sqrt(det/trace)是更保守的正下界，且加入其他Gaussian不削弱该绝对证书。
   这是标准矩阵/凸几何推论在本模型上的补充，不先行声称新颖性。

## 核验方案

- SymPy精确验证grad log sqrt(1−G)公式、正交/反向向量例及合法Gaussian实现。
- 固定NumPy seed13014，64组各6个向量，随机候选b；在b正交平面选最大
  Rayleigh方向，检查残差>=sqrt(lambda_2)（1e−10相对/绝对数值余量）。
  不以有限方向采样证明sup，只用它佐证已写出的解析证书。
- 固定真实查询点24个：原median_cpu_scan37/55/63中每个保存视角的
  第一条有限中位根射线，将保存深度按保存相机/像素回投影。该点不是GT表面点，
  也不宣称是真正世界Gaussian模型的精确中位根。
- 原3个PLY必须匹配旧SHA；对每点评估模型中**全部**Gaussian，不做半径、alpha或
  ray-contributor裁剪。完整双精度矩阵S、sum b_i、sum||b_i||和数值lambda2记录；
  Gaussian指数underflow/无穷/无效参数数量保留，不能默默删基元。
- 世界模型采用保存raw rotation的规范化正交旋转、filter后的scale和opacity；
  双精度为名义测量，精确模型用同一raw浮点参数作为实数输入定义。
- 各点选名义||b_i||最大项，再选对它垂直残差最大的另一项，保存完整raw参数和ID。
  80位有向区间算术从raw参数独立重算这两项；记录cross²的正下界和trace的正上界，
  从而给出条件于保存模型和查询点的保守正证书sqrt(det_lower/trace_upper)。
  若区间不能认证正值，该点记录为未认证，不更换查询点或隐藏结果。
- 记录product-v目标在7个方向的衰减差：三个坐标轴、(1,1,1)、(1,−1,0)、
  原查询相机方向，以及sum b的正交平面内最大Rayleigh方向。
  后一个方向是解析构造的证据方向；所有差都不是像素或重建误差。

逐点保存进度与结果。全部24点保留；没有必须达到某个效应大小的质量门槛。
数学核验失败保存源和日志；真实数值现象不能证明完整渲染器存在同样的几何缺陷，
也不能直接推荐新正则项。先独立核验证书，再决定是否值得方法开发。

## 原始文献与范围

GGGS https://arxiv.org/html/2601.17835v2 已给出连续透射率及近侧等价条件。
Gaussian Wrapping https://arxiv.org/html/2604.07337v1 已讨论方向衰减、互易与共享场，
不能把泛泛的“不一致”重新命名为原创。
Objects as Volumes https://arxiv.org/html/2312.15406v2 包含非Dirac法向分布；
本方案的单法向不可能性不否定其一般框架。
定向检索未见匹配不等于新颖性已成立。
