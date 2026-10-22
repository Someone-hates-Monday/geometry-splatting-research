# 方向衰减与单法向兼容性：完成数学补充和真实参数核验

2026-10-13。`directional_field_world_v1` complete；重放脚本exit0。
研究目标仍active。没有新图像/几何质量、训练收益、加速或原创性结论。

## 数学结论

完整证明见 `DIRECTIONAL_FIELD_THEORY.md`，预案见 `DIRECTIONAL_FIELD_PROTOCOL.md`。
固定世界Gaussian点处，令 b_i=grad log sqrt(1−G_i)，K(w)=sum|w dot b_i|。
存在任意单个向量b表示所有方向的K，当且仅当非零b_i共线；
若要求b=grad log product sqrt(1−G_i)=sum b_i，则还必须同向。
互易性本身不够。对任意b的最优一致方向误差有下界

    inf_b sup_unit_w |K(w)−|w dot b|| >= sqrt(lambda2(sum b_i b_i^T)).

合法Gaussian构造达到下界；双基元保守下界sqrt(cross²/trace)
在加入其余全部Gaussian后仍成立。这是点态单法向结论，不排除一般法向混合或随机实体。
SymPy精确梯度/构造/身份式及64个预定数值方向检查通过；有限采样不是解析证明。

## 24个真实参数查询点

原scan37/55/63各8个保存视角、每视角第一条有限旧中位根回投影点。
不是GT表面点，不宣称是真实世界Gaussian模型的精确中位根；不能解释为表面发生率。
三个原模型分别有1,010,226 / 1,052,385 / 510,660个基元，PLY SHA全部匹配旧记录。
主名义double计算保留全部基元，没有半径或ray-contributor删减。
各点都存在严格正的双基元区间下界，24/24；绝对值单位是世界坐标单位的倒数。
scan37最小证书只有3.4193e−10，严格非零不意味着有实际可见效应。

完整double谱下界除以sum||b_i||的名义比值：全部点min=.00900015、
median=.06790113、max=.21308058。各scan中位数 .05351782 / .08987450 / .03856023。
这些比值不是严格区间相对下界，不是像素/几何错误率或方法预计收益。
指定product-v在原相机方向的名义相对衰减差中位数为
.10048740 / .61897299 / .46915751。七方向明细均保存，正交证据方向是解析构造，不能当随机方向。
指数underflow为0的名义基元数按点完整记录；未把它们从精确模型中删除。
双基元证书由原始参数重新算出，不受全模型名义underflow的逻辑削弱。

## 独立核验

`report_directional_field.py`、`DIRECTIONAL_FIELD_REPLAY_PROTOCOL.md`、
`results/directional_field_world_report_v1.json`：

- 13实际源文件与zip字节、26实际产物、3完整PLY SHA全部匹配。
- 全部24点从保存相机/射线精确重构；48基元的ID/raw参数逐一匹配PLY原行。
- 80位exact endpoints逐一重放一致；Fraction精确检验24个L²r_upper<=d_lower与double<L。
- 独立120位标量log-vacancy函数经mp.diff求导：144分量均落入80位和100位区间；
  100位区间全部包含于80位区间。没有调用主代码显式梯度作为独立参照。
- 所有保存汇总、S特征值及方向product投影/差值重算通过。

mpmath 1.3.0官方文档明确iv仍为实验性功能，部分函数未完备支持。
这里使用exp/sqrt和基本运算，保存有理数端点并交叉核验，但不声称证明助手认证。
主实验与重放均已结束；`directional_field_world_health_v1.json`匹配进程为空。

## 文献与下一步边界

GGGS已说明近侧等价条件；Gaussian Wrapping已讨论非定向场的多视图限制并给出定向方案。
Objects as Volumes允许一般法向分布，不能将本单法向条件扩大成“所有全局几何不可能”。
K正是zonotope sum[-b_i,b_i]的支持函数；单向量对应线段。
Siegel 2023 Eq1.7/1.8与Torres 2024已有支持函数/zonotope逼近理论。
因此基础表示与一般逼近问题不是新发明；当前候选补充仅是此Gaussian模型的
显式适用条件、谱下界、合法尖锐构造和原参数证书。尚未完成足够全面的新颖性审查。

不直接加入“所有基元单法向”正则或启动训练：边缘、薄结构和多层可能本就需要多个方向。
本轮随后完成同24条世界相机射线与保存affine中位根的检查，
见 `WORLD_AFFINE_RAY_RESULT.md`；相对旧深度绝对位移中位.01262537%，尾部2.69459%。
这初步区分了局部方向量和median位移，但仍同时混合affine形状、裁剪/选择/clamp规则。
目前没有GT/跨视图误差因果归因；下一步匹配规则做因素分离，再考虑方法开发。

原始文献：
[GGGS](https://arxiv.org/html/2601.17835v2)、
[Gaussian Wrapping](https://arxiv.org/html/2604.07337v1)、
[Objects as Volumes](https://arxiv.org/html/2312.15406v2)、
[Siegel](https://arxiv.org/html/2307.15285v2)、
[Torres](https://arxiv.org/html/2407.13125v1)、
[mpmath 1.3.0官方文档](https://raw.githubusercontent.com/mpmath/mpmath/1.3.0/docs/contexts.rst)。
