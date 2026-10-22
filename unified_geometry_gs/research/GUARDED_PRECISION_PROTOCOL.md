# 条件精度与区间根：固定CPU开发/确认

2026-10-14。接续一致QR主候选失败，不能改旧判定。本轮固定单一策略，不再筛选缓存组合。
目标是未舍入的saved-activated世界covariance+共同amplitude射线中位根；不以量化模型的根替换目标。

## 可计算预算及数学界

CPU用53位significand的有向区间算术，从原参数构造F=diag(1/s)R^{-1}A^{-1}的区间、精确camera mean的区间。
名义Float64 QR给Qhat及正对角U；Qhat仅用于预算构造，缓存为6F32 U+3F64 mean，加两个向上舍入F32预算eta_U/eta_m（理论56B/基元）。
delta_Q>=||Qhat^T Qhat-I||_2，用区间Frobenius范数界给出；若delta_Q<1，存在正交polar Q满足
||Qhat-Q||_2 <= delta_Q/(1+sqrt(1-delta_Q))。
eta_U取||F-Qhat Ucache||_F的区间上界加polar误差*||Ucache||_F，覆盖||Q^T F-Ucache||_2。
这不是以MP真根回填预算；计算只读原参数与名义预处理结果。未来GPU预算实现及成本尚待验证。

固定d=q/||q||，名义u=Ucache*d、p=(u·v)/||u||²、r=v-pu、x=mcached-pd，e=||r||²。
令rho=||u||、zeta=eta_U*||x||+(||Ucache||_F+eta_U)*eta_m，a_min=(rho-eta_U)²>0。
则 |delta p| <= [eta_U*sqrt(e)+(rho+eta_U)*zeta]/a_min，
||delta r||<=zeta+|delta p|*(rho+eta_U)，|delta e|<=2sqrt(e)||delta r||+||delta r||²，
rho真值在rho±eta_U。名义量也用有向区间计算，包含求值舍入。amplitude和cap固定，e区间映射为alpha区间。
该实数扰动界利用u·r=0和空间残差x，避免用||mean||粗界导致严重放大。

## 固定求根策略

每层psi_alpha(x)随alpha及x单调递减，x=(t-p)rho，故系数区间给T(t)乘积区间。
采用稳定H残差及log1p级数处理小Gaussian tails；53位CPU区间具有无界指数，**不是IEEE FP64 GPU实测**。
终端mass先判定finite/insufficient/critical/unresolved，不强制有限根。
只有找到两侧有认证符号且宽度<=2e−3的括区时，才返回其midpoint与1e−3误差界；否则触发高精度路径。
高精度路径读取原参数，使用全F64 U/mean与重新计算的预算、未转F32的alpha，再作同一符号/括区检验。
仍无法认证者保留unresolved，不以MP真根直接补结果。当前只验证depth，不宣称稳定backward或训练收益。

## 输入和门槛

开发覆盖原7010个coherent_factor_v2输入的全部参数区间；原9真实射线+9临界双层root全保留。
新确认生成器在运行前锁定：seed13019，64个四层普通/16个近阈值双层/4个exact-critical双层/4个微小偏移双层/4个无有限根模型。
普通4层camera I，positive F32 scales、unit近z quaternion和F32 mean，amp .75~.95，query=(0,0,1)。
近阈值2层camera/rotation I，第1有效alpha约.5：amp .6~.9、横向scale .05~.2、offset为F32 sx*sqrt(2log(amp/.5))，第二amp .5；z gap .25~1，normal scale .002~.015。
exact-critical两层均alpha .5且query穿过均值，宽度可不同；无根包含单层alpha .5及三个alpha .1。
微小偏移双层第一amplitude=.5、第二=.5，第一y为F32 1e−20/1e−15/1e−10/1e−5，scale=.005，camera/rotation I、z=.5/1.5。
回退从原参数的精确二进制均值/射线偏移再用Float64 U求energy，避免v−p*u的大数相减；保留amplitude/energy，不把有效alpha作为唯一高精度信息。
所有生成字段与原参数存盘，fixed generator不因结果删样本或改种子。

独立MP80/140世界covariance逆用于核验系数包围、分类和返回括区；其结果不参与策略选择或回填。
门槛：全参数区间覆盖参照；所有存在有限根的18旧/88新模型均返回包含参照根的认证括区且midpoint误差<=1e−3；4无根分类正确。
不把返回unresolved当通过；记录低精度认证数、fallback数及失败，不将样本/CPU墙钟冒充GPU效率。
只有CPU开发和锁定确认全过，才评估CUDA实现可行性与完整成本；旧所有失败和接口下界保留。
