# 共享原参数几何缓存：相同预算与真实基元构造成本

2026-10-14，输出观察前固定。CSR完整名单已核验，但原GPU预算每个primitive-ray关联都重复inverse/QR/polar/residual。将不依query/amplitude的构造与预算按primitive/camera缓存，保持原物理模型和全部算术公式。

## 固定实现

使用reciprocal_root_v2的冻结gi/gp核心。缓存诊断格式28个double/primitive（224B）：两路径status、delta_Q、两个向上F32 eta、向上F32 nu、3个F64名义mean、3个真实mean区间、6个low U（F32值精确提升）、6个high U、padding。保留真实mean区间用于原严格aligned分支。这里不声称此前56B理论格式的实现或最优布局。

把原decode在query前的预算构造移入build；缓存解码恢复相同U/mean/eta/nu/delta后执行原query之后的代码。输出仍为26double/mode以核验完整位，不改exp级数、AMP、候选、mass、容差、ray定义、字段或错误码。original control重新从同一冻结源编译；原生产CUDA不改。

## 固定数学批次

原7322 source、110模型不变，以ids=0..7321、query=source[32:36]逐个构造/解码；要求全部52double预算输出逐位一致（NaN bits也比较），control与原archive逐位一致。再由同一冻结root core求106finite/4no-root，根/证书全数组逐位一致，继承此前MP240参照，不称新样本确认。

在相同数据上重复8次query，共58576项，比较原repeat preprocess与build once+cache decode的5对交替event/wall，所有repeat输出位一致。报告build与decode单独成本；此局部效率门槛仅预定合并比≤.95，另要求全部位一致。失败保留，不改输入或阈值。

## 真实完整24视角构造

原scan37/55/63各8视角、同原PLY/activated参数与相机输入，构造每个基元的缓存，不按先前接受率抽样或删除。所有cache状态报告，非有限/失败不能伪装通过。record cache row bytes、每场景P、额外GPU tensor内存、build event/wall（5次）、分位预算/状态、source实际tensor/view身份。Source rows的q/AMP只填占位并在构造中完全忽略，不能将这种缓存构造称为全部CSR射线解码或全图认证。

缓存输入shape/内容会与CSR原输入身份核验；build不向设备提供CPU因子/预算或MP参照。全部缓存可由原输入重建；保存每camera的小摘要与cache二进制内容hash，不永久复制约5GB全部相机缓存（已保存source/model/view引用）。标明这只是实际build和内存构造检验。

## 下一项与边界

将缓存接到真实CSR按需解码，避免全量26double/mode输出常驻，然后实现动态/射线内并行根和证书，保留原贡献项与初始范围语义，明确unresolved/公共F32深度舍入。世界VJP、完整方法成本、GT、训练和新颖性仍未完成。标准cache拆分不主张新的数学算法。
