# 共享几何预算：固定输出逐位保持，全部24视角缓存构造通过

2026-10-14。协议 `SHARED_CACHE_PROTOCOL.md`，主批 `experiments/shared_cache_v1`，独立审核 `results/shared_cache_audit_v1.json`。

## 实现与固定核验

将冻结gp::decode在query之前的inverse/QR/polar/residual与mean预算拆入按primitive/camera构造的缓存；query之后的原公式完整保留。缓存28double=224B/primitive，包括两路径状态/eta、delta_Q、nu、名义mean、真实mean区间、low/high六项U及padding。low U仍是F32值精确提升；高U为F64，预算仍按原方式向上F32。

该格式是诊断接口，不是56B拟定生产格式，也不称最优内存设计。保留真实mean区间是为了原严格aligned分支，不以名义共线假冒energy=0。query与AMP在build中不读取，decode时单独输入；源参数和物理目标不改。

原7322 source全部26double×2预算、106finite/4no-root的根与证书数组均逐位一致（含NaN payload）；重编original也与reciprocal_root_v2逐位一致。重复8次query共58576项的完整输出同样逐位一致。它继承先前MP240世界参照，不是新增未接触样本确认。

独立CPU核对25来源、28产物、12生成文件/实际binary/ZIP，检查原query后代码未变、完整预算/根位及缓存字段与原输出逐项相同，重算5对交替时序比和24视角元数据。真实缓存内容hash与状态在运行中记录，真实大数组未永久保存供第二次离线/MP全量审核，限制保留。

## 固定重复查询效率

5对交替比较原repeat preprocess与build once+cache decode；合并event比几何均值 **0.830169316**，约 **16.9831%** 时间减少，预定全位一致且比≤.95的局部重用门槛通过。所有输入已用于开发，8次重复不构成8个独立场景或真实性能泛化。

计时包括Torch输出allocation/fill以及query入口的ID范围检查/sync；不含host输入传输或完整renderer/optimizer。build、decode、原repeat和合并的完整event/wall另存于fixed_summary.json。不能从这批小样本直接推断870M真实关联或训练效率。

固定批次event中位数：build7322为0.081056ms，decode58576为1.160480ms，原repeat为1.481312ms，build+decode为1.231136ms。两项单独中位数不能当作每对合并值相加。

## 全部真实基元缓存

scan37/55/63各8个既定视角，原PLY/activated输入和相机身份与此前真实CSR输入核验一致。对每个camera的全部primitive构造缓存，不抽样、不根据可见性删掉失败。24视角两mode状态全部0、所有缓存值有限。

每视角完整实际缓存约114–236MB（224B行）；对应原source rows另需288B/primitive，本次是可复核的诊断中间格式。每view测5次build与event/wall、峰值allocated/reserved/added、状态、eta分布及缓存内容hash，编译/Scene加载另计。Source q/AMP只是忽略的占位，**并未完成全部真实CSR射线解码或根证书**。

各view的build event中位数范围3.343168–37.605183ms，保留全部时序不剔除高值。scan37 P=1,010,226/cache226,290,624B；scan55 P=1,052,385/cache235,734,240B；scan63 P=510,660/cache114,387,840B。该构造成本仍须纳入后续完整方法比较。

## 下一项

以已保存的动态CSR关联按需/批次解码，明确实际pixel query与AMP来源，避免全量26double/mode数组常驻；然后将候选/证书按射线内并行处理，保留原物理根与初始范围语义，明确unresolved和公共FP32深度的额外舍入。世界参数VJP、完整渲染成本、GT、跨场景配对训练/留出图像和新颖性仍未完成。原生产源码未变，主批session55325正常exit0。
