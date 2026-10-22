# CUDA 自主候选、认证与回退：固定模型

2026-10-14。device_budget_v1已完成：7322项全部两路径区间覆盖，220项设备mass分类正确；已有CPU bracket条件认证通过。此前已有bracket不输入本轮设备求根。

输入只有原36列参数和固定110模型的start/count。GPU先构造预算/缓存，再以名义系数产生候选：低路径将名义p/rho/alpha转F32，高路径保留原amplitude和名义energy、p/rho（F64）。候选可以使用标准math函数，它们不供证明；全算法输出必须通过独立严格区间残差符号。

普通候选用稳定prefix amplitude/energy常数和log尺度尾部比较、全局bracket与二分；两个名义alpha=.5/E=0的模型用解析加权peak。这只提出候选，不把浮点零残差当证明。低候选未给严格符号则高路径重新产生候选并认证；两路径都未认证输出unresolved。

端点用strictly below exact1/1000的binary64 tau0=0x1.0624dd2f1a9fbp-10，lo=roundUp(z-tau0)、hi=roundDown(z+tau0)。exact Fraction核验tau0<1/1000，以及每个返回端点距z≤1e−3、宽度≤2e−3。只有low.samePath两端或high.samePath两端有严格符号，才返回finite certified。no-root由设备区间mass分类，不强造有限根。

全部18旧+88新有限模型共106，4无根（2critical/2insufficient）不删。门槛全部finite输出包含独立原世界参照且误差≤1e−3，有严格证书；4分类正确。unresolved/非有限/错误分类都失败。MP值只在设备返回后核验，不参与选择或回填。fixed模型原accepted集合和camera-ray单位保持，未验证renderer的5步初始化/选择或全图。

分别保存低/高使用情况与失败、原输入、编译源/产物、独立复核；预算、求根(含证书)、两者合并分别计5批CUDA event。诊断output/metadata不是生产cache/layout，未包含完整renderer/GT/backward，没有效率采用结论。
