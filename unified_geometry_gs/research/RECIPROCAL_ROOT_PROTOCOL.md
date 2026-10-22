# 固定优化：精确整数倒数区间表

2026-10-14。device_root_v1固定开发通过但110模型合并event约705.568ms。本轮只替换严格级数中1/k的重复IEEE double division；不改阶数、余项、候选、预算、门槛或输入。

k=1..51，以Fraction(1,k)构造相邻binary64下/上端点，证明lo<=1/k<=hi且无中间binary64值。这等于IEEE roundDown/roundUp(1/k)，故原div(term,pt(k))中的两次动态倒数可替换为读取相同上下界、执行原mul。表在CUDA constant memory；6个调用位置改写，其他动态除法保留。

同一进程重编原archive控制与单一新variant，读取完全相同7322原参数/110模型start/count。预算、根/分类/符号/宽度的全部数组（含NaN位置）要求bitwise一致；不以epsilon替代。这样继承原独立MP240数值核验，但不称新的未观测输入确认。

计时预热后5对交替顺序（control/variant顺序逐对交换）测预算、根+证书、合并event和合并wall。报告每对ratio及geometric mean。开发采用要求所有输出bitwise相同且合并event ratio<=.95；failure保留，不改阈值/样本或筛选更多表。局部采用不证明完整renderer/GT/训练速度或学术新颖性。

保存精确表证书、输入、source/generated/archive、两臂输出与完整paired timing。本优化不是新的数学方法，仅优化已验证数值核心。通过再进入完整成本与反向链，原生产CUDA训练不改。
