# 动态求根稀有分支的完整条件审计

主批和最初固定891射线核验结束后，固定本探索性补充。审查全集：全部256 no-finite-root、全部29 high-mode accepted、全部26 short-certificate accepted的并集，允许重叠，不根据MP通过与否删除。最初固定探针未包含无根案例，不能用原gate涵盖该分支。

加载原冻结模型/相机/实际query/AMP与已固定CUDA binary，仅重建选定源系数，不重新计算或替换已存公共根。所有source/metadata/binary SHA检查。保存完整源及相同最终mode的26double系数。独立MP80/140世界逆矩阵核查系数及无根beta>=1/2；MP240稳定scaled-log检查既存public float32修正根的端点符号/区间/中心误差。全集审计不提供随机总体误差率/留出性能。任何错误保留并阻止采用。

第一view此前decode事件3258.099ms原样保留，不因为疑似首次加载就删除；本次提取不重新替换主批计时。完整方法速度/梯度/GT质量不在此检查范围。
