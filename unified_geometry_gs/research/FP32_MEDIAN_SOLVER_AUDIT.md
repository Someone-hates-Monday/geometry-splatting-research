# FP32 median 静态审计

2026-10-14。依据当前[生产源码](../submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu#L559)、[配置](../submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h#L31)及[真实异常复核](EXACT_OUTLIER_RESULT.md)，未追加实验。

实际gate为累计T≤.45；初始范围±.4，SPLIT8、5轮。逐项FP32乘积含expf/rsqrtf；T≈.5时可能将不同实数值舍入为同值。首轮端点用≥/≤确认包围，内部start_id用≥选择最右满足点；舍入平台可使分支偏右。后四轮复用端点值，不能恢复已丢失的符号信息。

最终割线权重经saturate限制于区间，不认证真实根；等端点会产生零分母，本审计未独立核验NaN映射。保存slope为带1e−6区间分母floor的割线，不能替代真导数或整个区间下界。若真实斜率很小，系数/求值扰动可放大为深度差；具体归因须看捕获trace。

实际18组保存模型的精确终端乘积均<.5，全部有限根。FP64稳定log-T对各自数学根最大误差2.32282e−7 ray，仍保留两表示的大根差：当前主要问题在保存系数的表示差，尚无raw世界参照判定哪臂正确。两高斯toy总T≈.5不通过.45 gate，不能充当完整renderer反例；其它未验证toy亦非结论。

唯一后续诊断：锁定772项做alpha/peak/rho八种混合根并接入raw高精度参照；同系数数值重放估求值误差、混合数学根估表示影响，真实斜率解释敏感性。不得把分配结果当GT收益或独立因果证明。
