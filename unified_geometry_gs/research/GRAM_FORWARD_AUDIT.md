# 保留的四个前向失败：Gram 行列式诊断

2026-10-12。此独立实验不修改 `factor_quality55_v1` 的任何排队源文件或门槛。

压力集原四个前向失败均为 conic：scan37 ids229809/574033、scan55 id163023、
scan63 id342897。其中三个在投影范围内；投影范围并不证明真实可见或重建贡献。
原最大归一化误差 .00290671，原门槛 .0001。

投影协方差是 M 两列 a,b 的 Gram 矩阵 C₀=[a b]ᵀ[a b]。标准恒等式给出

    det(C₀)=‖a×b‖²,
    det(C₀+kI)=‖a×b‖²+k(‖a‖²+‖b‖²)+k², k≥0.

其未截断导数为 ∇ₐdet=2b×(a×b)+2ka、∇ᵦdet=2(a×b)×a+2kb。
这避免最后一次大数乘积相减，但不排除叉积内部抵消、输入舍入、上下溢或病态性。
不声称新代数或普遍稳定性；已有 floor / mask 边界仍须单独处理。

`gram_forward_probe.py` 用原 computeCov2D 的新隔离命名空间输出 Gram determinant，
保留原 float32 图、相机 clamp、射线和法向运算。coef 用同一恒等式，原 floor 保留。
`audit_gram_forward.py` 精确符号展开通过，复用全部 44+120 输入、不删样本、不改原门槛。

`experiments/gram_forward_v1` complete：

- conic 失败从4降至0，最大误差从 .00290671 降至 .0000276635。
- 六输出块在全部164例通过原 .0001 / floor1e-5 规则，共984个前向块检查。
- 所有射线和法向输出逐位相同；不透明度补偿的差异仅在预定非零kernel合成输入中核验。
- 实际原源码、生成源码、全部GLM内联依赖、输入与二进制哈希保留。
- 不测试 null-scale 特征分解 fallback；不是反向、完整图像、几何质量或计时证据。

另建 `gram_factor_preprocess_backend.py`，forward 局部函数与探针逐字节一致，
backward 与 factor64 逐字节一致。`GRAM_FORWARD_SMOKE_PROTOCOL.md` 在完整smoke观测前
写定，20次真实Adam检查同一初态、有限梯度，以及首损失相对漂移≤1e-4（floor1e-5）。
`gram_smoke55_v1` 仅smoke，不自动进入完整训练；状态见其manifest。
任何新质量对照须另写协议，保留正在完成的 factor64 结果，不能把新前向收益归到该旧候选。

工作流参考 Kassis et al. (2026),
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。
