# 全部9个冻结异常像素：本轮完成结果

2026-10-14。目标尚未完成。本轮是完整24视角失败后的事后机制诊断，不是新的质量或效率成功。

## 已完成

- `exact_outlier_capture_v3` complete/exit0：全8个受影响视角均重放旧存档的两臂深度、割线斜率及RGB/alpha/normal FP32位模式。完整保存全部9个超阈值像素、17581个tile候选、772个accepted项以及实际prefix。
- 历史二进制因构建架构环境变化被重新编译，恢复旧环境后仍未恢复旧SHA。失败与旧SHA保留；v3固定独立的新binary SHA。输出重放等价不代表历史隐藏buffer或指令身份恢复。真实buffer与复制表达式probe的范围见 `EXACT_OUTLIER_REBUILT_PROTOCOL.md`。
- `exact_outlier_root_v1.json`：18/18组保存FP32系数的精确有理数终端乘积都小于1/2，全部存在有限根。120位稳定残差独立求根，36个根（含另列的camera-origin模型）的180位端点符号复查通过。不是向外舍入区间证明。
- `exact_outlier_solver_v2` complete/exit0：固定同一输入/初始区间、SPLIT8/5轮、真实初始gate .45和range .4，原FP32路径18/18深度与割线斜率逐位匹配。提高乘积精度及改用FP64 log-T的三路径trace全部保存。

## 结果及解释

最坏点scan55/cam0000/(161,215)，原z-depth差绝对值0.1796405315。在probe rln定义的射线长度坐标下：

`观测packet−factor −0.1820801374 = 保存系数数学根差 −0.1820989237 + 剩余数值/转换差 +0.00001878635`。

全9点剩余差最大绝对值8.9645251e−5 ray；相对各自观测差的最大比例2.7124813%。因此这些大差异主要已经存在于两组保存系数的数学根中，不能归结为单纯FP32乘积求根误差。FP64稳定log-T重放也没有消除大差异。

18个真实根的−T_t范围约3.1154e−4至1.97018e−2；不能用保存割线斜率替代真实斜率，也不能把根处斜率当整个根间区间的统一下界。稳定double与MP根的最大差约4.40e−13 ray；MP有限五轮搜索与MP真根最大差2.11056e−7 ray。

独立只读收尾复核 `results/exact_outlier_closeout_audit_v1.json` 检查22 solver源及archive、20产物、3生成源、1 binary、54组trace header及216次endpoint复用。相对同系数MP根，原FP32路径最大误差1.31649e−4 ray；只保留FP32 per-item factor而提高乘积/插值精度的mode1最大3.83096e−4，5/18输入反而更差。FP64稳定log-T的mode2最大2.32282e−7 ray，18/18更接近各自表示的根，但不能消除两种表示之间的大根差。mode1同时改变乘积与插值、mode2同时改变factor求值和log-T/插值精度，不是单个算子因果实验。

采集系数的逐项alpha差最多约1.01924e−5，但尚未隔离alpha、peak、rho各自影响。没有独立raw世界高精度参照，不能宣布packet或factor哪个是几何真值。9点均由异常选择，不能估计训练发生率、证明GT改善或科研新颖性。

## 保留的失败及决定

完整24视角比较仍失败：8视角、9像素超过1e−3最大差门槛；packet/factor GPU耗时比几何均值2.60018，效率门槛亦失败。不更改门槛，不扩展该版本训练。

capture_v1历史binary身份失败、capture_v2变量遮蔽错误、恢复binary环境失败均保留。solver_v1误写.5初始gate，9个实际初始T均≤.393823，恰未改变这些输入的结果；其源归档不改。v2已直接include真实config，不能把v1外推到其他输入。

## 后续最小计划

1. 先锁定同772项与全部9点，制定alpha/peak/rho的2³混合系数根对照，区分各表示误差及交互；仅作代数机制分离，不冒充因果或质量实验。
2. 从原PLY和相机参数独立高精度重算实际沿射线系数，判断两条实现对raw世界参照的偏差，检查根间平台和必要裕量。
3. 仅当得到可解释、可修复且有成本优势的机制，才推进完整后向核验和预登记质量对照。薄层深度/梯度不同裕量仍是理论候选，须补足相关工作与真正原创的证明或方法。

来源：`results/exact_outlier_summary_v1.json`、`results/exact_outlier_root_v1.json`、`results/exact_outlier_capture_schema_v3.json`、两组experiments目录及原失败协议。原LaTeX已原位更新并内置编译成功，记录见 `results/latex_compile_status_v22.json`。

收尾静态实现审计见 `FP32_MEDIAN_SOLVER_AUDIT.md`。本轮只读 `/proc` 健康快照 `results/exact_outlier_health_v1.json` 未发现capture/solver/oracle、明确列举的研究队列或项目train.py活进程；这是瞬时检查，不是GPU活性或历史死锁证明。
