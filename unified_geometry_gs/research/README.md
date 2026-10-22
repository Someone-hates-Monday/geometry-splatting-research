# 科研推进记录

> **交接快照（2026-10-22）**：[统一交接入口](../../HANDOFF.md)与[研究脉络](../../handoff/RESEARCH_MAP.md)。最新封存阶段为 TGS v29；以下“最新／当前”按写作当时理解，不是全项目当前状态。下一工作包尚未实现，没有新训练或 v30 结果。

本目录为 2026-10-12 新增审查与独立实验，不覆盖 `output_compare` 历史结果。
持续目标仍为学术上有价值、经过验证的理论补充或几何/图像/效率改善；目前尚未完成。

最新 `DEVICE_ROOT_RESULT.md`：设备原参数预算7322项两路径58576包围检查、110×2质量分类全过；自主产生并认证106finite根（76low/30high），4no-root全正确。
独立MP240复核106新括区/4408系数/精确宽度通过。无CPU root/bracket回填，max1.36184e−5 camera-ray误差。
合并event705.568ms/110数学模型，完整renderer/初始范围/backward/GT与效率未验证；下一步便宜充分认证及完整成本。原生产训练未改。

当前 `GUARDED_PRECISION_RESULT.md`：条件精度CPU固定检验通过。全旧7010项区间覆盖，18旧+88新有限根全有证书，4无根分类正确。
低路径76有限+4无根直接认证，30有限根回退；最大误差1.36184e−5 camera-ray单位。独立MP240复核106对端点/精确宽度和4408系数通过。
CPU区间无界指数不是GPU实测，预算/原参数偏移/稳定尾部仍需device实现及完整成本/梯度/质量验证。首版81个浮点宽度微超问题保留，v2同输入修复。
原LaTeX原位v26编译成功；当前未启动新的GPU训练，研究目标仍未完成。

后续 `STRICT_INTERVAL_RESULT.md`：共享严格数学核心不使用standard exp/log库；host774scalar/212端点、CUDA212端点全过。
GPU局部212端点中位event222.99ms，使用CPU区间与括区，未含device预算/求根、完整renderer、backward或GT；不能宣称加速。
下一步便宜充分认证、设备预算和完整成本。用户已明确允许WSL/CUDA，原自动审批timeout已解除，不重复请求授权。

此前 `COHERENT_FACTOR_RESULT.md`：一致QR_mixed在9真实异常中max根误差4.99316e−5，
但旧4096局部失败12、9临界双层失败5，固定主候选拒绝，全7010输入保留。
新增合法F32世界的正确舍入射线系数别名：同6个F32系数对应物理根相距>.976，仅该接口的统一误差下界>.488，
有向区间端点证书完成；不是所有metadata布局不可能或GT质量结论。下一项条件精度/质量裕量，当前无新训练。

最新机制结果：`EXACT_OUTLIER_COEFFICIENT_RESULT.md`。全部9×8混合数学模型显示有效alpha差主导；
只替换peak/rho的根变化最大9.58099e−7 ray。独立原PLY MP80/140两参照及共同opacity参考均factor更近8/9，
但仍有误差且不是GT改善/采用结论。根间tail充分界与MP180 direct-H检查定位3例跨平台两侧，
其余6项不充分保留。科研图保存全部9点总览及最大2例曲线。下一步一致能量构造及条件误差界。

最新（2026-10-14）：全9个冻结异常像素的采集、18个独立高精度根、18组CUDA位重放已完成。
全18保存系数模型都有有限根；最大观测差−.18208014 ray中，数学表示根差已为−.18209892。
FP64稳定log-T不消除大差异。全9余项/观测差比例最多2.713%，不是单纯求根精度问题。
新固定binary在8视角重放旧输出，历史binary身份漂移失败保留；详见 `EXACT_OUTLIER_RESULT.md`。
原LaTeX原位v22编译成功。按用户要求本轮先简报与归档，下一步锁定alpha/peak/rho机制分离
及raw世界高精度参照，不扩大已失败的首版packet训练。

最新（2026-10-14）：首版完整图像exact packet三臂完成24视角，8视角最大差门槛失败，
共9像素>1e-3，最大.17964；packet/factor完整GPU时间比2.60018，效率门槛亦失败。
全图保留、数值产物与二进制/原模型核验完成，不扩大GT或训练。
见 `FROZEN_EXACT_RENDER_RESULT.md`；下一项为全部异常像素的实际前缀/系数/稳定根重建。
反向链与TGS原文实现审计见 `EXACT_PERSPECTIVE_BACKWARD_AUDIT.md`、`TRIANGLE_THIN_LAYER_AUDIT.md`。

最新已收束：输入精度64组合筛查后，锁定两个候选在新4096构造失败13/11，均拒绝；
有限扰动界与特定量化表示双点反例已核验。实际复制生产CUDA预处理的诊断扩展覆盖原1992输入，
FP32 CUDA局部前向全部通过；未测完整反向/成本/GT收益，不撤销合成失败。
见 `PACKET_INPUT_PROTECTION_RESULT.md`、`PACKET_CONDITIONING_THEORY.md`、`PRODUCTION_PACKET_RESULT.md`。
下一步完整CUDA/VJP和冻结模型三臂GT对照，按用户要求先汇报，当前不启动新训练。

最新（2026-10-13）：64输入精度组合的两个开发集通过候选，在新4096构造上分别13/11失败，均拒绝；
见 `PACKET_INPUT_PROTECTION_RESULT.md`。有限扰动界及同一FP32表示的双点反例见 `PACKET_CONDITIONING_THEORY.md`。
复制生产源码的独立CUDA探针 `production_packet_v3` 已完成，1992真实输入局部forward全部通过；
不是已安装renderer或融合实现，无CUDA backward、GT质量、效率或新颖性完成结论。
完整GPU内部诊断及独立产物复核仍待完成；当前LaTeX保持v19版本，尚未纳入这些最新结果。

此前 `CLIPPED_PACKET_RESULT.md`：clip-aware表达覆盖全部1992输入（含46clip），
6符号身份、128SPD构造及全部原PLY80位forward复核通过；71712 raw FP64梯度分量通过。
但128构造的FP32中1例峰位门槛失败，整体开发门槛仍FAILED，不能直接训练。
已round packet用80位求值仍失败，需研究输入精度保护；无真实CUDA预处理/计时/质量结论。

最新（2026-10-13）：匹配因素检查24投影/射线bitwise一致，192混合根独立重解通过。
透视packet重参数的9符号身份、64SPD构造与1946适用真实输入通过；
全部1992输入保留，46 clipped center域外不计通过。原PLY独立80位复核全部1946，
最大误差2.63e−12；尚无CUDA/VJP、效率、质量或新颖性结论。
见 `MATCHED_RAY_FACTOR_RESULT.md`、`PERSPECTIVE_REPARAM_RESULT.md`；下一步需clip一般化与GPU检查。

当前最新数学补充：`DIRECTIONAL_FIELD_RESULT.md`、`DIRECTIONAL_FIELD_THEORY.md`。
世界Gaussian逐项绝对方向衰减的单法向兼容条件、尖锐谱下界及双基元证书已证明。
24个固定真实参数查询点全部正证书，13源/archive、26产物、3PLY、48raw基元、
24有理数不等式和144独立空间导数分量复核通过。不是GT表面/生产渲染质量结论。
zonotope与方向场问题有明确已有工作，原创性尚未确立。当前LaTeX原位编译v17通过。
同24条完整世界相机射线亦检查完成，相对旧affine深度位移中位.01262537%、尾部2.69459%。
40源/archive、26产物、3PLY和24行算术复核；没有GT质量、单一因素归因或严格区间根界。
见 `WORLD_AFFINE_RAY_RESULT.md`；方向衰减差不能自动解释为深度/质量误差。
新断流诊断v2确认最近两轮`stream disconnected before completion`；
holdout和新诊断均正常结束，匹配进程为空；没有训练完成触发CUDA死锁的证据。

最新进展（2026-10-13）：全新42训练/7测试的factor64对照已完成全部19阶段，
实际源码、二进制、共同状态、数据、模型与指标逐项复核通过；相关进程已退出。
TEST PSNR +0.090075357 dB、TEST SSIM −0.000304571，未通过预设+.30dB和非降门槛。
TRAIN PSNR +0.421707653 dB，基元+3.07519%；无显著泛化、独立重复或加速结论。
见 `FACTOR_HOLDOUT_RESULT.md`、`results/factor_holdout55_quality_report_v1.json`。
不自动扩大该RGB实验；原49全视角几何pilot改善.601861%仍判失败。

数学新增：精确条件性log-scale VJP误差、可达紧界，以及固定非单位四元数族的四次
相对误差反例，见 `PRECISION_SCALE_VJP_AUDIT.md`。精确SymPy、64独立AD和164保存输入
已核验，完整渲染器与学术新颖性尚未证实。LaTeX原位补入此定理和完整留出负结果，
该阶段内置编译v12成功。局部直接因子3664块、Gram前向984块/20步smoke证据独立保留；
Gram未进行完整质量训练。研究入口与后续数学核验见 `CONTINUATION.md`。

后续数学工作已完成：`THIN_QUANTILE_LIMIT.md`证明薄层深度和位置梯度收敛需要
不同裕量，并给出每个有限厚度均正则但梯度不收敛的解析反例。
`QUANTILE_BAND_RESULT.md`给出区间深度的C1空间积分、条件梯度界及表面混合代价；
统计量本身已有文献先例，未声称新颖性。独立积分、60位差分通过原门槛，v1失败封存，
方案符号错误在独立勘误中保留。当前英文LaTeX完整原位补入并内置编译v13成功。
`figures/thin_quantile_evidence_v1.png/.svg`为有数据来源的理想模型证据图，非真实质量证据。

真实冻结开发诊断已完成3072条射线；2614个完整有限区间全部主数值检查、24个独立
前向/差分通过，但78.4621%触发假想薄层混合代理守护，统一区间替代门槛失败。
有效opacity梯度范数也可能增大，最大同射线比267.366；没有启动该方案训练。
见 `QUANTILE_BAND_FROZEN_RESULT.md`。额外q空间全VJP未收敛，源码/日志封存；
独立product-T全分量前向差分通过相同两个极端样本，13源/archive与全部输入行/汇总
独立复核通过；不撤回原积分失败。当前LaTeX补入这些边界，最新内置编译v15。

## 阅读顺序

1. `MATHEMATICAL_AUDIT.md`：原消融证据、NCC 精确导数、负增益等价性、GGGS 条件性与差距公式。
2. `MEDIAN_SOLVER_THEORY.md`：中位深度非正则点、根存在条件、带权括区证书及否定的效率结果。
   `HYPOTHESES_AND_EVIDENCE.md`：四项待区分机制、竞争解释、预测和文献重叠。
   `FIELD_SLOPE_AUDIT.md`：已有 field-occ 直通近似的全导数与平移不变性检查。
   `VALIDATION_PROTOCOL.md`：场景/种子依赖、评估噪声、新视角和效率口径。
   `median_depth_note.tex`：独立英文技术说明；首次编译超时，第二次内置编译已通过。
   `REVISED_SOLVER.md`：新停止规则、保留的舍入失败例和计算量核验。
   `NCC_PILOT_RESULT.md`：完整 NCC 负结果、评估复验及归因限制。
   `CUDA_SOLVER_RESULTS.md`：实际 GPU 效率、冻结网格与反传对照。
   `DEPTH_ERROR_BUDGET.md`：求根误差进入 TSDF 的条件性界。
   `IQI_AND_ADJOINT_REFINEMENT.md`：插值负结果、局部反向精化的误差界和开发梯度/计时。
   `ADJOINT_TRAINING_PROTOCOL.md`：完整共同状态恢复及 scan55 训练对照的固定方案。
   `ADJQUAD_PILOT_RESULT.md`：完整 scan55 结果、质量失败及独立梯度参照的尾部限制。
3. `results/*.json`：原始数值检查与冻结模型诊断。
4. `experiments/<run>/manifest.json`：训练命令、源文件 SHA256、环境、阶段状态、计时。

所有概率公式与合成测试均声明适用条件，不能直接冒充生产渲染器的真实场景结论。

## 已完成

- 现有 15 扫描中 11 个有配对结果；4 个缺 baseline，历史记录保持原样。
- NCC 的 CUDA 梯度与稳定化前向表达式不一致，已做符号计算、独立 float64 自动微分、
  有限差分、低对比度及真实冻结模型检查。
- 新后端隔离 JIT 构建在 `research/build`，不重装原 `warp_patch_ncc`。
  `--ncc_exact_gradient` 默认关闭；开启时只改一项导数，前向与 validity 逐位相同。
- scan37 两组各 50 步集成已通过，各调用后端 50 次，首轮 48,667 块有效反传。
  此模式提前启用正则，不能用于质量、耗时优劣或正式配方结论。
- 连续随机实体差距恒等式与反例已数值核验。
- 296 个有限根随机混合通过括区证书检查；旧混合 Newton 评估次数中位数 33，
  二分 32；新停止规则的结果另列如下，均不直接等于 GPU 加速。
- 三个旧模型各 1024 射线的 CPU float64 诊断完成，V(T=0.5) 的中位数约
  0.68–0.73。三个场景共 3072 条射线已通过 CUDA 抽样核对：有效性零差异，
  深度误差全部≤10⁻⁴、透明度误差全部≤10⁻³；未验证逐基元或反向等价。
- 已有 field-occ STE 在一维合成例中不保持根处斜率的共同平移不变性；
  独立隐式全导数通过重新求根有限差分。这定量界定近似偏差，未证明历史失败的原因。
- 重合峰混合的平方根临界性与“中位根可微、根处斜率不可微”反例已推导，
  并经独立乘积式 T + Brent 求根核验；全导数条件已补正。
- 新局部端点复核版 Newton 在同一 296 个随机混合上，普通场评估中位数 12
  （旧混合版 33、二分 32）；另外发生 91 次 80 位复核，排序与不同精度开销
  尚未计入。共 305 例通过；保留了初版平坦根舍入失败例，不作 GPU 加速声明。
- 三场景同一 CPU 射线在 tol=10⁻⁶ 下，新求根普通场评估中位数 6–7，
  二分 14–15；深度差≤5.009×10⁻⁷。GPU 计时与独立后端见下。
- 完整 NCC pilot 已完成：scan37 修正后 overall 退化约 0.01119 mm，
  三个 evaluator seed 方向一致；49 个训练视角 PSNR 增加 0.155 dB。
  首次 NCC 前两组已不同，不能严格归因。见 `NCC_PILOT_RESULT.md`，不扩大此路线。
- IQI 未通过训练梯度门槛，最大漂移 4.6336%；一阶反向精化通过 21/24 组。
  另存 Halley/二次分母变体通过 24/24 组，最大漂移 0.080616%，并在六个冻结视角的
  RGB+depth 前向/反向计时中降低延迟 6.5%～8.7%；尚无完整训练质量或吞吐结论。
- 真实 PGSR 检查点复现旧恢复顺序的外观参数优化器绑定问题；research wrapper 修正通过回归。
  短程两臂完整状态哈希和首相机一致、各 20 次 NCC 调用。scan55 完整配方两臂从共同
  6999 步检查点分叉到 30k，完成后统一提取、三 evaluator seed 评估和 49 训练视角图像评估。
  该完整对照现已完成：平均 Δoverall=+0.000424087 mm、Δs2d=+0.000908171 mm，
  Δ训练 PSNR=+0.066834 dB，未过几何门槛。同状态五轮控制重复已结束：
  相对原控制 Δoverall=+0.001001144 mm，比原候选差异更大，不能从单次对照隔离方法效应；
  原失败门槛保留，见 `results/control_variation_scan55_v1.json`。
- 当前 `median_depth_note.tex` 已补入条件性二阶/三阶 Jacobian 界，并通过内置编译；
  粗 upstream 仍可导致一阶完整损失梯度误差，已有独立单高斯例核验。
- 一个开发视角的 123 射线已通过独立乘积式自动微分及 12 条重新求根有限差分核验。
  122 条接受高阶修正；1 条跨峰值被拒绝，退回粗根后保留约 0.466% 有效射线 Jacobian 误差。
  不能把聚合梯度门槛通过解读成全射线真导数保证。
- 峰值自适应 CUDA 方案通过冻结梯度检查，但前向基本无增益、前向/反向慢 0.42%～1.15%，
  拒绝当前效率实现，未扩大训练；见 `experiments/peak_development_v1`。
- 真实训练早期十步的 profiler 已复核，仅累加实际 kernel/memcpy/memset：目标深度采样约占
  38.16%，渲染前后向约 29.38%，NCC Jacobian 内核约 0.36%。这些是设备活动时间和，
  不是墙钟占比。旧 `profile_summary.json` 含合成范围重计；只用 `profile55_trace_summary_v2.json`。
- 采样分组 batch1 的六冻结视角输出逐位相同，完整损失/反传计时降低约 4%～8.5%。
  追加 18 视角完整梯度有 12/162 组未过 0.1% 门槛；同后端重复也出现漂移。
  暂不启动已准备的训练计时，先核验投影参数至三维尺度/旋转的数值稳定性；
  见 `SAMPLING_AND_STABILITY.md`。没有完整训练加速或质量改善结论。
- 仅提高世界参数 backward 预处理中间精度的追加检查：18 视角、162 组全部通过，
  FP64 batch1 对 FP64 batch2 最大漂移 0.00333%。但六视角完整冻结路径仍比原控制慢
  约 0%～4.1%，尚无效率采纳结论；有条件精度方案另存检查。
- 选择性精度方案通过相同完整梯度门槛，但三对交替 500 步真实训练的计时墙钟比为
  1.001882、0.949033、1.023238，几何均值 0.990889，未过预设加速门槛。
  六进程均已结束，无挂起的这组训练；见 `results/mixed_training_timing_report_v1.json`。
- 单基元法向归一化 VJP 相对 forward 缺少原法向模长的倒数，已给出非奇异正定反例
  并用符号、自动微分、有限差分核验；这不解释法向上游为零的几何损失波动。
  见 `NORMAL_PREPROCESS_VJP.md`；没有声称质量改进，原生产源码未改。
- 仿射相机法向可化简为 n=−Pd/‖Pd‖，原未归一化长度为 1/|nᵀd|。
  已给出条件性代数证明与 64 个独立 float64 世界参数 VJP 计算图核验；尚未移植为
  独立 CUDA 世界导数参照，后续任务见 `CONTINUATION.md`。

## 运行环境与复现

使用现有 WSL Ubuntu 的 `/home/someone/miniconda3/envs/uggs/bin/python`，
从本仓库目录执行。隔离 CUDA 扩展构建需要以下环境：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/someone/miniconda3/envs/uggs/bin:/usr/local/cuda-12.8/bin:$PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=12.0

python research/validate_ncc_backend.py
python research/verify_stochastic_surface.py
python research/verify_median_solver.py

# CPU 独立参考；设置单线程以减轻与训练的数据加载争用：
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/audit_median_cpu.py \
  --model output_compare/ggs_pipeline/scan37 --views 8 --rays 128 \
  --out research/results/new_median_scan37.json

# 在当前成对训练结束后进行 CUDA 核对；输出文件名必须全新：
python research/validate_affine_cuda.py --cpu research/results/median_cpu_scan37.json \
  --out research/results/new_affine_cuda_scan37.json

python research/audit_ncc.py --model output_compare/ggs_pipeline/scan37 \
  --views 8 --samples 4096 --exact_backend \
  --out research/results/ncc_scan37_exact.json

python research/run_ncc_pair.py --scan scan37 \
  --out research/experiments/new_unique_name --smoke

# 正式训练结束后分别对 legacy/exact 运行，避免 GPU 争用污染训练计时：
python research/evaluate_images.py --model research/experiments/exact37_v1/legacy \
  --out research/experiments/exact37_v1/legacy_images.json
```

正式实验去掉 `--smoke`。输出目录必须全新，脚本拒绝覆盖已有目录。
实际已启动的正式实验为 `experiments/exact37_v1`：legacy 与 exact 各 30k，
λ_dn=0、λ_ncc=0.6、λ_geo=0.02、τ=0.9、随机单近邻、regularization 从 7k 开启。
两组均从头用现有固定种子 0 训练；GPU 不保证逐位确定性。
每组随后用同一 voxel=0.002、trunc=4、depth_max=8、cluster=1 提取网格并评估。
所有 checkpoint/point_cloud/mesh 保留，没有自动清理步骤。

`run_post_pair.py` 的 `exact37_v1/postprocess_v2` 已完成三个冻结旧模型的
CUDA 射线核对和 legacy/exact 各完整 49 个训练视角的图像评估。
v1 保留 SSIM 脚本维度报错记录；v2 修复后重跑，未覆盖模型或失败记录。

`seeded_eval_v2` 已在两组冻结网格上用预定 evaluator seed 0/1/2 复评，
并额外重复 legacy seed 0，获得相同数值。v1 因前置评估失败而未启动。
每次使用带源码记录的隔离副本，不改原评估；这是评估重复，不能冒充训练种子。

`median_cuda_backend.py` 隔离构建原始同源 control、四轮细分 four 和
Newton+端点检查 hybrid。保留原第一轮括区/可见性门控，检查失败回退原四轮。
GPU 端点采用工程浮点余量，不是区间算术证书；解析斜率输出与旧 secant 有差异。
`benchmark_median_cuda.py` 在三个旧模型、预定 24 个视角做正确性和随机顺序
CUDA event 计时已完成：hybrid 慢约 22%～25%，四轮控制延迟降低约 14%。
六次重新提取的三个场景均通过预设 0.0005 mm 工程容差；反传均有限，
但参数梯度漂移最高 1.396%，未过 0.1% 门槛，所以训练保留五轮默认值。
完整数据 `results/median_cuda_frozen_v1.json`；解释见 `CUDA_SOLVER_RESULTS.md`。

冻结网格提取的 opt-in 入口（WSL，需上述 CUDA 环境；输出目录必须新建）：

```bash
python research/extract_with_solver.py --model output_compare/ggs_pipeline/scan37 \
  --variant four --out research/experiments/my_new_frozen_extraction
```

该入口不重装后端，前后核对原模型及元数据哈希。三个模型上的效率/网格证据
不等于新颖性、全数据集等价或训练吞吐改善；参见 `DEPTH_ERROR_BUDGET.md`。

`training_integration.json` 记录后端调用、首轮有效块及反传、峰值 torch 显存和墙钟。
峰值 torch 显存不包含所有驱动/第三方分配，墙钟包含初始化、诊断与保存。
两组顺序执行一次的时间差不能证明加速。

## 必须继续完成的证据

- 首个完整成对实验和评估重复已保存，几何门槛未通过；不更改门槛找正结果。
  负结果报告包含输入状态不同这一归因限制；尚未做分区域可视化或独立训练重复。
- 图像质量另用冻结模型做明确相机列表的完整评估。当前正式脚本的 `--quiet`
  会隐藏训练报告 stdout，且未配置 TensorBoard，因此不能从日志承诺读取 PSNR。
  `train.py` 内置报告只用 5 个训练视角；历史另有 16 视角诊断，两者均不是新视角评估。
  `evaluate_images.py` 已备好只读冻结评估入口，明确记录每个相机与两种 PSNR 约定，
  并拒绝把 eval=False 的模型标为新视角测试；完整真实调用已完成。
- 有方向性改善后，至少原三个 pilot 和三个种子复验；增加独立新视角训练协议，
  报 PSNR/SSIM/LPIPS、Gaussian 数、模型体积、渲染耗时及峰值显存。
- 冻结真实模型测 Δ、界面斜率、固定括区失败率；区分表示差异和数值求根误差。
  结合 Gaussian Wrapping、TSPE-GS、PGSR 进一步核验新颖性。
- 补齐缺失的四个 baseline 后才可作全 15 场景对照结论。

## 工具准备

已安装 `jupyter-notebook`、`sympy`、`scientific-critical-thinking`、
`hypothesis-generation`、`statistical-analysis`、`scientific-visualization`、`literature-review`。
官方技能来自 openai/skills；后六项来自 K-Dense-AI/scientific-agent-skills，
固定 SHA `92ace75ac21efe19a620434e0ca4e356081fe807`。新安装技能需在后续轮次加载。
已实际阅读并应用 SymPy、批判性审查、假设生成和文献审查流程。来源：
Kassis et al., Scientific Agent Skills (2026), https://doi.org/10.48550/arXiv.2609.00065 。

Consensus 文献检索/记录读取、Wolfram 符号计算已有成功调用；
Wolfram context 查询曾内部错误，Hugging Face 论文工具当前不可用。
没有声称已安装 Blender 插件；3D 使用本地可运行的 Open3D 0.19.0 / Trimesh 4.11.3。
