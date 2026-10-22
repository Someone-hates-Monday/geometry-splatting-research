import { useState, useHostTheme, Stack, Row, Grid, H1, H2, Text, Table, BarChart, LineChart } from "cursor/canvas";

const tabs = ["现有证据", "数学发现", "实验门槛", "CPU 射线诊断", "求根原型", "完整 NCC pilot", "GPU 求根对照", "高阶反向精化", "采样与反传稳定性", "世界参数导数", "薄层与区间深度"];
export default function ResearchAudit() {
  const theme = useHostTheme();
  const [tab, setTab] = useState(10);
  const tabStyle = (active: boolean) => ({ padding: "8px 14px", cursor: "pointer", border: `1px solid ${theme.stroke.tertiary}`, borderRadius: 5, background: active ? theme.fill.secondary : theme.bg.editor, color: active ? theme.accent.primary : theme.text.secondary });
  return <Stack gap={22} style={{ padding: 24, maxWidth: 1120, margin: "0 auto", color: theme.text.primary }}>
    <Stack gap={6}>
      <Text size="small" tone="secondary">3DGS / GGGS / 2DGS / 三角形 TGS · 2026-10-13 审查快照</Text>
      <H1>质量门槛失败；薄层梯度反例与区间深度定理已核验</H1>
      <Text>factor64全49视角几何pilot和42/7留出RGB对照均已完成，分别未过预设几何、TEST PSNR与SSIM门槛。最新进展为固定射线模型的薄层极限、正则根梯度反例和分位区间的条件稳定性证明；新颖性、真实质量收益与加速仍未证实。</Text>
    </Stack>
    <Row gap={8} wrap>{tabs.map((label, i) => <button key={label} style={tabStyle(i === tab)} onClick={() => setTab(i)}>{label}</button>)}</Row>
    {tab === 0 && <Stack gap={24}>
      <Table headers={["证据范围", "已核验", "解释边界"]} rows={[
        ["完整 DTU 配对", "11 / 15；10 改善，1 退化", "缺 scan69/106/114/118 baseline"],
        ["11 场景平均 Δoverall", "−0.0295654 mm", "训练与提取组合，不能归因于单个模块"],
        ["8 个非 pilot 配对", "平均 Δoverall −0.0316400 mm", "描述性结果，不代表独立重复与统计显著性"],
        ["NCC 冻结模型诊断", "3 场景，98,304 个采样块", "图像块有相关性，不当作独立样本做显著性检验"]
      ]} />
      <Grid columns="repeat(auto-fit, minmax(320px, 1fr))" gap={24}>
        <Stack gap={8}>
          <H2>已接受 NCC 块中的负相关比例</H2>
          <Text size="small" tone="secondary">纵轴：比例（%）；横轴：DTU 场景</Text>
          <BarChart categories={["scan37", "scan55", "scan63"]} series={[{ name: "负相关 / 接受块", data: [5.81320451, 0.4306437, 4.8522794], tone: "neutral" }]} height={220} valueSuffix="%" />
          <Text size="small" tone="secondary">来源：research/results/ncc_scan*.json；每场景 8 视角 × 4096 块，seed 0，2026-10-12。负相关不自动等于错误匹配。</Text>
        </Stack>
        <Stack gap={8}>
          <H2>原 G1 门槛扫描的平均 overall</H2>
          <Text size="small" tone="secondary">纵轴：平均 overall（mm，越低越好）；横轴：NCC 阈值 τ</Text>
          <LineChart categories={["0.3", "0.5", "0.9", "1.2"]} series={[{ name: "原 pilot 均值", data: [0.4001606, 0.3834533, 0.3832441, 0.3841185], tone: "neutral" }]} height={220} beginAtZero={false} valueSuffix=" mm" />
          <Text size="small" tone="secondary">来源：output_compare/metrics/g1_verdict.json；原三个 pilot 快照。本曲线不能证明普遍最优阈值，τ=1.2 基本关闭相关性筛选。</Text>
        </Stack>
      </Grid>
      <Text tone="secondary">原三个 pilot：no_dn 的平均 Δoverall 为 −0.00377814；冻结 no_dn 后 trunc4 为 −0.02033675。不同评估随机采样使这些差值不能与当前结果逐位相加。图像指标来自训练视角。</Text>
    </Stack>}
    {tab === 1 && <Stack gap={20}>
      <Stack gap={8}>
        <H2>1 · 精确导数修正，默认关闭</H2>
        <Text>A = xᵀx，B = yᵀy，C = xᵀy，S = C² / (AB + ε)。正确 ∂S/∂B = −SA / (AB + ε)；现有 CUDA 写成 −S / (B + ε)。</Text>
        <Text>隔离 CUDA 后端保留原前向与筛选逻辑，低对比度合成测试中两者逐位相同。深度梯度绝对误差中位数从 0.0448646 降至 9.96×10⁻⁷；这验证导数，不验证 Chamfer 改善。</Text>
        <Text tone="secondary" size="small">来源：research/results/ncc_backend_validation.json；纹理对比度 0.004；参照独立 float64 自动微分及有限差分，2026-10-12。</Text>
      </Stack>
      <Stack gap={8}>
        <H2>2 · GGGS 等价条件、偏差恒等式与局部误差界</H2>
        <Text>log(V(t)/T(t)) = Σ后峰基元 log((1−Gᵢ(t))/(1−αᵢ)) ≥ 0。该式明确 T=V 何时成立，补充原论文已陈述的近侧条件。</Text>
        <Text>若 T(tₘ)=q、Δ=log(V(tₘ)/q)，且 log V 在含根的区间严格递减，负斜率介于 m 与 M，则 Δ/M ≤ tᵥ−tₘ ≤ Δ/m。必须先满足单调性与根存在条件。</Text>
        <Text>两个 α=0.4、峰相隔 1、宽度 0.03 的基元可使 T 具有中位交点，而全局 V 始终大于 0.5。正反向射线中位位置约 0.978 与 0.022；数值求根更精确并不能消除这一表示差异。</Text>
        <Text tone="secondary" size="small">来源：research/results/stochastic_surface_verification.json；200 组随机混合恒等式误差 ≤1.07×10⁻¹⁴。此为合成数学证据；创新性与真实场景 Δ 分布尚未确认。</Text>
      </Stack>
    </Stack>}
    {tab === 1 && <Stack gap={8}>
      <H2>3 · 有限根不保证稳定的隐式梯度</H2>
      <Text>单基元 α=0.75、q=0.5 时，中值在峰值，界面导数为零；重合峰混合也具有平方根临界性。另有中位根可微、根处斜率不可微的反例：其一侧斜率导数约为 −26.47755 与 +26.47755。</Text>
      <Text>带权分位数可构造包含有限根的括区，296 个随机混合通过独立 Brent 根核验。旧混合 Newton 的评估次数中位数为 33，二分为 32；新停止规则另列于求根原型页，GPU 原型另列于 GPU 对照页。</Text>
      <Text tone="secondary" size="small">来源：MEDIAN_SOLVER_THEORY.md、results/median_solver_verification.json。普遍结论来自证明；合成测试用于检查实现。真实训练敏感性尚未验证。</Text>
    </Stack>}
    {tab === 1 && <Stack gap={8}>
      <H2>4 · 既有 field-occ 直通近似的边界</H2>
      <Text>根处斜率的全导数为 q[Hₜθ−HₜₜHθ/Hₜ]，需 Hₜ&gt;0 及相关二阶偏导存在；避开所有贡献峰位是充分条件。现有 STE 没有完整计入 Gaussian 值与中位根随参数的移动。</Text>
      <Text>全部一维峰位同时平移时，根处斜率应保持不变。单基元 α=0.9、σ=0.1 的精确平移导数为 0，现有 STE 转写约为 +75；四个例子的隐式全导数与重新求根有限差分最大误差为 5.04×10⁻⁶。</Text>
      <Text tone="secondary" size="small">来源：FIELD_SLOPE_AUDIT.md、results/field_slope_calculus.json。此为已声明 STE 的合成偏差检查，未调用生产 CUDA，未证明它造成历史质量退化。当前 NCC 训练关闭此项。</Text>
    </Stack>}
    {tab === 2 && <Stack gap={20}>
      <H2>先隔离机制，再决定是否发展为论文主线</H2>
      <Table headers={["阶段", "固定项", "判定依据"]} rows={[
        ["训练集成检查", "scan37；两组各 50 步；提前启用正则", "已通过：两组各调用 50 次，首轮 48,667 块有效反传；不评价质量"],
        ["NCC 单因素 pilot", "scan37；两组 30k；no_dn + 同一 trunc4", "几何、训练视角图像、耗时和显存；只作探索性判断"],
        ["扩大复验", "既有三个 pilot；至少三个种子；独立新视角协议", "平均 Δoverall≤−0.0005 mm、Δs2d≤0、ΔPSNR≥−0.2 dB；同时报告波动"],
        ["理论路线决策", "冻结已有模型；测量真实 Δ 与跨视角深度误差", "若 Δ 近零则保留适用条件说明；有稳定影响才开发表面提取方法"]
      ]} />
      <Text>采纳门槛是工程判据。没有重复实验时，不使用“统计显著”或“论文级提升”的表述。正相关筛选、稳定方差与 TSDF 改进各自单独注册，避免同时改动造成归因混乱。</Text>
      <Text tone="secondary">工具：已安装官方 Jupyter 和六项 K-Dense 科研 skills；Consensus 与 Wolfram 有成功调用；Hugging Face 论文工具当前不可用。3D 使用已验证 Open3D / Trimesh。来源与固定版本见 MATHEMATICAL_AUDIT.md。</Text>
    </Stack>}
    {tab === 3 && <Stack gap={18}>
      <H2>真实冻结参数上的独立参考测量</H2>
      <Text>每场景 8 相机 × 128 个均匀像素，seed 0；包含背景。下列统计限于存在有限透射率中值的射线。3072 条射线通过 CPU/CUDA 前向抽样核对，深度和透明度在预定容差内，有效性零差异；这些机制量不等于生产错误率或质量改善。</Text>
      <Grid columns="repeat(auto-fit, minmax(320px, 1fr))" gap={24}>
        <Stack gap={8}>
          <H2>T=0.5 处空隙率 V 的中位数</H2>
          <Text tone="secondary" size="small">纵轴：V（无量纲）；横轴：DTU 场景。理论等价参照值为 0.5。</Text>
          <BarChart categories={["scan37", "scan55", "scan63"]} series={[{ name: "有限根处 V 的中位数", data: [0.6994963, 0.7274761, 0.6783381], tone: "neutral" }]} height={220} />
          <Text tone="secondary" size="small">来源：results/median_cpu_scan37/55/63.json，2026-10-12；每场景有限根分别 886/890/845。尚无 GT 或因果验证。</Text>
        </Stack>
        <Stack gap={8}>
          <H2>区间策略与可见性策略分别检查</H2>
          <Table headers={["场景", "有限根数", "±0.4 未括根", "final T 门控拒绝"]} rows={[["scan37", "886", "4", "2"], ["scan55", "890", "3", "1"], ["scan63", "845", "2", "4"]]} />
          <Text tone="secondary" size="small">后两类可能重叠，不能相加。深度恢复得更多不自动意味着质量更好。单位为射线数量，深度区间为归一化场景单位。</Text>
        </Stack>
      </Grid>
      <Text>四项待区分机制：NCC 导数偏差、透射率与空隙率差距、低斜率优化敏感性、括区可靠性。编号不表示排名。竞争解释与停止条件已写入 HYPOTHESES_AND_EVIDENCE.md。</Text>
      <Text tone="secondary">已有文献约束：Gaussian Wrapping、Unbiased 2DGS、TSPE-GS、Triangle Splatting+ 均与部分候选方向重叠。定向检索未建立新颖性证明；检索边界已保存。</Text>
    </Stack>}
    {tab === 4 && <Stack gap={18}>
      <H2>局部端点复核：减少计算量代理，GPU 迁移结果见独立对照页</H2>
      <Text>新版本允许区间内 Newton 步，并在接近根时直接复核小区间两端，避免继续收缩远处端点。每四个连续 Newton 步后强制二分。括区和 Newton 本身不作原创声明。</Text>
      <Table headers={["场景", "有限根数", "新：初始中点", "新：原初始化", "二分"]} rows={[["scan37", "886", "7", "7", "15"], ["scan55", "890", "6", "7", "14"], ["scan63", "845", "6", "7", "15"]]} />
      <Text tone="secondary" size="small">单位：每条射线的普通场函数评估次数，中位数；tol=10⁻⁶ 归一化射线距离。来源：results/median_newton_frozen_cpu.json，2026-10-12。CPU 仿射前向抽样核对已通过；函数评估次数不是 GPU 遍历数或耗时。</Text>
      <Text>与更严格旧参考根的最大差为 5.0081×10⁻⁷。每种初始化仅在 scan55 触发两次 80 位复核。分位数排序、导数、高精度成本和 GPU 同步/分支均未计入。</Text>
      <H2>数值失败保留并单独处理</H2>
      <Text>α=0.5000001 的极平坦尾部使初版 float64 端点判断失效，旧方案也受影响。失败参数已保存；当前版本对舍入尺度内残差作 80 位复核。高精度数值复核不是正式区间算术证明。</Text>
      <Text>同一 296 个随机混合、tol=10⁻⁹：新普通评估中位数 12，旧混合版 33，二分 32；另有 91 次高精度评估。包含附加边界例共 305 例通过。不能跨误差标准直接比较次数。</Text>
      <Text tone="secondary" size="small">来源：REVISED_SOLVER.md、results/median_newton_verification.json、results/median_newton_failure_v1.json。英文技术说明 median_depth_note.tex 已由内置编译器编译通过，尚待外部数学审阅。</Text>
    </Stack>}
    {tab === 5 && <Stack gap={18}>
      <H2>scan37 完整 NCC pilot：几何未达采纳门槛</H2>
      <Text>各 30k；训练种子 0；no_dn + trunc4。距离越低越好，图像指标越高越好。</Text>
      <Table headers={["指标", "legacy", "exact", "exact−legacy"]} rows={[["overall（mm）", "0.465459", "0.476716", "+0.011257"], ["d2s（mm）", "0.454697", "0.449326", "-0.005371"], ["s2d（mm）", "0.476220", "0.504106", "+0.027885"], ["PSNR（dB，训练视角）", "31.654657", "31.809812", "+0.155155"], ["SSIM（训练视角）", "0.966697", "0.967028", "+0.000332"]]} />
      <Text>三个预定 evaluator seed 的平均 Δoverall 为 +0.01119016 mm，范围 +0.01110359～+0.01125731 mm，均退化；legacy seed 0 重复得到相同数值。评估重复不能冒充训练重复。</Text>
      <Text>首个 NCC 调用的输入块数为 333,162 与 325,858。在任何 NCC 反传前状态已不同，所以相同种子没有提供相同 warmup；当前对照不能严格归因于导数修正。</Text>
      <Text>保留导数正确性修复；暂不扩大 NCC 训练。没有新视角、统计显著性或训练加速结论。</Text>
      <Text tone="secondary" size="small">来源：NCC_PILOT_RESULT.md、results/exact37_report.json、exact37_v1/postprocess_v2、seeded_eval_v2；2026-10-12。相同完整 49 个训练相机，未留出测试相机。</Text>
    </Stack>}
    {tab === 6 && <Stack gap={18}>
      <H2>GPU 实测：当前 Newton 原型更慢，四轮控制约快 14%</H2>
      <Text>冻结旧模型，每场景预定 8 个视角。5 轮 × 20 次 CUDA event 计时，预热后随机化变体和 depth on/off 顺序；下列为每视角轮次中位数再跨视角求均值。</Text>
      <Text tone="secondary" size="small">纵轴：带深度渲染 GPU 时间（ms）；横轴：DTU 场景。图例为不同隔离后端。</Text>
      <BarChart categories={["scan37", "scan55", "scan63"]} series={[{"name": "同源五轮控制", "data": [19.75827579498291, 13.630222988128661, 10.977253532409668], "tone": "neutral"}, {"name": "四轮控制", "data": [17.032740783691406, 11.747287273406982, 9.386029624938965], "tone": "info"}, {"name": "Newton hybrid", "data": [24.80656337738037, 16.691682434082033, 13.776366996765137], "tone": "neutral"}]} height={260} valueSuffix=" ms" />
      <Table headers={["场景", "五轮（ms）", "四轮（ms）", "hybrid（ms）", "四轮延迟降低中位数"]} rows={[["scan37", "19.758", "17.033", "24.807", "13.8%"], ["scan55", "13.630", "11.747", "16.692", "13.7%"], ["scan63", "10.977", "9.386", "13.776", "14.5%"]]} />
      <Text>所有 24 个完整图像的有效性掩码一致，RGB、alpha、法向最大差为 0。同源 JIT 与已安装五轮后端的深度逐位一致。四轮相对五轮的最大深度差约 5.866×10⁻⁵（归一化相机深度单位），抽样 CPU 误差均在 10⁻⁴ 内。</Text>
      <Text>hybrid 延迟增加约 22%～25%，拒绝当前效率实现。四轮为简单控制，不能作为新颖性声明；三场景冻结网格的 Δoverall 为 −0.00001495、+0.00009067、−0.00009111 mm，均通过预定工程容差。六视角反传均有限，但梯度 L2 漂移最高 1.396%，24 组中 19 组超出 0.1% 门槛；同后端重复漂移最多约 0.000235%。因此仅保留冻结网格提取候选，训练维持五轮默认。此处没有训练吞吐结论。网格来源：experiments/solver_mesh_v1/paired_mesh_metrics.json；梯度来源：results/solver_backward_repeat_v1.json。</Text>
      <Text tone="secondary" size="small">来源：results/median_cuda_frozen_v1.json、median_cuda_summary_v1.json、CUDA_SOLVER_RESULTS.md；RTX 5090 Laptop，2026-10-12。GPU 端点检查非严格区间证书，hybrid 斜率通道与历史 secant 有差异。</Text>
    </Stack>}
    {tab === 7 && <Stack gap={18}>
      <H2>更少前向细分，在已有反向遍历中局部精化</H2>
      <Text>前向保持普通四轮。反向额外累积 T、R′、R″，在光滑局部使用 Halley 步，并以二次展开近似隐式分母；没有新增基元遍历，但增加算术和寄存器。</Text>
      <Table headers={["变体", "最大参数梯度相对 L2 漂移", "通过 0.1% 工程门槛"]} rows={[["普通四轮", "1.39629%", "5 / 24"], ["逆二次插值", "4.63360%", "14 / 24"], ["Newton 反向精化", "1.37980%", "21 / 24"], ["Halley + 二次分母", "0.08062%", "24 / 24"]]} />
      <Text>固定 3 场景 × 2 视角 × 4 参数组，使用相同有效像素与 Rademacher 权重；对照是同源五轮。全部梯度有限。该表属于开发集检查，不能证明导数精确或训练质量。</Text>
      <H2>冻结完整图像前向加反向时间</H2>
      <Text size="small" tone="secondary">横轴：DTU 场景；纵轴：GPU 时间（ms）。每视角 5 轮 × 10 次，每轮随机变体顺序；先取每视角轮次中位数，再平均同场景 2 个视角。</Text>
      <BarChart categories={["scan37", "scan55", "scan63"]} series={[{"name": "同源五轮", "data": [33.1671100616455, 23.931235313415527, 18.488884735107423], "tone": "neutral"}, {"name": "普通四轮", "data": [30.410043048858643, 21.972640132904054, 16.817544078826906], "tone": "neutral"}, {"name": "高阶反向四轮", "data": [30.431774520874022, 22.35792636871338, 16.93503370285034], "tone": "info"}]} height={260} valueSuffix=" ms" />
      <Text>损失为 RGB 均值 + 0.01 × 完整中位深度均值。高阶变体的六视角延迟降低约 6.5%～8.7%；不含优化器、密化或 NCC，不能当作完整训练加速。</Text>
      <H2>完整 scan55 对照：已结束，几何门槛失败</H2>
      <Text>两臂共享核验过的 6999 步完整状态，再训练至 30k；统一五轮提取，固定 evaluator seeds 0/1/2。平均 Δoverall +0.000424087 mm、Δs2d +0.000908171 mm，训练视角 PSNR +0.066834 dB。原采纳门槛失败。相同状态五轮控制重复 Δoverall +0.001001144 mm；不选取较有利的重复基线翻转判定。没有独立训练种子或新视角结论。</Text>
      <Text>修正恢复顺序，使优化器绑定恢复后的外观参数。真实 PGSR 检查点复现原实现外观参数不更新；隔离修正恢复了更新。本次两臂都使用相同修正，历史从头训练不受此问题影响。</Text>
      <Text>数学：在峰值分离、简单根、所需导数有界时，底层 Jacobian 可有二阶/三阶局部误差；粗前向 upstream 仍可能造成一阶损失梯度误差。已开技术说明补入条件性证明并通过内置编译。Newton、Halley、IQI 及隐式微分不作原创声明。</Text>
      <Text size="small" tone="secondary">来源：results/iqi_backward_v1.json、adjoint_backward_v1.json、adjquad_backward_v1.json、adjquad_step_timing_v1.json、checkpoint_restore_v1.json；experiments/adjquad_smoke55_v1/manifest.json、adjquad55_v1/manifest.json；2026-10-12。此页为静态快照。</Text>
    </Stack>}
    {tab === 8 && <Stack gap={18}>
      <H2>实际早期训练：目标深度采样是主要设备活动</H2>
      <Text size="small" tone="secondary">横轴：实际 CUDA 活动分组；纵轴：各组时长占实际活动时长之和（%）。来源：profile55_trace_summary_v2.json；scan55 迭代 7005–7014。排除合成 profiler annotation，不是墙钟关键路径占比。</Text>
      <BarChart categories={["目标采样", "渲染前后向", "其余内核", "预处理", "内存操作", "NCC Jacobian"]} series={[{name:"实际设备活动时长占比",data:[38.158370347,29.383065557,29.017597051,2.528302020,0.554837948,0.357827077],tone:"info"}]} height={250} valueSuffix="%" />
      <H2>只改查询分组：六开发视角冻结计时有收益</H2>
      <Table headers={["场景/相机", "采样前后向 / 控制", "完整损失反传 / 控制"]} rows={[["scan37/0000", "0.903649", "0.938638"], ["scan37/0027", "0.895988", "0.943297"], ["scan55/0000", "0.876085", "0.943031"], ["scan55/0027", "0.858896", "0.914905"], ["scan63/0000", "0.900036", "0.960482"], ["scan63/0027", "0.909299", "0.918570"]]} />
      <Text size="small" tone="secondary">五轮求根保持；batch1 / 原 batch2。每视角五轮 × 十次，报告轮次中位数比。完整路径包括 RGB、PatchMatch、NCC 及 backward，不含 Adam、密化或外观更新。来源：sampling_batch_v2/results.json。</Text>
      <H2>追加完整损失梯度未通过，暂不进入训练</H2>
      <Text>其余 18 个视角的 RGB/NCC/几何/总损失逐位相同，但 12/162 参数梯度组超过 0.1% 门槛，最大 0.944%；同后端重复也有 11 组超门槛，最大 0.572%。六视角稀疏采样损失的通过不能替代完整损失检查。原门槛和失败记录保留。</Text>
      <Text>同一前向图五次反传：scan55/0013 的几何尺度梯度控制重放差约 31.6%，独立采样尺度梯度约 16.6%，查询点梯度则逐位稳定。scan63/0048 的 RGB 尺度梯度重放差约 1.05%。这些是失败后选择的机制诊断相机，不能作留出质量证据。</Text>
      <Text>直接逆因子局部修正不足以稳定完整反传。仅提高世界参数 backward 预处理中间精度后，18 视角、162 组梯度全部通过；选择性 FP64 也通过。但它们共享解析公式和 float32 基元累加，重复性不是世界导数真值。局部独立 CUDA 世界参数 VJP 参照已完成，详见新增页面；完整渲染尚未独立核验。</Text>
      <Table headers={["追加 18 视角的完整损失梯度", "最大相对 L2", "0.1% 门槛"]} rows={[["原 float32 分组候选 / 原控制", "0.943811%", "失败：12/162"], ["FP64 batch1 / FP64 batch2", "0.003331%", "通过：162/162"], ["轴比 >10 使用 FP64 / FP64 batch2", "0.010293%", "通过：162/162"]]} />
      <H2>三对交替实际训练：没有通过加速门槛</H2>
      <Table headers={["轮次", "执行先后", "控制 block（s）", "混合精度 block（s）", "候选 / 控制"]} rows={[["1", "control → mixed1", "32.224005", "32.284658", "1.001882"], ["2", "mixed1 → control", "31.975423", "30.345740", "0.949033"], ["3", "control → mixed1", "30.433321", "31.140537", "1.023238"]]} />
      <Text>每进程同一完整 6999 步状态，真实更新 500 步，前 50 步预热，计时 7050–7499；含实际密化、滤波和 Adam。墙钟比几何均值 0.990889，未达到预设 ≤0.97，且第三对超过 ≤1.01。拒绝效率采纳；这三对进程不是独立训练种子或完整质量实验。六进程均已结束。</Text>
      <H2>独立解析问题：法向归一化漏掉长度因子</H2>
      <Text>相对 forward 的局部 VJP 缺少原始相机法向模长的倒数。正定例的正确导数 −1/(2√2)，现有局部式 −1/2；符号、自动微分和方向有限差分通过。进一步化简得到 n=−Pd/‖Pd‖、原始长度=1/|nᵀd|，并通过 64 个独立世界 VJP 图检查。局部 CUDA 世界导数现已核验，完整训练质量已评估且未过门槛；不解释法向上游为零的几何损失波动。</Text>
      <Text size="small" tone="secondary">来源：world_precision_report_v1.json、mixed_training_timing_report_v1.json、normal_preprocess_vjp_v1.json、affine_normal_identity_v1.json；2026-10-12。效应只在各自测试口径成立，不作统计显著性或新颖性声明。</Text>
      <Text>峰值自适应求根另一路线已结束：24 组梯度通过，但六视角 RGB+depth 前向/反向慢 0.42%–1.15%，拒绝当前效率实现。</Text>
      <Text size="small" tone="secondary">来源：sampling_development_report_v1.json、sampling_gradient_diagnostic_v1、inverse_gradient_diagnostic_v1、peak_development_v1；2026-10-12。研究说明 SAMPLING_AND_STABILITY.md。科研工具溯源 Kassis et al., Scientific Agent Skills, DOI 10.48550/arXiv.2609.00065。</Text>
    </Stack>}
    {tab === 9 && <Stack gap={18}>
      <H2>从常规通过到极端失败：相同门槛，全部输入保留</H2>
      <Text>原独立参照：72 个真实模型与 48 合成基元，公式修正 FP64 共 2784 参数块通过。新压力集按轴比预选 44 基元，最大轴比 34819，含投影范围内外；原 FP64 往返路径在法向、平面、射线精度均失败。先前常规结果不足以代表尾部。</Text>
      <Table headers={["压力集输出上游", "原修正 FP64 最大归一化误差", "直接因子 FP64 最大归一化误差", "失败参数块：旧 → 新"]} rows={[["法向", ".0329352", "6.008e-6", "12 → 0 / 176"], ["平面斜率", ".00317795", "3.399e-6", "3 → 0 / 176"], ["射线精度", "47.2941", "2.089e-6", "28 → 0 / 176"], ["投影 conic", "1.678e-5", "1.678e-5", "0 → 0 / 176"], ["opacity 系数", "4.649e-8", "4.649e-8", "0 → 0 / 176"]]} />
      <Text size="small" tone="secondary">来源：world_preprocess_stress_v1、factor_preprocess_stress_v1，2026-10-12。误差为 ‖gCUDA−g物理AD‖/max(‖g物理AD‖,1e−5)，门槛 .001；含结构零，非统计样本。旧最坏射线梯度真模长 .364575，非结构零。</Text>
      <H2>数学机制：正交残差被轴比平方放大的上界</H2>
      <Text>Σ=RD²Rᵀ，P=RD⁻²Rᵀ。若 δ=‖RᵀR−I‖₂、a=smax/smin，则 ‖PΣ−I‖₂≤δ[1+(1+δ)a²]。标准四元数多项式满足 δ=4|‖q‖²−1|‖v‖²；float32 规范化后再升精度不会自动恢复精确单位范数。保存样本最大 PΣ−I 谱范数 28.0014。</Text>
      <Text>直接对 P 的因子图求导，不依赖 R 严格正交；仍需 raw quaternion 规范化 pullback。9 个精确符号分量、164 样本上界检查、64 非正交 AD 例通过。上界是充分条件，不是逐基元误差预测、完整梯度误差界或原创矩阵微分声明。</Text>
      <H2>新组合路径的局部入选检查</H2>
      <Table headers={["固定保存输入", "参数块数", "新路径最大归一化误差", "原 .001 门槛"]} rows={[["44 极端基元", "880", "1.6782e-5", "全部通过"], ["72 真实 + 48 合成", "2784", "1.19672e-4", "全部通过"]]} />
      <Text>实际源码和生成源码归档已逐字节复核。物理 float64 图是入选参照，device q 值图仅帮助定位舍入。新路径不改 forward，原压力集四个前向分量失败仍存在；局部 VJP 不保证完整渲染、硬掩码或训练质量。</Text>
      <H2>独立前向变体：保存的四个 conic 失败降为零</H2>
      <Text>投影因子两列 a,b 的 Gram 行列式等于 ‖a×b‖²；加 kernel k 后再加 k(‖a‖²+‖b‖²)+k²。这避免最后一次大数乘积相减，但仍可能有叉积内部抵消、舍入和上下溢。精确符号恒等式通过；这是标准代数，不声称新公式。</Text>
      <Table headers={["全部164保存输入的conic前向", "最大归一化误差", "超原1e-4门槛个数"]} rows={[["原float32行列式", ".00290671", "4 / 164"], ["Gram float32行列式", ".0000276635", "0 / 164"]]} />
      <Text>六输出块共984检查通过，射线与法向逐位不变。新完整 Gram+factor 后端20次Adam通过，初态一致、梯度有限、首损失最大归一化漂移1.2822e-5，低于预定1e-4。已完成的 factor64 质量候选保留原前向；Gram 未启动完整训练，不能混用其结果或宣称图像/几何/速度收益。Null-scale fallback未测试。</Text>
      <H2>三个公式补充与可积性反例</H2>
      <Text>法向 VJP 应除以原向量模长；κ=√(rᵀPr/L) 的视线导数须包含分子项；η=√(D₀/D₁) 的 D₁ 导数须含 η²。旧 κ viewport 场存在正定旋度反例 √10/45，各向同性旧中心场产生人工 κ log cosθ 项。仅对固定输出上游成立，不证明完整训练没有替代目标。</Text>
      <H2>完整scan55结果：原几何门槛保留失败判定</H2>
      <Text>旧 corrected64 在压力失败后未启动。旧父队列被暂停，控制子进程继续到 30k；结束父队列并保留原 manifest 后，明确记录新协议。固定复用质量观测前预选的 legacy64 控制，后续不换基线。新 factor64 完整后端三个局部函数与探针逐字节一致；20 次 Adam 更新的初态、首相机和首损失一致，梯度有限。</Text>
      <Table headers={["指标", "legacy64", "factor64", "候选减控制"]} rows={[["overall（mm，三评估种子均值）", ".315025878", ".313129859", "−.001896019（改善.601861%）"], ["s2d（mm）", ".401591461", ".398632827", "−.002958633"], ["PSNR（dB，49 TRAIN视角）", "34.687950", "35.220003", "+.532053"], ["SSIM（49 TRAIN视角）", ".96126704", ".96439554", "+.00312850"], ["高斯基元数", "943641", "967292", "+23651（+2.50636%）"]]} />
      <Text>14阶段均exit 0。overall改善未到原≥1%，因此pilot失败；s2d和TRAIN PSNR子门槛通过，不能改阈值翻转总判定。35/49训练视角PSNR提高、14下降；相关视角不是独立训练重复。组合干预需消融，容量增加可能参与图像收益；无留出泛化、统计显著性或加速结论。</Text>
      <H2>全新42/7留出实验：两项RGB门槛失败</H2>
      <Table headers={["指标", "legacy64", "factor64", "预设判定"]} rows={[["TEST PSNR（dB，7视角）", "28.032109", "28.122184", "+.090075＜+.30，失败"], ["TEST SSIM（7视角）", ".89905294", ".89874837", "−.00030457＜0，失败"], ["overall（mm，三采样均值）", ".332269708", ".331238519", "比例.9968965≤1.01，守护通过"], ["基元数", "929179", "957753", "+3.07519%≤10%，守护通过"], ["TRAIN PSNR（dB，42视角）", "35.528275", "35.949982", "+.421708；不是TEST"]]} />
      <Text>19阶段全部exit0，484源码、32生成文件、实际模型、GT与每图结果已复核。5/7测试视角PSNR提高、2下降；单对训练不能给出显著性。原几何pilot和新RGB pilot均保留失败，不自动扩大此RGB训练。</Text>
      <Text>全新42训练/7测试warmup及实际训练/NCC/filter池已核验；旧全49检查点未复用。SfM可能使用全视角，scan55先前已用于选题。新配准按图像ID映射中心RMS .0000765214mm；旧firstN会错到380.593721mm。旧49视角结果保持。</Text>
      <H2>尺度VJP的精确误差与可达紧界</H2>
      <Text>在固定R、正尺度和相同对称精度上游H下，误差为−2⟨H,Bᵢ⟩；Bᵢ=vᵢvᵢᵀ−uᵢuᵢᵀ，Mᵢⱼ=⟨Bᵢ,Bⱼ⟩。单位Frobenius上游的最大向量误差恰为2√λmax(M)，可构造上游达到。固定非单位四元数族的单分量相对误差按δ²a⁴增长，归一化极限1/4。</Text>
      <Text>精确SymPy与64独立AD通过，164保存输入保留。H=I下保存样本整体向量最大相对误差仅7.045e−7，单分量比785.079；不可混用。最坏上游并非实际训练上游；不是完整渲染器梯度或质量定理。早期失败源码/日志均归档。</Text>
      <Text>停顿诊断：上一队列的 running 状态滞留于被暂停的父进程，训练子进程实际已完成。新留出队列自动完成19阶段并退出。最近 app 摘要有一轮 interrupted，但未给出原因；不能认定平台故障或 GPU 死锁。新只读检查器核对实际 argv、进程状态和日志写入时间，不因 manifest 旧而重启训练。</Text>
      <Text size="small" tone="secondary">来源：factor_audit_report_v1.json、precision_roundtrip_defect_v1.json、training_stall_diagnosis_v1.json、factor_quality55_report_v1.json、factor_holdout55_quality_report_v1.json、precision_scale_vjp_defect_v4.json；2026-10-13 完整factor与留出结果静态快照，非实时监控。另见 gram_forward_report_v1.json、gram_smoke55_v1；LaTeX 现含Gram恒等式，最新内置编译v15通过。</Text>
    </Stack>}
    <Text size="small" tone="tertiary">所有数字均为本地结果快照。完整公式、限制、脚本与原始 JSON 位于 unified_geometry_gs/research；本视图不自动更新训练进度。</Text>
    {tab === 10 && <Stack gap={18}>
      <H2>深度收敛不保证梯度收敛</H2>
      <Text>严格平台跨越使薄层中位深度趋于所选峰位；位置Jacobian趋于one-hot还需避开内部零斜率峰值。三层反例每个有限厚度的根均正则，根恒为0，但位置权重恒为(1/2,0,1/2)。</Text>
      <Table headers={["理想射线结论", "核验证据", "边界"]} rows={[
        ["严格跨越 + 非临界裕量", "缩放根和位置Jacobian指数尾界；20个预定域案例", "固定有效opacity、峰位与width，不等同完整2DGS/TGS"],
        ["临界三层族", "7个200位案例、精确符号；opacity导数指数发散", "不是实际训练遇到该对称族的频率"],
        ["平台等号", "6个案例，根始终为宽度加权中间点", "离散等号约定不能自动由薄化得到"],
        ["分位区间C1泛函", "空间积分梯度不除T斜率；位置导数非负/和1", "已有统计量，应用新颖性尚未确认"],
        ["表面混合代价", "跨平台区间极限为峰位加权均值", "不能据梯度有界直接宣称重建更好"]
      ]} />
      <H2>同一临界族，两种深度目标</H2>
      <Table headers={["epsilon=.075", "单个q=.5中位深度", "区间[.45,.55]均值"]} rows={[
        ["深度（任意射线单位）", "0", "0.000459138855"],
        ["中层opacity偏导", "−1.44629e37", "−0.532260208"],
        ["位置Jacobian", "(.5,0,.5)", "约(0,1,0)"],
        ["真实质量/速度结论", "无", "无"]
      ]} />
      <Text>区间梯度不等于原中位根梯度。7个临界族、1个单层奇点、12个平台/区间例已计算。v1 double差分未分辨小尾部，源码与日志封存；v2用60位重新积分，保留原1e−5/floor1e−8门槛，最大误差2.651e−11。</Text>
      <Text>非对称区间[.40,.55]在两层(−1,+1)上的极限为+1/3；预案手算−1/3为符号错误，原案和独立勘误保留。未以该错误预测更换参数或删掉观测。</Text>
      <H2>真实冻结射线：统一替代门槛失败</H2>
      <Table headers={["原3072条均匀采样射线", "实际观测", "解释"]} rows={[
        ["q=.5有限 / 全区间有限", "2621 / 2614", "7条只有有限单根；全部输入保留"],
        ["主数值检查 / 独立前向与差分", "2614通过 / 24通过", "CPU参考，不是完整CUDA梯度"],
        ["偏移 > .1 median(width)", "204/2614 = 7.80413%", "scan37自身11.4253%；无GT质量解释"],
        ["假想薄层混合代理 > .1", "2051/2614 = 78.4621%", "超过预设10%守护，统一替代失败"],
        ["opacity范数 band/median", "p50=1.01412；max=267.366", "不能保证逐射线缩小梯度"]
      ]} />
      <Text>区间正则性定理成立，但固定宽度统一区间不宜直接推进训练。hypothetical thin mixing是模型代理，不是真实表面错误率。额外极端q空间全VJP积分未收敛，失败源码/日志保留；相同两个极端样本的独立product-T全分量前向差分通过，最大位置/opacity相对误差6.50e−8/1.593e−10；13源/archive与全部输入行及汇总复核通过。此成功不等于原q空间积分已修复。</Text>
      <Text size="small" tone="secondary">来源：THIN_QUANTILE_LIMIT.md、thin_quantile_limit_v2.json、QUANTILE_BAND_RESULT.md、quantile_band_v2.json、QUANTILE_BAND_PROTOCOL_ERRATUM.md；2026-10-13。PNG/SVG导出与来源清单见figures/thin_quantile_evidence_v1。当前LaTeX原位编译v15通过，目标仍active；真实冻结射线偏差诊断已完成且未过统一替代门槛，不启动训练。</Text>
    </Stack>}
  </Stack>;
}




