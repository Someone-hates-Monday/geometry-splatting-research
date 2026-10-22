import { useState, useHostTheme, Stack, Row, Grid, H1, H2, Text, Table, BarChart, LineChart } from "cursor/canvas";

const tabs = ["现有证据", "数学发现", "实验门槛", "CPU 射线诊断", "求根原型", "完整 NCC pilot", "GPU 求根对照", "高阶反向精化", "采样与反传稳定性", "世界参数导数", "薄层与区间深度", "方向兼容性", "匹配因素与透视系数", "裁剪扩展与精度负例", "精度候选与CUDA核验", "完整图像门槛", "全部异常根诊断", "有效alpha与世界参照", "一致因子与根别名", "条件精度根证书", "严格尾部CUDA组件", "设备预算与自主根", "精确倒数表优化", "短证书与回退", "完整图像名单容量", "动态CSR实际名单", "共享几何缓存"];
export default function ResearchAudit() {
  const theme = useHostTheme();
  const [tab, setTab] = useState(26);
  const tabStyle = (active: boolean) => ({ padding: "8px 14px", cursor: "pointer", border: `1px solid ${theme.stroke.tertiary}`, borderRadius: 5, background: active ? theme.fill.secondary : theme.bg.editor, color: active ? theme.accent.primary : theme.text.secondary });
  return <Stack gap={22} style={{ padding: 24, maxWidth: 1120, margin: "0 auto", color: theme.text.primary }}>
    <Stack gap={6}>
      <Text size="small" tone="secondary">3DGS / GGGS / 2DGS / 三角形 TGS · 2026-10-13 审查快照</Text>
      <H1>共享几何缓存保持全部固定输出位；真实射线解码与认证待接入</H1>
      <Text>factor64全49视角几何pilot和42/7留出RGB对照均已完成，分别未过预设几何、TEST PSNR与SSIM门槛。完整24视角、10834488像素的三臂比较已完成：8视角最大深度差门槛失败，共9像素超阈值；packet/factor完整forward时间比2.60018。全9点的18个保存系数模型均有有限根，CUDA逐位重放通过；FP64稳定log-T仍保留大差异。全72混合根已定位有效alpha差；MP80/140世界参照与共同opacity参照完成。一致QR主候选真实9点改进，但12局部构造/5临界根失败；合法F32世界在正确系数舍入后别名，接口统一误差下界超过.488。条件精度CPU全7010区间与106有限根/4无根通过，30根触发回退；严格尾部核心CUDA212端点通过，局部event约223ms；设备7322预算/110mass/自主106根与4无根已通过，合并event约706ms；完整renderer/backward/GT与效率待验证，无新训练扩展。</Text>
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
    {tab === 11 && <Stack gap={18}>
      <H2>固定点：何时能用单个法向表示所有方向衰减</H2>
      <Text>K(w)=Σ|w·bᵢ|，bᵢ=∇log√(1−Gᵢ)。存在任意b使K(w)=|w·b|对所有单位方向成立，当且仅当所有非零bᵢ共线；若指定b=Σbᵢ，即product-v梯度，还必须同向。互易性本身不够，点态兼容也不保证全局可积。</Text>
      <Text>E=inf_b sup_unit_w |K(w)−|w·b|| ≥ √λ₂(Σbᵢbᵢᵀ)。两正交梯度的合法Gaussian构造达到下界。任意两项的√(cross²/trace)仍是全模型绝对下界，添加其他Gaussian不会削弱它。</Text>
      <Table headers={["证据", "已核验", "范围"]} rows={[
        ["数学检查", "精确SymPy + 64预定数值见证", "完整解析证明给出全方向结论"],
        ["真实参数查询", "24/24正区间下界；3完整模型", "8保存视角/scan；不是GT表面点"],
        ["原始来源", "13源/archive、26产物、3PLY、48raw基元", "24点按原相机/射线精确回投影"],
        ["独立复核", "24有理数不等式；144空间导数分量", "120位标量求导落入80/100位区间"],
        ["名义归一化下界", "min .9000%；median 6.7901%；max 21.3081%", "非区间相对界；非像素/几何错误率"]
      ]} />
      <H2>全部24点：保存绝对证书与名义方向统计</H2>
      <Table headers={["查询点 scan / camera", "绝对下界（世界单位⁻¹）", "名义谱下界 / Σ||bᵢ||（%）", "原相机product衰减差（%）"]} rows={[["scan37 / 0000", "51.9341", "13.6485", "41.2816"], ["scan37 / 0006", "18.1252", "3.1486", "10.5696"], ["scan37 / 0013", "2.73588", "1.1190", "0.5264"], ["scan37 / 0020", "16.5517", "3.0832", "80.5527"], ["scan37 / 0027", "19.2206", "4.9172", "1.7086"], ["scan37 / 0034", "1.06264", "12.2169", "9.5279"], ["scan37 / 0041", "3.41933e-10", "16.6744", "96.1674"], ["scan37 / 0048", "5.07494", "5.7863", "0.2019"], ["scan55 / 0000", "12.1822", "21.3081", "87.0191"], ["scan55 / 0006", "26.4606", "7.4904", "14.7920"], ["scan55 / 0013", "4.66606", "6.0378", "31.0504"], ["scan55 / 0020", "25.247", "6.0898", "46.5387"], ["scan55 / 0027", "28.3339", "13.3984", "53.8202"], ["scan55 / 0034", "24.6105", "10.0924", "79.4578"], ["scan55 / 0041", "28.4777", "7.8825", "79.2328"], ["scan55 / 0048", "41.2536", "13.7484", "69.9744"], ["scan63 / 0000", "0.029322", "0.9000", "6.5354"], ["scan63 / 0006", "7.25442", "16.9159", "40.7277"], ["scan63 / 0013", "5.79478", "1.8656", "4.2712"], ["scan63 / 0020", "19.8287", "16.1570", "95.2194"], ["scan63 / 0027", "7.52571", "2.6379", "25.0247"], ["scan63 / 0034", "0.054919", "12.5586", "53.1038"], ["scan63 / 0041", "5.4788", "3.1561", "80.1219"], ["scan63 / 0048", "6.3706", "4.5560", "74.7165"]]} />
      <Text size="small" tone="secondary">来源：directional_field_world_v1全部point_00..23.json、directional_field_world_report_v1.json；2026-10-13。绝对下界由双基元区间支持；后两列为完整double测量。最小证书3.4193e−10，非零不等于可见效应。所有名义指数underflow计数保留。</Text>
      <H2>同24条射线：方向衰减差不自动对应中位深度位移</H2>
      <Text>完整世界Gaussian限制到射线后，用T_camera(t)=T_minus(t)/T_minus(0)求相机中位根。24/24有限；点态logV与射线图最大差1.98e−13，attenuation最大比例差8.47e−13；根log residual最大3.04e−11。无严格区间根界。</Text>
      <Table headers={["完整世界模型", "绝对深度位移/旧深度 中位（%）", "最大（%）", "旧affine根处 world T_camera中位"]} rows={[["scan37", "0.008766", "2.694594", "0.484583"], ["scan55", "0.011509", "0.062343", "0.479000"], ["scan63", "0.052613", "0.343859", "0.518242"]]} />
      <Text>全部24点的相对位移中位.0126254%。最大例scan37/camera0041/pixel(601,131)：6.089402428→6.253487084，+2.69459%；世界坐标单位，未换算mm。保留全部点与尾部。</Text>
      <Text size="small" tone="secondary">来源：world_affine_rays_v1全部24行、world_affine_rays_report_v1.json；2026-10-13。40源/archive、26产物、3PLY与全部算术重放通过，但未独立重算所有根。比较同时改变affine形状、near-plane/tile/alpha/prefix选择及clamp，无法单独归因为透视近似。没有GT或质量收益结论。</Text>
      <H2>文献重叠与下一步</H2>
      <Text>GGGS近侧条件、Gaussian Wrapping方向问题与Objects as Volumes一般法向分布均为已有工作。K是zonotope支持函数，单法向是线段；Siegel2023与Torres2024已有一般逼近理论。因此不能把基础表示或泛泛多视图问题作为原创。本补充的新颖性尚未完成全面审查。</Text>
      <Text>当前世界场不等于生产per-view affine图；24点不是GT表面或世界模型精确median。下一步需区分affine误差、近侧条件和真实跨视图误差，不据此直接加入全模型单法向惩罚或训练。mpmath.iv仍实验性；高精度与有理数重放不等于证明助手认证。</Text>
      <Text size="small" tone="secondary">LaTeX当前原位v19编译成功。同射线检查亦complete、匹配进程为空。holdout健康v4与方向诊断健康v1均complete、匹配进程为空。最新线程记录确认两轮stream disconnected before completion；此前父队列暂停是另一种已定位情况。没有训练完成触发断流或CUDA死锁的证据。</Text>
    </Stack>}
    {tab === 12 && <Stack gap={18}>
      <H2>同24条射线：匹配基元和逐项规则</H2>
      <Text>24投影和24射线所有原CPU数组bitwise一致。固定原selected集合的全部8种peak/width/amplitude组合，192根全有限且独立重解通过。再依固定顺序解除筛选并改变起点，24条完整路径均有限，总和误差最大4.34e−19。</Text>
      <Table headers={["因素/固定路径增量", "定义样本数", "绝对增量/旧深度 中位（%）", "最大（%）"]} rows={[["peak排列平均", "24", "0.0020919448", "3.3621159"], ["width排列平均", "24", "3.8927276e-05", "0.0043369336"], ["amplitude排列平均", "24", "0.00082629761", "0.2914493"], ["取消world .99 cap", "24", "0", "0"], ["取消affine prefix stop", "24", "2.7303506e-13", "0.026229346"], ["取消power/alpha eligibility", "24", "0.011589381", "0.67636898"], ["取消像素tile", "24", "0.00035056598", "0.010465958"], ["取消projection/image extent", "24", "2.2480498e-07", "0.00096071737"], ["取消center near-plane", "24", "0", "0.040836139"], ["改为有限相机起点", "24", "3.7428074e-11", "0.0015452718"]]} />
      <Text size="small" tone="secondary">来源：matched_ray_factors_v1/summary.json、matched_ray_factors_report_v1.json；2026-10-13。前三项为标准六排列平均分配，其余为预定路径差，不是唯一物理/因果归因。69源/archive、50产物、3PLY、24membership复核；192固定集合根独立重解最大差2.69e−13，大集合阶段根未独立重解。</Text>
      <Text>中位相消比例29.1519%，最大87.0742%。最大尾部scan37/cam0041：peak +.204732767、取消eligibility −.041186829，完整net +.164084657个世界单位。不能把总差全归因某个因素；此处不是GT质量对照。</Text>
      <H2>条件性透视重参数：复用已有affine量</H2>
      <Text>未clip中心、screen kernel=0且P与Sigma互逆时，令g=q₀·δ/||q₀||²，f=||q₀+δ||²/||q₀||²，ell=g−Delta_A/t₀，chi=E_A(s₀/t₀)²，D=(1+ell)²+chi。精确世界ray peak=t₀√f(1+ell)/D，width=s₀√f/√D，alpha=amplitude·exp(−E_A/(2D))。</Text>
      <Text>Schur补给出conic能量身份；Delta_A是直接plane-offset线性项，E_A必须来自cap前conic。该表达是精确代数身份，仍可能继承conic舍入/坏条件，不保证FP32稳定或加速。</Text>
      <Table headers={["核验", "实际结果", "解释边界"]} rows={[
        ["符号与固定SPD", "9身份；64/64构造通过", "原1e−8/floor1e−12门槛"],
        ["真实selected输入", "1992保留；1946适用全部通过", "46 clipped center域外，不能算通过"],
        ["独立原PLY 80位oracle", "全部1946；max p/width/alpha=4.99e−15/1.66e−15/2.63e−12", "数值交叉核验，非严格区间/形式化认证"],
        ["实际来源与产物", "124源/archive、50产物、3PLY", "全体输入、域分类和模型SHA核验"]
      ]} />
      <Text>HTGS和3DGUT已有一般透视正确Gaussian评估，不能将精确ray概念作为原创。此packet形式是否有独立价值仍需新颖性与实现比较。下一步先一般化clip域，再做FP32/极端数值、CUDA forward/VJP、实际开销；通过后再做固定模型质量对照和决定训练。</Text>
      <Text size="small" tone="secondary">来源：PERSPECTIVE_REPARAM_THEORY.md、PERSPECTIVE_REPARAM_RESULT.md、perspective_reparam_theory_v1.json、perspective_reparam_report_v1.json；2026-10-13。当前LaTeX原位v18编译成功。无GPU、梯度、速度、GT、新视角或重建收益声明。</Text>
    </Stack>}
    {tab === 13 && <Stack gap={18}>
      <H2>参考中心可以被clip：完整二次型仍可恢复</H2>
      <Text>kernel0且Sigma=P⁻¹时，现有conic、plane两项、width提供参考基底中的Schur参数(b,c,K)。P在该基底的Gram为b[[K+ccᵀ,c],[cᵀ,1]]，六个自由度可唯一恢复；这是标准块分解的Gaussian应用，不是新的矩阵分解。</Text>
      <Text>令a=qray_xy−qbar_xy、m=qmean_xy−qbar_xy，A=1+c·a、B=1+c·m，den=A²+aᵀKa、cross=AB+aᵀKm。peak=z||qray||cross/den，width=||qray||/√(b·den)。能量用Gram非负分解det(K)(a×m)²+(Am−Ba)ᵀK(Am−Ba)计算，仍可能继承浮点det/cross敏感性。</Text>
      <Table headers={["核验范围", "已完成", "限制"]} rows={[
        ["符号/固定SPD", "6身份；128/128，其中112 clip", "一般精确算术身份，不保证浮点稳定"],
        ["同24射线真实输入", "1992/1992，包括原46 clip", "旧选择集合不变，无fallback"],
        ["独立原PLY80位", "全部1992；max p/s/alpha=4.72e−15/1.52e−15/2.61e−12", "180源/archive、50产物、3PLY核验"],
        ["raw FP64 AD图", "71712/71712分量通过；max4.19e−7", "1e−4/floor1e−8；共用raw变换与Torch AD"],
        ["真实FP32 packet", "1992/1992通过；p/s相对3.13e−7/2.71e−7，alpha绝对5.59e−6", "仅round double数据，不是实际FP32/CUDA预处理"],
        ["合成FP32开发守护", "127/128通过；case2峰位失败", "development_gate_pass=false，不能删例或放宽门槛"]
      ]} />
      <H2>保留的FP32失败：输入舍入与求值误差都存在</H2>
      <Table headers={["同一case2的计算方式", "峰位相对误差", "原1e−5门槛"]} rows={[["原double packet / 80位求值", "1.8514187e-12", "通过"], ["已round FP32 packet / 80位求值", "1.713951e-05", "失败"], ["已round FP32 packet / FP32求值", "5.2905905e-05", "失败"]]} />
      <Text>真peak≈−.02839817738，FP32≈−.02839967981。独立80位decode已经round过的packet仍超阈值，单提高求值精度不足。该例真alpha≈2.3005e−5712，double下为0，物理响应极小；仍保留原失败判定，不据此证明任意decoder不可能。</Text>
      <Text size="small" tone="secondary">来源：clipped_packet_theory_v1.json、clipped_packet_report_v1.json、clipped_packet_precision_v1/summary.json、clipped_packet_precision_report_v1.json；2026-10-13。233源/archive、51产物、所有行/梯度算术复核；失败例误差分离是事后分析。逐输入参数复制不测试共享基元atomic累加，非完整CUDA backward。</Text>
      <H2>下一项：输入精度保护与实际生产数据</H2>
      <Text>先处理输入舍入敏感性，任何新方法/门槛另立方案并保留本失败；再核验实际FP32预处理、CUDA forward/backward与资源/开销。当前不训练，无GT质量、速度或新颖性结论。LaTeX原位v19已编译通过，原文档保持打开。</Text>
    </Stack>}
    {tab === 14 && <Stack gap={18}>
      <H2>独立构造拒绝开发集通过的两种输入保护</H2>
      <Text>64种保护组合覆盖1992真实+128旧构造后，最小通过mask8只保护qmean，mask21保护plane/conic/camera_geometry。锁定后，以seed13017生成4096同分布新构造，两者均失败；不在看到结果后重新搜索组合。</Text>
      <Table headers={["固定方法", "失败数/输入数", "peak最大相对误差", "全部通过"]} rows={[["fp32", "21/4096", "0.001717234", "False"], ["all_rounded_double_decode", "11/4096", "0.0005474119", "False"], ["qmean_protected", "13/4096", "0.0004250883", "False"], ["other_minimal_protection", "11/4096", "0.0001862746", "False"], ["all_double", "0/4096", "5.397692e-11", "True"]]} />
      <Text>原门槛peak≤1e−5、width≤1e−4（floor1e−8）、alpha绝对≤1e−4。所有保护组合使用double求值，不是实际FP32预处理。选择阶段旧6360系数逐位一致；新失败并集24例独立80位因子参照保持全部分类。不是独立真实场景质量实验。</Text>
      <H2>实际生产CUDA操作：原1992真实输入局部通过</H2>
      <Table headers={["诊断扩展的decode臂", "失败数/输入数", "peak相对max", "width相对max", "alpha绝对max"]} rows={[["production_uv_fp64_decode", "0/1992", "4.952861e-07", "1.184963e-06", "1.181539e-05"], ["production_uv_numpy_fp32_decode", "0/1992", "5.538997e-07", "1.297824e-06", "1.181684e-05"], ["production_uv_cuda_fp32_decode", "0/1992", "5.538997e-07", "1.297824e-06", "1.181684e-05"], ["extra_camera_ratio_fp64_decode", "0/1992", "4.950003e-07", "8.807188e-07", "1.008607e-05"], ["archived_double_geometry_fp64_decode", "0/1992", "5.788039e-07", "8.782294e-07", "4.152963e-06"]]} />
      <Text>复制完整生产preprocessCUDA及其依赖函数直接启动；46clip保留、0cull。676源/archive、50产物、生成文件与二进制、3原PLY核验。5976 CUDA输出重放逐位一致，实际CUDA den/det/energy无效0。非已安装renderer二进制、完整反向或效率实验；两额外geometry臂不是仅有packet的部署方案。v1/v2环境编译失败日志保留。</Text>
      <H2>理论范围与下一步</H2>
      <Text>有限扰动界保留cross接近零的相对敏感性；5符号身份与2120数值界检查通过。同一binary32字节的两个合法SPD输入，其peak接受区间分离下界约4.92775e−7，仅约束直接round qmean的八输入表示。mpmath.iv是实验性可信依赖；不扩写成全部生产布局不可能。</Text>
      <Text>贡献审计确认IFT/exact ray/Schur工具已有明确重合。薄层两种裕量与梯度反例仍为理论候选，新颖性未证实；Triangle原论文技术核验尚缺。下一项完整CUDA/VJP与实际成本，然后冻结模型affine/exact packet/optimized direct factor三臂GT对照；未过门槛不扩大训练。</Text>
      <Text size="small" tone="secondary">来源：packet_input_protection_v1、packet_input_confirmation_v1、packet_input_protection_report_v1.json、packet_conditioning_v1.json、production_packet_v3、production_packet_report_v1.json；2026-10-13。原LaTeX原位v20编译成功；用户要求先简报，当前没有新训练。</Text>
    </Stack>}
    {tab === 15 && <Stack gap={18}>
      <H2>局部通过不足以代表完整图像：首版实现被拒绝</H2>
      <Text>三个既有开发场景各8预定视角，共10834488像素，affine/exact_packet/exact_factor全部保存。只替换median T内(p,rho,alpha)，legacy选择、RGB/normal、cap/prefix/排序/初值与五轮solver保持。sample/NCC/integrate/backward显式禁止，不是完整exact renderer。</Text>
      <Table headers={["完整24视角的检查", "结果", "原门槛判定"]} rows={[
        ["全数组有限；RGB/alpha/normal逐位保持", "全部通过", "因素隔离成立，非图像提升"],
        ["packet/factor有效mask", "24/24完全一致", "通过"],
        ["共同有效depth p99", "各视角≤1.90735e−6模型长度单位", "通过≤1e−4"],
        ["共同有效depth max", "8/24视角失败；9像素>1e−3，最大.1796405315", "整体数值门槛失败"],
        ["完整GPU render packet/factor", "GM2.6001822；max2.6573840", "失败：要求GM≤.90/max≤1.05"],
        ["同步wall packet/factor", "GM2.5999135", "无训练加速结论"]
      ]} />
      <H2>全部失败视角的最大异常位置</H2>
      <Table headers={["scan", "camera", "最大误差pixel(x,y)", "超阈值像素数", "depth最大绝对差", "depth p99绝对差"]} rows={[["37", "0020", "[30, 1]", "1", "0.0013184547", "7.15256e-07"], ["37", "0034", "[223, 276]", "1", "0.0089430809", "9.53674e-07"], ["55", "0000", "[161, 215]", "1", "0.17964053", "4.76837e-07"], ["55", "0020", "[59, 1]", "1", "0.0031929016", "4.76837e-07"], ["63", "0000", "[259, 146]", "1", "0.0038070679", "4.76837e-07"], ["63", "0006", "[644, 340]", "1", "0.089951992", "7.15256e-07"], ["63", "0013", "[268, 97]", "2", "0.0019004345", "4.76837e-07"], ["63", "0041", "[23, 533]", "1", "0.0010921955", "9.53674e-07"]]} />
      <Text>这些是按失败选出的描述性位置，不能用于估计真实训练发生率。最大例scan55/cam0000/(161,215)：affine2.687563658、packet2.502209425、factor2.681849957。保存割线斜率约2.1103/.00485/.00490，此阶段尚未重建根；后续已完成全部9点的实际前缀和高精度根诊断，见下一页。不能从割线直接宣布小斜率因果机制。</Text>
      <Text>三臂系数求值均FP64标量、输入/solver仍FP32；direct factor额外48B/基元与预处理含在forward成本。每视角5次预热、3轮×10、seed13018串行随机化顺序；像素和计时重复不是独立场景样本。当前版本重复求值与FP64成本可能有优化空间，未实现设想不改变本次拒绝。</Text>
      <H2>决定与后续</H2>
      <Text>按预定门槛不进入GT改善解释或训练扩展。后续已完成的有界失败诊断覆盖全部9异常像素：真实metadata/membership/prefix、两臂系数、稳定高精度根与FP32响应，区分表示误差、条件数、乘积舍入和有限bracket。TGS原文与固定CUDA实现已补齐；sigma是屏幕锐度，不能等同GGGS沿射线厚度。完整CUDA后向仍未实现。</Text>
      <Text size="small" tone="secondary">来源：frozen_exact_render_v1、frozen_exact_render_report_v1.json、frozen_exact_assets_v1.json；2026-10-14。17源/archive、50产物、3binary、51生成文件、652原源与3原PLY核验；这是全图/计时算术重放，非独立rerender。原LaTeX原位v21编译通过。</Text>
    </Stack>}
    {tab === 16 && <Stack gap={18}>
      <H2>大差异已存在于保存系数的数学根中</H2>
      <Text>完整保留按24视角失败选出的9个像素，覆盖8视角、17581个tile候选和772个accepted项。18/18终端乘积用精确二进制有理数判定小于1/2，全部有有限根；这些样本排除了false finite假设。</Text>
      <Table headers={["独立检查", "完成结果", "范围"]} rows={[
        ["120位根与180位端点符号复查", "18个原模型根及18个另列camera-origin根通过", "数值括区，非区间证明"],
        ["CUDA原FP32路径depth/slope位重放", "18/18通过", "同系数与初值，.45 gate/.4 range、8分割/5轮"],
        ["全9点剩余数值/转换差", "最大8.9645251e−5 ray；占观测差最大2.71249%", "不是单纯乘积误差归因"],
        ["稳定double与MP真根", "最大约4.40e−13 ray", "独立数值核验，非原kernel"],
        ["MP有限五轮与MP真根", "最大2.11056e−7 ray", "同保存初始区间"],
        ["更换FP64稳定log-T", "大packet/factor差仍存在", "单独提高求根精度不足以修复"]
      ]} />
      <H2>全部9点的射线长度差分解</H2>
      <Text>单位均为模型射线长度（使用probe rln转换）；packet减factor。观测差=数学表示根差+剩余数值/转换差。z-depth旧最大差为.1796405315，两种单位不混用。</Text>
      <Table headers={["场景/相机", "pixel(x,y)", "accepted项", "观测ray差", "数学根ray差", "余项ray差"]} rows={[["scan37/0020", "[30, 1]", "30", "-0.0013837314", "-0.0014186229", "3.489151e-05"], ["scan37/0034", "[223, 276]", "32", "0.0090015211", "0.0090051709", "-3.649856e-06"], ["scan55/0000", "[161, 215]", "111", "-0.18208014", "-0.18209892", "1.878635e-05"], ["scan55/0020", "[59, 1]", "22", "0.0033364602", "0.0033931537", "-5.669344e-05"], ["scan63/0000", "[259, 146]", "98", "0.0038410559", "0.0037514106", "8.964525e-05"], ["scan63/0006", "[644, 340]", "156", "-0.091403749", "-0.091407097", "3.34723e-06"], ["scan63/0013", "[268, 97]", "174", "0.0019238642", "0.0018716798", "5.218446e-05"], ["scan63/0013", "[110, 300]", "49", "-0.0010986222", "-0.0010809464", "-1.767578e-05"], ["scan63/0041", "[23, 533]", "100", "-0.0011413916", "-0.0011559606", "1.4569e-05"]]} />
      <H2>证据限制与下一项</H2>
      <Text>构建架构环境曾改写缓存binary；旧SHA恢复失败已保留。新固定binary在全8受影响视角重放旧depth/slope/RGB/alpha/normal位模式，但不证明历史隐藏buffer或指令身份恢复。prefix来自真实buffer，系数与初始化来自复制表达式probe。</Text>
      <Text>18根真实−T_t约3.1154e−4至1.97018e−2，不能当根间统一下界。两组F32系数都尚未确立为raw世界真值；9点由异常选出，不估计普遍频率或宣布质量、新颖性。完整图像及2.60倍耗时失败保持。</Text>
      <Text>下一步锁定9点/772项，做alpha/peak/rho的8组合数学根机制分离，再从原PLY和相机独立高精度重算。仅有可解释、可修复并有成本优势的证据后，才推进完整后向与预登记质量对照。</Text>
      <Text size="small" tone="secondary">来源：exact_outlier_capture_v3、exact_outlier_root_v1.json、exact_outlier_solver_v2、exact_outlier_summary_v1.json；2026-10-14。LaTeX原位v22编译成功。静态研究快照，不自动读取进程。</Text>
    </Stack>}
    {tab === 17 && <Stack gap={18}>
      <H2>有效alpha表示差决定本批大根差</H2>
      <Text>固定全部9点/772 accepted项，factor=mask0、packet=mask7，bit0/1/2替换alpha/peak/rho。全9×8数学模型有限；120位根和180位端点72/72通过，18端点与旧oracle吻合。混合模型不一定对应共同3D Gaussian，六顺序分配只是代数机制分离。</Text>
      <Table headers={["固定混合检查", "最大变化（camera-ray模型单位）", "含义"]} rows={[
        ["全替换根差绝对值", ".1820989237（case2）", "保存模型的差"],
        ["alpha-only与全替换差", "9.58250e−7（全9最大）", "沿射线有效alpha主导"],
        ["只替换peak+rho", "9.58099e−7", "在这些选定模型中较小"],
        ["alpha×几何交互", "9.54527e−7", "保留非线性交互"],
        ["有效alpha含off-ray能量衰减", "非单独训练opacity参数", "不能叫opacity训练因果"]
      ]} />
      <H2>独立世界参照：结果依赖明确的数值对象</H2>
      <Text>原3 PLY逐ID核验772原始F32字段；saved-activated保留F32激活参数且真正逆非正交R covariance，raw-physical重新MP激活和unit quaternion。80位cov逆与140位inverse factor/residual相符，36根端点复查通过。相机用保存F32仿射view精确逆，t为单位camera-ray参数，非假定单位world弧长。参考alpha不先round F32。</Text>
      <Table headers={["case", "场景/相机", "packet−raw根", "factor−raw根", "较近模型", "中央60%区间界 P/F"]} rows={[["0", "scan37/0020", "-0.00020891572", "0.0012097072", "packet", "左/不充分"], ["1", "scan37/0034", "0.0082843402", "-0.00072083076", "factor", "右/左"], ["2", "scan55/0000", "-0.18169411", "0.00040481593", "factor", "左/右"], ["3", "scan55/0020", "0.0030197097", "-0.00037344399", "factor", "不充分/左"], ["4", "scan63/0000", "0.0036157762", "-0.00013563437", "factor", "不充分/左"], ["5", "scan63/0006", "-0.091426002", "-1.890572e-05", "factor", "左/右"], ["6", "scan63/0013", "0.0016324167", "-0.00023926309", "factor", "不充分/左"], ["7", "scan63/0013", "-0.0011064154", "-2.5468981e-05", "factor", "不充分/右"], ["8", "scan63/0041", "-0.00083347572", "0.00032248488", "factor", "不充分/右"]]} />
      <Text>两世界参照及第三共同co.w参照均factor较近8/9、packet较近case0。这是选定异常的描述，非GT、统计显著性或总体排名；factor在case0仍有约.0012097 ray误差，未获完整采用结论。</Text>
      <H2>平台裕量与信息丢失</H2>
      <Text>无peak区间[a,b]中，H=M_L−左tails+右tails，故M_L−左tails(a)≤H(t)≤M_L+右tails(b)。case1/2/5的两臂落在区间相反侧，其余6个单臂界不充分。独立MP180 direct-H核验18模型/126点；不是有向区间证书。最大两例中点H_t约5.584e−13/6.4e−161，根处局部斜率不能界跨层gap。</Text>
      <Text>实际共同co.w与filtered opacity在2307/17581候选、96/772 accepted末位不同，accepted最大5.96046e−8。控制这个共同amplitude后8/9比较仍同，参考根移位最多1.78528e−6 ray；两exact臂共享co.w，不是两臂独立缓存差。</Text>
      <Text>候选信息损失点：packet混用covariance→conic与precision→plane，且从投影xy反算mean的舍入/1e−7分母与factor view mean路径不同。尚不能唯一归因于其中一项。下一步一致precision factor构造b/c/Schur能量及条件误差界，再核验锁定旧输入与新构造。9F32缓存预算不等于速度或稳定性证据。</Text>
      <Text size="small" tone="secondary">来源：exact_outlier_mix_report/check_v1、exact_outlier_world_v1/world_audit_v1、gap_v1/gap_audit_v1；2026-10-14。原LaTeX原位v23编译成功。完整24视角数值及2.60倍耗时失败保持，当前无GT/训练扩展。</Text>
    </Stack>}
    {tab === 18 && <Stack gap={18}>
      <H2>一致构造改进真实异常；固定主候选仍被拒绝</H2>
      <Text>主候选QR_mixed以真正R逆及camera仿射A逆构造precision factor，QR三角U统一给peak/rho/energy。6F32 U+3F64 mean共48B；全F32九项36B、全F64九项72B为预定对照。均先double解码再将(p,rho,alpha)转F32；没有因结果换主候选。</Text>
      <Table headers={["固定组", "全部项数", "QR32局部失败", "mixed局部失败", "QR64局部失败"]} rows={[["real1992", "1992", "0", "0", "0"], ["outliers772", "772", "0", "0", "0"], ["stress132", "132", "3", "0", "0"], ["synthetic4096", "4096", "13", "12", "0"], ["critical18", "18", "0", "0", "0"]]} />
      <Table headers={["根比较（camera-ray单位）", "QR32", "QR_mixed", "QR64"]} rows={[
        ["9真实异常max误差", "3.89654e−4", "4.99316e−5", "5.61782e−6"],
        ["9真实异常>1e−3", "0", "0", "0"],
        ["9临界双层>1e−3", "5", "5", "4"],
        ["临界max误差", ".48882027", ".48882027", ".48882027"]
      ]} />
      <Text>全7010项保留，primary整体FAILED。旧4096属于开发，不是独立确认。未进入新seed确认、整图计时或训练；16.2s CPU数值检验不是GPU性能。v1最后组参照相减断言失败保留，v2用等价残差二次型，不clamp或删例。</Text>
      <H2>合法F32三维输入，正确舍入后根信息已经丢失</H2>
      <Text>相机/rotation I、query(0,0,1)、epsilon=8589935/2³²，两Gaussian z=.5/1.5，amplitude=.75/.5。第一中心y_A=7735379/2³²，y_B=15470759/2³³，相差一个F32间隔2⁻³³；第二y=0。均是合法正定世界输入。</Text>
      <Table headers={["对象", "世界A", "世界B"]} rows={[
        ["未round有效alpha₁", ".5000000205022779", ".4999999942937956"],
        ["正确F32 alpha₁", ".5", ".5"],
        ["全部6个F32 p/rho/alpha字节", "与B相同", "与A相同"],
        ["量化模型精确median", "1", "1"],
        ["未round物理根区间证书", "r_A<.512", "r_B>1.488"]
      ]} />
      <Text>60/120位有向区间验证两alpha位于.5的正确舍入单元，以及T_A(.512)小于.5、T_B(1.488)大于.5。物理根相距超过.976，因此仅依据相同F32射线系数产生同一z的算法，必有一侧误差超过.488。量化对称模型的根1本身是正确的；不能用更精确求解这个不同模型恢复已丢的信息。</Text>
      <Text>该存在反例在固定fixtures之后选择；不估计发生率，不排除原世界参数/完整metadata含额外信息，也未核验完整renderer选择、初始bracket、CUDA指令、backward或GT。区间证书信任mpmath.iv，并非proof-assistant核验。一般quantile敏感性、新颖性仍待审查。</Text>
      <H2>下一项：条件精度与可计算误差区间</H2>
      <Text>不继续缓存组合筛查。对层间质量裕量不足的像素，需要保留几何输入及有效alpha的必要精度，核验沿root链的误差区间/触发规则。先在全旧输入和锁定新构造检验正确性，再做完整CUDA成本/后向及真实质量验证。原首版packet24视角和2.60倍耗时失败保持。</Text>
      <Text size="small" tone="secondary">来源：coherent_factor_v2、coherent_factor_audit_v1.json、COHERENT_FACTOR_RESULT.md；2026-10-14。42源/archive、12产物、21030误差向量、54根记录核验；9quaternion inverse符号分量通过。原LaTeX原位v24编译成功，静态快照无活任务声明。</Text>
    </Stack>}
    {tab === 19 && <Stack gap={18}>
      <H2>条件精度：先验证数值证书，再进入GPU</H2>
      <Text>缓存仍由一致QR构造，增加可计算的U/mean误差预算；名义56B/基元尚未验证GPU对齐。低精度求根仅提出候选，两侧残差区间给严格符号才认证；否则读取原参数回退，仍需同一证书。区间预算和候选都不读取MP真根。</Text>
      <Table headers={["固定集合", "模型数", "直接认证", "回退", "未过门槛", "max根误差(camera-ray)"]} rows={[["旧真实9", "9", "7", "2", "0", "8.47554e-06"], ["旧临界9", "9", "0", "9", "0", "5.34208e-11"], ["新普通四层", "64", "64", "0", "0", "2.06531e-08"], ["新近阈值双层", "16", "0", "16", "0", "8.41328e-11"], ["新精确临界双层", "4", "4", "0", "0", "2.15574e-08"], ["新微小偏移", "4", "1", "3", "0", "1.36184e-05"], ["新无有限根", "4", "4", "0", "0", "无有限根"]]} />
      <Text>旧7010个peak/rho/alpha/energy包围参照；新seed13019共92模型在运行前固定。合计106有限根全部有括区，4无根分类正确；76有限根和4无根直接认证，30有限根回退。最大误差1.36184e−5，不是world mm、GT或图像质量；选择集合的回退比例不估计真实像素发生率。</Text>
      <H2>保留信息：amplitude/energy与精确横向偏移</H2>
      <Text>先由原二进制mean/query求横向偏移，再投影，避免大数相减吞掉微小energy。层间常数用amplitude/energy的log1p/expm1分解，避免把alpha接近.5的信息提前舍掉。预算来自有向区间、QR正交缺陷及polar扰动界；利用u·r=0给残差/peak误差界。普通混合精度已有先例，特定median证书新颖性未验证。</Text>
      <H2>独立复核与首版端点问题</H2>
      <Text>世界covariance MP240复核106对端点符号、4408使用中系数区间、4无根及精确有理括区宽度通过；56源/archive、9产物校验。v1有81个名义2e−3宽度微超，最多6.67911e−16；v2只向内舍入并保存完整残差区间，输入和判定不变。v1结果及旧QR失败保留。</Text>
      <Text>证书信任实验性mpmath.iv：53位significand、有向舍入、无界指数。CPU约11.71s不是GPU效率；尚无IEEE FP64 device等价、完整renderer初始括区/选择、backward、GT或训练证据。精确共线及双半质量有解析分支，可避免尾部下溢，但不是所有device输入都已解决。</Text>
      <H2>下一项与当前状态</H2>
      <Text>先实现和核验GPU预算/尾部符号/原参数回退，测完整成本；通过再做24视角、全部梯度及跨场景配对训练、held-out图像和几何评价。当前CPU句柄已终态，无新训练启动，整体研究目标未完成。</Text>
      <Text size="small" tone="secondary">来源：guarded_precision_v2/summary.json、guarded_precision_audit_v1.json、GUARDED_PRECISION_RESULT.md；2026-10-14。原LaTeX原位v25编译成功。现有视图静态快照；TypeScript语法检查不替代SDK/runtime验证。</Text>
    </Stack>}
    {tab === 20 && <Stack gap={18}>
      <H2>共享严格数学核心：host与CUDA端点通过</H2>
      <Text>CUDA12.8原文说明standard数学函数的ULP边界来自非穷尽测试、不是保证界。严格路径采用基本IEEE有向运算、range reduction和级数余项；不调用standard exp/log/log1p/expm1。host用MSVC严格浮点和动态舍入，device用CUDA double intrinsic，所有源与输入归档。</Text>
      <Table headers={["核验范围", "全部项数", "失败", "参照/限定"]} rows={[
        ["Windows scalar", "774", "0", "固定seed13023，独立Decimal"],
        ["Windows端点", "212", "0", "106有限模型，复用独立MP240世界端点"],
        ["CUDA端点", "212", "0", "同一header，CPU区间与CPU选定bracket"],
        ["精确有理常量", "ln2/tail/drop界", "0", "Fraction atanh级数及余项"]
      ]} />
      <H2>在log域保留小于IEEE标量范围的尾部</H2>
      <Text>g=exp(ell)，b=-log(1-g)/2。当ell小于−40，log b介于ell−log2和ell−log2+2⁻⁵⁶；从g小于2⁻⁵⁷及几何级数界推出。正负残差项用共同log尺度有向求和；缩放后低于exp(−700)仍保留正上界2⁻¹⁰⁰⁹。区间重叠或峰位交叉保留unresolved，不删例。</Text>
      <H2>测得开销：仍为参考组件</H2>
      <Table headers={["计时项", "值", "包括", "尚未包括"]} rows={[
        ["212端点event中位数", "222.9916504ms", "输出allocation/fill及kernel", "源参数预算、transfer、选择、求根"],
        ["设备", "RTX5090 Laptop / sm120", "CUDA12.8 gcc/g++11", "完整renderer与backward/GT"]
      ]} />
      <Text>本次设备验证使用保存的CPU coefficient区间和bracket；不是完整GPU条件精度算法。device未单独验证全部774scalar。本局部计时没有完整renderer对照，不能宣布加速。下一步便宜充分认证、设备预算/offset/mass/回退/求根，以及全部成本核验，严格核心保留作难点参照。</Text>
      <Text>之前WSL查询两次自动审批timeout没有执行；用户明确允许后，CUDA编译/运行完成。无GPU死锁证据，当前host和CUDA诊断均终态，没有新训练。原先质量与效率失败仍保留，整体科研目标未完成。</Text>
      <Text size="small" tone="secondary">来源：STRICT_INTERVAL_RESULT.md、strict_interval_host_v1、strict_interval_cuda_v1、strict_interval_audit_v1.json；2026-10-14。15+8源archive、9+5产物、9generated核验；host未执行CUDAwrapper的same-device guard源漂移明确记录。原LaTeX原位v26编译成功，静态视图仅作结果快照。</Text>
    </Stack>}
    {tab === 21 && <Stack gap={18}>
      <H2>从原参数到认证根：全部在设备，固定开发通过</H2>
      <Text>输入只有原mean/scales/quaternion或R、camera affine、query、amplitude和110模型start/count。真正F逆用有向算术，two-pass modified Gram–Schmidt提供名义Q/U，polar/residual预算覆盖缓存与构造误差。设备求根未输入CPU候选、bracket或MP根；参照只在返回后核验。</Text>
      <Table headers={["阶段", "完整固定范围", "结果"]} rows={[
        ["设备预算", "7322层×2路径×4系数=58576", "全部参照包围，零失败"],
        ["设备mass", "110模型×2=220", "106finite/4no-root分类正确"],
        ["旧CPU bracket端点组件", "106finite模型", "low79/high106认证"],
        ["设备自主候选/新bracket", "106finite+4no-root", "finite76low/30high，4分类正确"],
        ["自主根max误差", "camera-ray单位", "1.361838888e−5，门槛1e−3不改"]
      ]} />
      <Text>候选可用标准math函数，但输出必须由严格基本算术/级数/log-tail给两端符号。低系数转F32，仅作候选；未认证则高路径读取amplitude/energy并重新求根。tau0严格小于1/1000，向内舍入保证精确宽度。已观测fixed seed属于开发，不再称独立新确认或真实像素发生率。</Text>
      <H2>独立世界复核与局部成本</H2>
      <Text>MP240 covariance逆重算106个新设备端点对，4408使用中系数、scaled包围、精确有理宽度和4无根通过。56+14源/archive、9+7产物、10+11generated校验。原生产CUDA与训练未变。</Text>
      <Table headers={["自主根运行的CUDA event中位数", "规模", "ms"]} rows={[
        ["原参数预算", "7322双路径", ".441504"],
        ["求根与证书", "110数学模型", "705.173584"],
        ["两阶段合并", "上述固定输入", "705.568237"]
      ]} />
      <Text>包括输出allocation/fill与相应kernel，未包括host transfer或完整renderer，诊断26double/路径输出不代表56B生产cache的布局实测。成本尚需降低，未证明加速。正确forward也不说明临界薄层opacity梯度可表示或训练会改善。</Text>
      <H2>下一项：便宜充分认证与完整成本</H2>
      <Text>先让健康情况用便宜充分界，严格核心作难点参照；保持相同物理根和全部临界输入。然后核验完整24视角选择/初始范围/成本与世界backward，通过后再跨场景训练、held-out图像与几何评价。当前诊断均终态，无新训练；整体科研目标未完成。</Text>
      <Text size="small" tone="secondary">来源：DEVICE_ROOT_RESULT.md、device_budget_v1、device_root_v1、device_root_audit_v1.json；2026-10-14。原LaTeX原位v27编译成功；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
    {tab === 22 && <Stack gap={18}>
      <H2>单一实现优化：同一输出，局部诊断快1.83倍</H2>
      <Text>将严格级数六处整数倒数替换为CUDA constant-memory表，k=1..51。精确有理数验证实际编译表的相邻binary64下/上界，等于原IEEE directed reciprocal；区间乘法、余项、候选、预算、门槛和7322输入不改。原生产CUDA与训练源码未改。</Text>
      <Table headers={["逐位检查", "完整范围", "结果"]} rows={[
        ["control / variant预算", "7322层×2路径×26double", "全部位一致"],
        ["control / variant根", "110模型×12double，包括NaN bits", "全部位一致"],
        ["重编control / 原device_root_v1", "完整预算与根", "全部位一致"],
        ["有限根与无根", "106finite：76low/30high；4no-root", "分类/严格符号/精确宽度保持"]
      ]} />
      <H2>五对交替CUDA event：同进程对照，ms</H2>
      <Table headers={["批次", "control中位数(ms)", "variant中位数(ms)", "配对比几何均值"]} rows={[
        ["7322双路径预算", "0.403200", "0.268448", "0.681301"],
        ["110模型根与证书", "705.381042", "384.888275", "0.545663"],
        ["预算+根合并", "705.739014", "385.180573", "0.545837"]
      ]} />
      <Text>预定所有输出位一致且合并比≤0.95的局部门槛通过：合并耗时减少45.4163%，速度比1.83205。每对合并比0.545781–0.545884；不是独立训练种子或显著性检验。包含输出allocation/fill，排除host输入传输、完整renderer/backward/GT/optimizer。</Text>
      <H2>仍需完成的证据</H2>
      <Text>位一致继承旧MP240世界参照，本轮不是未观测样本确认。整数倒数缓存是数值实现优化，尚无新颖性或完整训练收益。385ms仍是高成本诊断；下一步固定普通像素便宜充分认证及严格回退协议，再核验冻结24视角的选择、初始范围、实际延迟/内存和世界VJP，通过后再扩展配对训练。</Text>
      <Text>v1仅在源码匹配计数断言处失败，未运行CUDA比较；修正r/x循环计数后v2 complete/exit0(session15185)，失败来源与记录保留。当前无新训练卡死证据，整体科研目标未完成。</Text>
      <Text size="small" tone="secondary">来源：RECIPROCAL_ROOT_RESULT.md、reciprocal_root_v2、reciprocal_root_audit_v1.json；2026-10-14。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
    {tab === 23 && <Stack gap={18}>
      <H2>便宜的充分认证：同一物理根，失败仍走严格路径</H2>
      <Text>固定log九项、exp十二阶、小范围log1p/expm1八阶。余项由有向IEEE基本运算包围；两端严格相反符号才short返回。否则同候选调用原长证书，再按原逻辑进入high；仍不能认证保留unresolved。参数预算、候选、mass、容差和全部固定输入不变，原生产CUDA/训练未修改。</Text>
      <Table headers={["主批完整核验", "范围", "结果"]} rows={[
        ["预算/重编control", "7322层，完整输出位", "与旧倒数表一致"],
        ["独立scalar", "774保存输入，Decimal240+MP240", "包围全过"],
        ["有限根", "106：76low/30high", "最终106 short；strict曾调用30次"],
        ["世界逆/端点", "106对端点、4408使用系数、4无根", "MP240及精确宽度全过"]
      ]} />
      <H2>五对交替CUDA event：倒数表control到short，ms</H2>
      <Table headers={["批次", "control中位数(ms)", "variant中位数(ms)", "配对比几何均值"]} rows={[
        ["110模型根与证书", "384.782196", "245.457794", "0.637784"],
        ["预算+根合并", "385.132538", "245.741669", "0.638155"]
      ]} />
      <Text>合并耗时减少36.1845%，速度比1.56702；预定全数值通过且比≤0.95的局部门槛通过。含输出allocation/fill，不含host transfer/完整renderer/backward/GT/训练。此前705→385与此次385→246属于不同运行，不能称同进程三臂比较。主批全是已观测开发输入。</Text>
      <H2>另一个固定平坦族：严格成功与未解分支都已实测</H2>
      <Text>单Gaussian amplitude=3/4、peak=1、scale=10^j，j=0..7，解析median=1。只加载主批冻结二进制，解析根不输入设备；这8例不混入主批门槛或计时。</Text>
      <Table headers={["scale", "control", "hybrid"]} rows={[
        ["1、10、100", "3 certified", "3 short"],
        ["1000", "certified", "strict成功救回"],
        ["10000至10000000", "4 unresolved", "4 unresolved"]
      ]} />
      <Text>全部返回证书均过独立MP240单基元公式、scaled包围、精确宽度，无误报成功。未解案例保留，完整renderer必须明确处理；不代表真实像素发生率或统一可认证保证。标准级数界暂不主张学术新颖性。</Text>
      <H2>下一项：完整渲染的可扩展成本与反向门槛</H2>
      <Text>固定24视角的选择/生产初始范围和成本协议，检查缓存与射线内并行计算；独立世界VJP通过后才扩展跨场景配对训练、留出图像及几何评价。极平坦族的精度限制和此前质量失败仍在，整体科研目标未完成。</Text>
      <Text size="small" tone="secondary">来源：SHORT_CERTIFICATE_RESULT.md、short_certificate_v1、short_certificate_audit_v1.json、short_fallback_v1；2026-10-14。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
    {tab === 24 && <Stack gap={18}>
      <H2>全部24原分辨率视角：容量192不能用于完整图像</H2>
      <Text>scan37/55/63各8个既定开发视角，共10,834,488像素。只在原affine内核真正RGB接受及median第一次迭代接受位置计数，无容量截断或抽样。新的同源control/计数版本保留全部原RGB/alpha/normal/depth/slope/radii位；原invalid depth=0不变，RGB与median接受数在initial eligible中零不一致。</Text>
      <Table headers={["全部实际名单分布", "initial eligible", "finally valid"]} rows={[
        ["像素数", "9,165,957", "9,145,921"],
        ["min / p50 / p90 / p99 / max", "3 / 88 / 155 / 234 / 461", "3 / 88 / 155 / 234 / 461"],
        ["超过192", "334,083 (3.644824%)", "334,072 (3.652688%)"],
        ["超过256", "41,536 (0.453155%)", "41,536 (0.454148%)"],
        ["超过512", "0", "0"]
      ]} />
      <Text>512覆盖本批并不证明通用容量。后续名单按真实计数动态分配，逐像素检查写入数；不能删去超容量像素后声称全图认证。initial与finally valid分别保留，约870M primitive-ray关联是全部24视角汇总，不是单视角同时常驻内存。该固定名单未观察到非ball贡献项。</Text>
      <H2>容量尾部有场景差异</H2>
      <Table headers={["场景(各8视角)", "initial max", "finally valid超过192像素数"]} rows={[
        ["scan37", "444", "153,403"],
        ["scan55", "382", "33,951"],
        ["scan63", "461", "146,718"]
      ]} />
      <Text>scan63/0041的valid超过192比例10.6250%，不能只看全图平均比例。独立CPU审核669来源、50产物、两臂各16生成源、实际binary/vendor header/ZIP，重算全部保存depth/slope位、计数及直方图；RGB/alpha/normal/radii在运行中完整比较，未留存第二次离线位审核数组。</Text>
      <H2>计数开销与下一步</H2>
      <Text>每视角3次预热、5对交替native render，event配对比的视角几何均值1.025855，约2.59%计数开销。control中位数8.235–21.230ms，instrumented8.521–21.301ms；新增peak-added allocated约6.69–7.95MB。它们与110模型245.7ms不是同一工作量，不构成方法速度比。</Text>
      <Text>下一项捕获动态CSR实际primitive-ID名单，逐像素验证数量与顺序，再接共享参数缓存、射线内并行认证/回退；明确unresolved政策和世界VJP后才扩大训练。当前没有完整认证renderer、GT改善、训练收益或新颖性结论，整体目标仍未完成。</Text>
      <Text size="small" tone="secondary">来源：FULL_TOPOLOGY_RESULT.md、full_topology_v1、full_topology_audit_v1.json；2026-10-14。首个离线审核因默认GBK读UTF8源失败，记录保留，显式UTF8后通过。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
    {tab === 25 && <Stack gap={18}>
      <H2>动态CSR：完整实际ID与位置，不再用192截断</H2>
      <Text>用冻结计数二进制读首轮实际接受数，int64 prefix-sum分配offsets及int32 ID/position pairs；在原FIRST接受位置写入。检查pixel span及总容量，越写尝试置bit1，数量不符置bit2；任何非零状态不算成功。原生产源码未变。</Text>
      <Table headers={["完整范围", "核验", "结果"]} rows={[
        ["24视角 / 10,834,488 pixels", "原native图像与计数", "全部逐位一致"],
        ["870,178,922 associations", "与原initial名单数量", "精确相同"],
        ["全部关联GPU checker", "ID / 递增位置 / 前缀 / tile / point-list", "全部通过，status=0"],
        ["全部关联独立CPU", "offset差 / 每条ID与原位置 / 同pixel顺序", "870,178,922条通过，无抽样"]
      ]} />
      <Text>36来源、54产物、18生成源、binary/vendor headers/ZIP均复核。完整depth/counters/CSR和原point-list离线保存；RGB/alpha/normal/radii在运行中全部比较，未为第二次离线位审核保存数组。全部仍是此前观察的开发视角。</Text>
      <H2>错误分配保护与成本</H2>
      <Table headers={["预定guard probe", "目标pixel状态", "其他状态/原图像"]} rows={[
        ["少分配1项", "3，越写尝试被阻止且数量不符", "其他pixel=0；原图像/计数位一致"],
        ["多分配1项", "2，恰好1个sentinel未填", "其他pixel=0；原图像/计数位一致"]
      ]} />
      <Text>每视角records本体约157–383MB，offsets/status另计；6.96GB是24视角汇总，不是一次常驻。5对交替capture/count使用已分配CSR、含fill/reset，配对比1.074844：这是捕获阶段约7.48%开销。</Text>
      <Text>完整count→scan/总数读取/allocation→capture→checker单次event观察为23.680–50.505ms，同步wall23.815–49.580ms；包含host间隙，CPU传输/压缩/编译另计。没有完整流程配对性能采用试验，不代表最终认证/训练速度。</Text>
      <H2>下一项：共享源参数缓存与射线内并行</H2>
      <Text>原世界因子/误差预算按primitive共享，减少870M关联中的重复构建；用动态名单求候选/证书，保留原物理根与初始范围语义，明确unresolved及公共F32输出舍入。世界VJP、完整成本、跨场景训练、留出RGB/几何及新颖性仍未完成。</Text>
      <Text size="small" tone="secondary">来源：CSR_CAPTURE_RESULT.md、csr_capture_v1、csr_capture_audit_v1.json；2026-10-14。主批session19995与CPU session95619均exit0。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
    {tab === 26 && <Stack gap={18}>
      <H2>query之前按primitive缓存；query公式完整保留</H2>
      <Text>原inverse/QR/polar/residual与mean预算拆入共享cache；保留真实mean区间以证明aligned分支。low六项U是F32精确提升，high为F64，eta/nu向上F32。query与AMP只在decode输入，不改物理模型、余项、候选、mass、容差或原生产代码。</Text>
      <Table headers={["固定完整检查", "范围", "结果"]} rows={[
        ["全部预算 / 原archive", "7322×2×26double", "完整位一致"],
        ["根/证书", "106finite+4no-root", "含NaN bits全部一致"],
        ["八次query重用", "58576项", "完整预算位一致"],
        ["独立审核", "25来源 / 28产物 / 12generated", "原query代码/缓存字段/数组/hash均过"]
      ]} />
      <H2>固定5对重用计时，CUDA event ms</H2>
      <Table headers={["阶段", "中位数(ms)"]} rows={[
        ["build7322", "0.081056"],
        ["decode58576", "1.160480"],
        ["original repeat8", "1.481312"],
        ["build once+decode", "1.231136"]
      ]} />
      <Text>配对比几何均值0.830169，约16.9831%时间减少；原位一致且比≤0.95局部门槛通过。包括输出allocation/fill和ID范围检查/sync，不含host输入传输或完整renderer/training。重复query不是新场景，继承旧MP240参照而非新增确认。</Text>
      <H2>完整24camera的全部基元cache构造</H2>
      <Table headers={["场景", "P", "每camera cache bytes"]} rows={[
        ["scan37", "1,010,226", "226,290,624"],
        ["scan55", "1,052,385", "235,734,240"],
        ["scan63", "510,660", "114,387,840"]
      ]} />
      <Text>全部24视角两mode status=0、cache有限；相机/PLY/activated输入身份与原CSR一致，不按visibility删primitive。真实build event中位数3.343–37.605ms，全部高值保留。224B/row为诊断格式，另有288B/source row，不是此前56B拟定生产布局。</Text>
      <Text>真实cache hash/eta统计在运行中记录，大数组未永久存为第二次离线/MP全量审核。真实build尚未解码全部870M关联，更不是完整图像根证书。下一项按需CSR解码、射线内并行求根，保留原range语义并明确unresolved/F32输出舍入；世界VJP、完整成本、GT与训练仍未完成。</Text>
      <Text size="small" tone="secondary">来源：SHARED_CACHE_RESULT.md、shared_cache_v1、shared_cache_audit_v1.json；2026-10-14。session55325 exit0。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
  </Stack>;
}




