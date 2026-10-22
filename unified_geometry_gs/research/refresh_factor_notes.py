"""Update current continuation notes and stage the existing audit canvas."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=R/'CONTINUATION.md';old=p.read_text(encoding='utf-8');end=old.index('## 实际终态与入口')
    current='''# 继续研究时先读这里

2026-10-12。目标 active；真实质量/效率与学术新颖性尚未证实。先查实际进程和产物，禁止因聊天中断重启已有训练。

## 当前最新工作：factor64 完整质量候选已开始

- **正在运行 `factor_quality55_v1`，exec session 86385，WSL 父 PID418、训练子 PID639（PID 会变，须核验命令）。**
  入口 `run_factor_quality_pair.py`，日志 `results/factor_quality55_v1.log` 与目录内 `train_factor64.log`。
  已构建完整隔离后端、通过 20 次实际 Adam smoke，正在从共同 6999 状态训练 factor64 至 30k。
  队列随后自动对预选 legacy64 和 factor64 统一提取、配准、evaluator seeds 0/1/2、49 训练视角评估。
  不修改该 manifest/source_sha256 中的已排队源码或 `FACTOR_PREPROCESS_PROTOCOL.md`。
  不重复启动此训练。终态后验证集成记录、完整初态、首损失和所有实际结果字节，再作质量报告。
- **旧 `corrected_quality55_v1` 已 superseded_before_candidate；不恢复旧父队列。**
  旧父 PID405 在极端检查失败后被 SIGSTOP，控制子 PID425 继续并完成，随后成为 zombie；
  因父进程暂停，旧 manifest 仍称 running。控制 30k 模型及 integration 核验后才 SIGTERM→SIGCONT
  结束父队列，未启动 corrected64。原 manifest、暂停记录、偏离说明与控制均保留。
  见 `protocol_deviation_factor_v1.json`、`results/training_stall_diagnosis_v1.json`。
  app 摘要有一轮 interrupted，但无具体原因；不能宣称 GPU 死锁或平台故障已定位。
- 新只读检查器 `inspect_research_run.py --run research/experiments/factor_quality55_v1 --out <新json>`：
  在 WSL 中检查 argv、/proc 状态、日志写入年龄、集成产物；不自动重启或杀进程。
  `results/factor_queue_health_v1.json` 为静态存活快照，不是实时监控。
- **原组合 FP64 并非普遍通过。** `world_preprocess_stress_v1` 的全部 44 极端基元（最大轴比34819），
  normal/plane/ray_precision 有12/3/28组超过原 .001 门槛，最坏47.294；4个前向分量也失败。
  全部失败输入与归档保留。旧72真实+48合成通过不能外推到尾部。
- 直接精度因子路径避免依赖精确正交的 inverse→covariance 往返。新证明：
  δ=‖RᵀR−I‖₂，a=smax/smin 时，‖PΣ−I‖₂≤δ[1+(1+δ)a²]；
  标准四元数多项式 δ=4|‖q‖²−1|‖v‖²。保存样本最大残差28.0014。
  `results/precision_roundtrip_defect_v1.json`：9个精确符号分量、164保存样本上界、64非正交AD例通过。
  这是充分上界和标准矩阵微分，不是完整梯度误差界或原创代数声明。
- `factor_preprocess_stress_v1`：完整原44基元、880参数块通过，最大1.68e-5。
  `factor_regular_inputs_v1` 仅输入适配；`factor_preprocess_regular_v1`：全部72真实+48合成，
  2784参数块通过，最大1.20e-4。合计3664，含结构零，不当作独立统计样本。
  两套参照均保存；仍以物理float64图作入选门槛，device值图仅定位舍入。
  新路径不改forward，4个原极端前向失败仍未修复，也未独立核验完整渲染器。
- `factor_preprocess_backend.py` 的3个局部CUDA函数逐字节等于新探针；forward/config原样。
  新20步smoke梯度有限，初态c97bfd...、首相机0048、首损失与完成的legacy64一致。
  控制已在候选和mesh/image质量观测前预选并完成；固定复用，无论后续结果如何都不换基线。
  新候选是公式+因子+精度的组合干预，单项因果归因仍需消融。当前kernel_size=0系数修正理想无效。
- 质量门槛不变：mean overall至少改善1%，mean s2d不增加，TRAIN PSNR不下降超过.2dB。
  一对pilot、evaluator重复、训练视角不等于独立训练重复、统计显著性或新视角结论。
  顺序墙钟仅资源记账。未过门槛则保留负结果，不自动扩展训练。
- `PRECISION_FACTOR_AUDIT.md` / `results/factor_audit_report_v1.json` 保存机制与归档字节复核。
  当前LaTeX原位补入完整证明与限制，内置编译成功且源哈希见 `results/latex_compile_status_v8.json`。
  现有研究视图将更新为静态结果快照，不打开新标签，不自动读取进程。

## 已完成且保留的公式诊断

三个局部链式修正：法向原向量长度、射线精度分子、opacity系数平方因子。
`PREPROCESS_CHAIN_RULE_AUDIT.md`、`preprocess_chain_rules_v1.json`记录推导；
`ray_precision_integrability_v1.json`证明固定κ上游的旧viewport场一般旋度非零，
正定反例√10/45；各向同性旧式产生人工κ log cosθ项。仅针对该分量，不声称完整训练无替代目标。
原 `world_preprocess_synthetic_v2` 与 `world_preprocess_real_v1` 常规检查通过，边界见上述压力失败。
旧smoke v1调用帧失败、v2修复通过均保留。所有原始生产CUDA、冻结恢复器和此前负结果保留。

'''
    p.write_text(current+old[end:],encoding='utf-8')
    p=R/'README.md';old=p.read_text(encoding='utf-8');a=old.index('最新进展');b=old.index('## 阅读顺序')
    p.write_text(old[:a]+'''最新进展（2026-10-12）：极端轴比检查否定了原组合FP64的普遍正确性。
新直接精度因子反传在全部164个保存基元、3664局部参数块上通过原门槛；
四元数正交误差与轴比平方的充分上界已证明，并补入当前LaTeX，内置编译成功。
四个极端前向失败仍保留，完整渲染、质量和新颖性未证实。
旧corrected64未启动，完成的legacy64控制固定保留；新 `factor_quality55_v1`
已通过真实20步smoke，正按 `FACTOR_PREPROCESS_PROTOCOL.md` 训练候选并自动衔接评测。
详见 `PRECISION_FACTOR_AUDIT.md`、`CONTINUATION.md`；不要因聊天中断重复启动。

'''+old[b:],encoding='utf-8')
    canvas=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
    text=canvas.read_text(encoding='utf-8');original_sha=sha(canvas)
    a=text.index('      <H1>');b=text.index('    </Stack>',a)
    text=text[:a]+'''      <H1>极端反传失败已定位；直接因子候选正在训练</H1>
      <Text>原公式修正 FP64 在常规输入通过，但 44 个极端轴比基元揭示逆矩阵往返的问题。直接精度因子 VJP 在全部保存的 164 基元、3664 局部参数块通过原门槛；新后端 20 步真实训练检查通过，scan55 候选正在训练。几何和图像收益仍待统一评测。</Text>
'''+text[b:]
    a=text.index('    {tab === 9 &&');b=text.index('    <Text size="small" tone="tertiary">所有数字',a)
    section='''    {tab === 9 && <Stack gap={18}>
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
      <H2>三个公式补充与可积性反例</H2>
      <Text>法向 VJP 应除以原向量模长；κ=√(rᵀPr/L) 的视线导数须包含分子项；η=√(D₀/D₁) 的 D₁ 导数须含 η²。旧 κ viewport 场存在正定旋度反例 √10/45，各向同性旧中心场产生人工 κ log cosθ 项。仅对固定输出上游成立，不证明完整训练没有替代目标。</Text>
      <H2>完成控制；新候选训练和评测自动衔接</H2>
      <Text>旧 corrected64 在压力失败后未启动。旧父队列被暂停，控制子进程继续到 30k；结束父队列并保留原 manifest 后，明确记录新协议。固定复用质量观测前预选的 legacy64 控制，后续不换基线。新 factor64 完整后端三个局部函数与探针逐字节一致；20 次 Adam 更新的初态、首相机和首损失一致，梯度有限。</Text>
      <Text>factor_quality55_v1 正从共同 6999 状态训练至 30k，之后脚本自动统一提取、配准、evaluator seeds 0/1/2、49 TRAIN 视角评估。门槛不变：overall 至少改善 1%、s2d 不增加、TRAIN PSNR 不降超过 .2dB。组合干预仍需单项消融；kernel_size=0 时系数平方修正理想无效。无独立训练重复、留出图像、统计显著性或加速结论。</Text>
      <Text>停顿诊断：上一队列的 running 状态滞留于被暂停的父进程，训练子进程实际已完成。最近 app 摘要有一轮 interrupted，但未给出原因；不能认定平台故障或 GPU 死锁。新只读检查器核对实际 argv、进程状态和日志写入时间，不因 manifest 旧而重启训练。</Text>
      <Text size="small" tone="secondary">来源：factor_audit_report_v1.json、precision_roundtrip_defect_v1.json、training_stall_diagnosis_v1.json、factor_quality55_v1；SNAPSHOT_TIME 静态快照，非实时监控。LaTeX 原位补入证明，内置编译通过。</Text>
    </Stack>}
'''.replace('SNAPSHOT_TIME',datetime.now(timezone.utc).isoformat())
    text=text[:a]+section+text[b:]
    staged=R/'canvas_factor_prepared.tsx';staged.write_text(text,encoding='utf-8')
    (R/'results/canvas_factor_source_guard_v1.json').write_text(json.dumps({'original_sha256':original_sha,'prepared_sha256':sha(staged),'target':str(canvas),'prepared':str(staged)},indent=2)+'\n')
    print('Updated continuation and README; staged existing canvas with expected-source guard.')
if __name__=='__main__':main()
