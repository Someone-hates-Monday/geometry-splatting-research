"""Prepare a guarded update of the existing results canvas from verified data."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
managed=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
source=managed.read_text(encoding='utf-8-sig')
oldhash=hashlib.sha256(managed.read_bytes()).hexdigest()


def replace(old,new):
    global source
    if source.count(old)!=1:raise ValueError('Canvas anchor mismatch: '+old[:80])
    source=source.replace(old,new)


replace('完整 pilot 未过几何门槛；当前定位完整损失反传的数值稳定性',
        '已定位世界参数反传不稳定；完整质量与加速门槛仍未通过')
replace('直接逆协方差因子求导的精确恒等式与 float64 检查通过，但局部 CUDA 修正仍不足以稳定完整反传。下一项有限检查是提高世界参数 backward 预处理精度；没有完整训练加速、质量改善或新颖性声明。',
        '直接逆因子局部修正不足以稳定完整反传。仅提高世界参数 backward 预处理中间精度后，18 视角、162 组梯度全部通过；选择性 FP64 也通过。但它们共享解析公式和 float32 基元累加，重复性不是世界导数真值。下一步是独立 CUDA 世界参数 VJP 参照。')
marker='      <Text>峰值自适应求根另一路线已结束：'
if source.count(marker)!=1:raise ValueError('Canvas insertion marker mismatch')
timing=json.loads((ROOT/'research/results/mixed_training_timing_report_v1.json').read_text(encoding='utf-8'))
rows=[[str(r['round']+1),' → '.join(r['order']),
       f"{r['variants']['control']['block_wall_seconds']:.6f}",
       f"{r['variants']['mixed1']['block_wall_seconds']:.6f}",
       f"{r['mixed1_control_block_wall_ratio']:.6f}"] for r in timing['rows']]
new='''      <Table headers={["追加 18 视角的完整损失梯度", "最大相对 L2", "0.1% 门槛"]} rows={[["原 float32 分组候选 / 原控制", "0.943811%", "失败：12/162"], ["FP64 batch1 / FP64 batch2", "0.003331%", "通过：162/162"], ["轴比 >10 使用 FP64 / FP64 batch2", "0.010293%", "通过：162/162"]]} />
      <H2>三对交替实际训练：没有通过加速门槛</H2>
      <Table headers={["轮次", "执行先后", "控制 block（s）", "混合精度 block（s）", "候选 / 控制"]} rows={ROWS} />
      <Text>每进程同一完整 6999 步状态，真实更新 500 步，前 50 步预热，计时 7050–7499；含实际密化、滤波和 Adam。墙钟比几何均值 0.990889，未达到预设 ≤0.97，且第三对超过 ≤1.01。拒绝效率采纳；这三对进程不是独立训练种子或完整质量实验。六进程均已结束。</Text>
      <H2>独立解析问题：法向归一化漏掉长度因子</H2>
      <Text>相对 forward 的局部 VJP 缺少原始相机法向模长的倒数。正定例的正确导数 −1/(2√2)，现有局部式 −1/2；符号、自动微分和方向有限差分通过。进一步化简得到 n=−Pd/‖Pd‖、原始长度=1/|nᵀd|，并通过 64 个独立世界 VJP 图检查。尚未验证 CUDA 世界导数或完整训练质量；不解释法向上游为零的几何损失波动。</Text>
      <Text size="small" tone="secondary">来源：world_precision_report_v1.json、mixed_training_timing_report_v1.json、normal_preprocess_vjp_v1.json、affine_normal_identity_v1.json；2026-10-12。效应只在各自测试口径成立，不作统计显著性或新颖性声明。</Text>
'''.replace('ROWS',json.dumps(rows,ensure_ascii=False))
source=source.replace(marker,new+marker)
prepared=ROOT/'research/canvas_precision_prepared.tsx'
prepared.write_text(source,encoding='utf-8')
guard={'managed_old_sha256':oldhash,'prepared_sha256':hashlib.sha256(prepared.read_bytes()).hexdigest()}
(ROOT/'research/results/canvas_update_source_guard_v2.json').write_text(json.dumps(guard,indent=2)+'\n',encoding='utf-8')
print('Prepared completed-result canvas with explicit UTF-8 and source hash guard.')
