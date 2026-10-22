"""Append the completed all-nine mechanism diagnosis to the existing canvas."""
from pathlib import Path
import hashlib
import json

BASE = Path(__file__).resolve().parent
TARGET = Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
PREVIOUS = '46ec5274fc0bbf871186974845c3ec733e0721de9eb0d3a741f157459481eda8'
raw = TARGET.read_bytes()
assert hashlib.sha256(raw).hexdigest() == PREVIOUS
text = raw.decode('utf-8').replace('\r\n', '\n')


def replace(old, new):
    global text
    assert text.count(old) == 1, old
    text = text.replace(old, new)


replace('"完整图像门槛"];', '"完整图像门槛", "全部异常根诊断"];')
replace('useState(15)', 'useState(16)')
replace('完整图像拒绝首版packet；异常机制待重建', '全9异常点已重建：大差异主要存在于系数表示')
replace('当前不扩大GT或训练，先重建实际异常射线。',
        '全9点的18个保存系数模型均有有限根，CUDA逐位重放通过；FP64稳定log-T仍保留大差异。当前不扩大GT或训练，下一步隔离系数并对照raw世界参照。')
replace('但尚未读实际贡献前缀或重解高精度根，不能宣布已找到小斜率因果机制。',
        '此阶段尚未重建根；后续已完成全部9点的实际前缀和高精度根诊断，见下一页。不能从割线直接宣布小斜率因果机制。')
replace('下一项有界失败诊断覆盖全部9异常像素：', '后续已完成的有界失败诊断覆盖全部9异常像素：')
r = json.loads((BASE / 'results/exact_outlier_summary_v1.json').read_text(encoding='utf-8'))
rows = [[f"scan{x['scan']}/{x['camera']}", str(x['pixel']), str(x['terms']),
         f"{float(x['observed_ray_gap']):.8g}", f"{float(x['represented_ray_root_gap']):.8g}",
         f"{float(x['remaining_ray_path']):.7g}"] for x in r['rows']]
block = '''    {tab === 16 && <Stack gap={18}>
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
      <Table headers={["场景/相机", "pixel(x,y)", "accepted项", "观测ray差", "数学根ray差", "余项ray差"]} rows={ROWS} />
      <H2>证据限制与下一项</H2>
      <Text>构建架构环境曾改写缓存binary；旧SHA恢复失败已保留。新固定binary在全8受影响视角重放旧depth/slope/RGB/alpha/normal位模式，但不证明历史隐藏buffer或指令身份恢复。prefix来自真实buffer，系数与初始化来自复制表达式probe。</Text>
      <Text>18根真实−T_t约3.1154e−4至1.97018e−2，不能当根间统一下界。两组F32系数都尚未确立为raw世界真值；9点由异常选出，不估计普遍频率或宣布质量、新颖性。完整图像及2.60倍耗时失败保持。</Text>
      <Text>下一步锁定9点/772项，做alpha/peak/rho的8组合数学根机制分离，再从原PLY和相机独立高精度重算。仅有可解释、可修复并有成本优势的证据后，才推进完整后向与预登记质量对照。</Text>
      <Text size="small" tone="secondary">来源：exact_outlier_capture_v3、exact_outlier_root_v1.json、exact_outlier_solver_v2、exact_outlier_summary_v1.json；2026-10-14。LaTeX原位v22编译成功。静态研究快照，不自动读取进程。</Text>
    </Stack>}
'''.replace('ROWS', json.dumps(rows, ensure_ascii=False))
marker = '  </Stack>;\n}'
assert text.count(marker) == 1
text = text.replace(marker, block + marker)
stage = BASE / 'canvas_exact_outlier_stage.tsx'
assert not stage.exists()
stage.write_text(text, encoding='utf-8')
record = {'previous_sha256': PREVIOUS, 'staged_sha256': hashlib.sha256(stage.read_bytes()).hexdigest(),
          'destination': str(TARGET)}
(BASE / 'results/canvas_exact_outlier_guard_v1.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))
