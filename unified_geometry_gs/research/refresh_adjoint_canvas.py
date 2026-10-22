"""Prepare the existing research canvas update from completed local evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
CANVAS=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    text=CANVAS.read_text(encoding='utf-8')
    if '高阶反向精化' in text:
        raise ValueError('Already updated; inspect before rewriting.')
    old='"GPU 求根对照"];'
    if old not in text:raise ValueError('Tab anchor changed.')
    text=text.replace(old,'"GPU 求根对照", "高阶反向精化"];')
    text=text.replace('NCC 未过几何门槛；四轮控制通过冻结网格容差',
        '高阶反向精化通过冻结梯度门槛；共同状态训练正在验证')
    old='三场景网格通过工程容差，训练梯度仍未通过。'
    text=text.replace(old,'三场景网格通过工程容差。普通四轮梯度未通过；另存高阶反向变体已通过 24 组开发检查，完整 scan55 对照正在运行。')
    grad=[]
    for file,label in (('solver_backward_repeat_v1.json','普通四轮'),('iqi_backward_v1.json','逆二次插值'),
                       ('adjoint_backward_v1.json','Newton 反向精化'),('adjquad_backward_v1.json','Halley + 二次分母')):
        data=json.loads((ROOT/'research/results'/file).read_text())
        vals=[c['relative_l2_drift'] for r in data['rows'] for c in r['gradient_comparison'].values()]
        grad.append([label,f'{100*max(vals):.5f}%',f'{sum(v<=.001 for v in vals)} / {len(vals)}'])
    timing=json.loads((ROOT/'research/results/adjquad_step_timing_v1.json').read_text())
    models=sorted({r['model'] for r in timing['rows']})
    cats=[m.rsplit('/',1)[-1] for m in models]
    series=[{'name':label,'data':[statistics.mean(r['gpu_ms_median'] for r in timing['rows']
        if r['model']==m and r['variant']==v) for m in models],'tone':tone}
        for v,label,tone in (('control','同源五轮','neutral'),('four','普通四轮','neutral'),('adjquad4','高阶反向四轮','info'))]
    dump=lambda x:json.dumps(x,ensure_ascii=False)
    addition='''    {tab === 7 && <Stack gap={18}>
      <H2>更少前向细分，在已有反向遍历中局部精化</H2>
      <Text>前向保持普通四轮。反向额外累积 T、R′、R″，在光滑局部使用 Halley 步，并以二次展开近似隐式分母；没有新增基元遍历，但增加算术和寄存器。</Text>
      <Table headers={["变体", "最大参数梯度相对 L2 漂移", "通过 0.1% 工程门槛"]} rows={GRAD} />
      <Text>固定 3 场景 × 2 视角 × 4 参数组，使用相同有效像素与 Rademacher 权重；对照是同源五轮。全部梯度有限。该表属于开发集检查，不能证明导数精确或训练质量。</Text>
      <H2>冻结完整图像前向加反向时间</H2>
      <Text size="small" tone="secondary">横轴：DTU 场景；纵轴：GPU 时间（ms）。每视角 5 轮 × 10 次，每轮随机变体顺序；先取每视角轮次中位数，再平均同场景 2 个视角。</Text>
      <BarChart categories={CATS} series={SERIES} height={260} valueSuffix=" ms" />
      <Text>损失为 RGB 均值 + 0.01 × 完整中位深度均值。高阶变体的六视角延迟降低约 6.5%～8.7%；不含优化器、密化或 NCC，不能当作完整训练加速。</Text>
      <H2>完整 scan55 对照：正在运行</H2>
      <Text>短程分叉已通过：相同完整模型/优化器、filter3D、绝对密化统计、Python/NumPy/Torch 随机状态和相机队列，首个相机一致。完整实验共同预热 6999 步，再分别到 30k；统一五轮提取，固定 evaluator seeds 0/1/2。尚无完整质量结果。</Text>
      <Text>修正恢复顺序，使优化器绑定恢复后的外观参数。真实 PGSR 检查点复现原实现外观参数不更新；隔离修正恢复了更新。本次两臂都使用相同修正，历史从头训练不受此问题影响。</Text>
      <Text>数学：在峰值分离、简单根、所需导数有界时，底层 Jacobian 可有二阶/三阶局部误差；粗前向 upstream 仍可能造成一阶损失梯度误差。已开技术说明补入条件性证明并通过内置编译。Newton、Halley、IQI 及隐式微分不作原创声明。</Text>
      <Text size="small" tone="secondary">来源：results/iqi_backward_v1.json、adjoint_backward_v1.json、adjquad_backward_v1.json、adjquad_step_timing_v1.json、checkpoint_restore_v1.json；experiments/adjquad_smoke55_v1/manifest.json、adjquad55_v1/manifest.json；2026-10-12。此页为静态快照。</Text>
    </Stack>}
'''.replace('GRAD',dump(grad)).replace('CATS',dump(cats)).replace('SERIES',dump(series))
    anchor='    <Text size="small" tone="tertiary">所有数字均为本地结果快照。'
    if text.count(anchor)!=1:raise ValueError('Footer anchor changed.')
    text=text.replace(anchor,addition+anchor)
    args.out.write_text(text,encoding='utf-8')
    print(json.dumps({'source_canvas_sha256':hashlib.sha256(CANVAS.read_bytes()).hexdigest(),
        'prepared_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),'gradient_rows':grad},ensure_ascii=False))


if __name__=='__main__':
    main()
