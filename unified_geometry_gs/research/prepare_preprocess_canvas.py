"""Guarded in-place update of the existing research canvas, no extra editor tab."""
from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
managed=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
oldhash=hashlib.sha256(managed.read_bytes()).hexdigest();source=managed.read_text(encoding='utf-8-sig')
def replace(old,new):
    global source
    if source.count(old)!=1:raise ValueError('Canvas anchor mismatch: '+old[:60])
    source=source.replace(old,new)
replace('"采样与反传稳定性"];','"采样与反传稳定性", "世界参数导数"];')
replace('useState(0)','useState(9)')
replace('已定位世界参数反传不稳定；完整质量与加速门槛仍未通过',
        '已通过局部导数核验；同精度质量对照正在训练')
replace('完整 scan55 高阶反向 pilot 已结束：平均 Δoverall +0.000424087 mm、Δs2d +0.000908171 mm，未过几何门槛。同状态控制重复的 Δoverall +0.001001144 mm，说明单次对照不足以隔离方法作用。采样分组的冻结完整损失/反传时间降低约 4%–8.5%，但追加完整梯度门槛失败，暂不扩大训练。',
        '本轮建立了独立 CUDA 世界预处理导数参照，定位法向长度、射线精度分子及滤波系数三处局部问题。组合 FP64 修正在 48 合成与 72 真实模型基元上通过原门槛；20 步实际训练 smoke 通过，scan55 同 FP64 精度的 30k 质量对照正在进行。此前高阶求根、分组和混合精度的失败结论保留。')
replace('下一步是独立 CUDA 世界参数 VJP 参照。','局部独立 CUDA 世界参数 VJP 参照已完成，详见新增页面；完整渲染尚未独立核验。')
replace('尚未验证 CUDA 世界导数或完整训练质量；不解释法向上游为零的几何损失波动。',
        '局部 CUDA 世界导数现已核验，完整训练质量仍待结果；不解释法向上游为零的几何损失波动。')
report=json.loads((ROOT/'research/results/preprocess_audit_report_v1.json').read_text(encoding='utf-8'))
real=report['evidence']['world_preprocess_real_v1'];syn=report['evidence']['world_preprocess_synthetic_v2']
labels={'normal':'法向','plane_slope':'射线平面斜率','ray_precision':'射线精度','opacity_coef':'opacity 补偿','conic':'投影 conic'}
rows=[[label,f"{real['summary']['legacy64'][k]['max_scaled_error']:.7g}",
    f"{real['summary']['all64'][k]['max_scaled_error']:.7g}",str(real['summary']['all64'][k]['failed_groups'])]
    for k,label in labels.items()]
timing=json.loads((ROOT/'research/experiments/corrected_quality55_v1/manifest.json').read_text(encoding='utf-8'))
if timing['status']!='running':raise ValueError('This version is explicitly a running-quality snapshot.')
new='''    {tab === 9 && <Stack gap={18}>
      <H2>独立局部 CUDA 参照：三处链式求导修正</H2>
      <Text>直接调用实际 CUDA 预处理函数，固定 conic、射线参数或法向上游；独立 float64 图通过矩阵公式和简化法向求导。scale 激活、四元数规范化纳入 pullback，conic 非对角上游按实际 shader 的 1/2 约定处理。</Text>
      <Table headers={["数学检查", "实际结论", "适用边界"]} rows={[["n=c/‖c‖", "VJP 应除以原 c 模长", "非零 c；单基元相机法向"], ["κ=√(rᵀPr/L)", "∂uκ=(Pr)₁/(κL)−κu/L；旧式漏分子项", "P 正定，κ>0；clamp 须继续链式求导"], ["η=√(D₀/D₁)", "∂D₁η=−(∂D₀η)η²；旧式为 −η", "光滑未截断分支；epsilon/floor 仍保留"]]} />
      <H2>三个真实模型的预选基元：相同精度比较</H2>
      <Table headers={["固定输出上游", "旧式 FP64 最大归一化误差", "修正 FP64 最大归一化误差", "修正超门槛组数"]} rows={ROWS} />
      <Text>scan37/55/63，每场景相机 0013；16 个轴比≤10 和 8 个轴比&gt;10 的投影范围内基元，共 72。五输出块 × 四参数块 = 1440 检查，含结构零。规则 ‖gCUDA−gAD‖/max(‖gAD‖,1e−5)≤.001；参照不是同源反传或控制重放。最大 forward 归一化差 3.865e−6，最小 FD 步长的最大归一化差 1.175e−6。</Text>
      <Text>组合 FP64 全部通过；组合 FP32 未过相同门槛，保留失败。真导数为零的检查使用绝对下限，其误差不能称为非零真梯度的相对百分比。投影范围不等于实际可见性，也未覆盖极端轴比与特征分解 fallback。</Text>
      <Text>SYNTHETIC</Text>
      <H2>额外证明：旧射线反传一般不可积</H2>
      <Text>固定 κ 上游为 1、其他输出上游为 0。旧 viewport 场的旋度为 (u h₂−v h₁)/(κL²)；正定 P=diag(1,2,3)、u=v=.5 时等于 √10/45≈.07027284，因此附近不存在以此为梯度的 C² 标量函数。各向同性时 κ 本应与方向无关，旧中心场却等于 ∇[κ log cosθ]。</Text>
      <Text>精确符号恒等式、64 个 AD 例和收缩回路环流通过；该项不代表完整训练没有替代目标，不证明历史重建失败原因，也不作新颖性声明。</Text>
      <H2>已通过真实训练 smoke，完整质量结果尚未产生</H2>
      <Text>legacy64 / corrected64 各 20 个 Adam 更新，完整共同状态、首相机 0048 和首个前向损失分量一致；参与梯度有限。两完整后端局部函数与探针对应函数逐字节相同。首次 smoke 的记录器调用帧失败已保留，修复后在 v2 新目录重跑。</Text>
      <Text>质量对照继续同一 6999 步状态至 30k，旧式和修正式均用 FP64，原五轮、batch2、NCC 与损失配方不变。依次训练、统一提取/配准、evaluator seeds 0/1/2 和 49 训练视角评估。进入下一阶段门槛：平均 overall 至少降低 1%、s2d 不增加、训练 PSNR 不下降超过 .2 dB。</Text>
      <Text>真实配方 kernel_size=0，因此系数平方修正在理想算术下没有效应。不能把非零 kernel 的合成错误解释为本 pilot 的收益。一次训练对照和三个 evaluator seeds 均不构成独立训练重复或统计显著性；顺序墙钟不作为提速结论。</Text>
      <Text size="small" tone="secondary">SOURCE</Text>
    </Stack>}
'''
new=new.replace('ROWS',json.dumps(rows,ensure_ascii=False))
new=new.replace('SYNTHETIC',f"另有 48 个合成例覆盖 FoV 内部和 x clamp、kernel=.3；组合 FP64 的 1344 检查全部通过。旧 FP64 的 opacity 系数最大归一化误差 {syn['summary']['legacy64']['opacity_coef']['max_scaled_error']:.6g}，修正后 {syn['summary']['all64']['opacity_coef']['max_scaled_error']:.6g}。")
new=new.replace('SOURCE','来源：preprocess_audit_report_v1.json、preprocess_chain_rules_v1.json、ray_precision_integrability_v1.json、corrected_smoke55_v2、corrected_quality55_v1/manifest.json；静态快照 '+datetime.now(timezone.utc).isoformat()+ '。质量训练状态不是实时监控。')
marker='    <Text size="small" tone="tertiary">所有数字'
if source.count(marker)!=1:raise ValueError('Canvas footer anchor changed.')
source=source.replace(marker,new+marker)
prepared=ROOT/'research/canvas_preprocess_prepared.tsx';prepared.write_text(source,encoding='utf-8')
guard={'managed_old_sha256':oldhash,'prepared_sha256':hashlib.sha256(prepared.read_bytes()).hexdigest()}
(ROOT/'research/results/canvas_update_source_guard_v3.json').write_text(json.dumps(guard,indent=2)+'\n',encoding='utf-8')
print('Prepared existing canvas from verified derivative evidence, explicit running-quality scope.')
