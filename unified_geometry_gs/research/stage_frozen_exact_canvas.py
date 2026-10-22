"""Append the complete-image failed gate to the existing research results canvas."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
PREVIOUS='0c99d77c8ed878a2b5627c0e0e7aed8051ea833329c2fbb477b11d166703e914'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==PREVIOUS
text=raw.decode('utf-8').replace('\r\n','\n')
def replace(old,new):
    global text
    assert text.count(old)==1,old;text=text.replace(old,new)
replace('"精度候选与CUDA核验"];','"精度候选与CUDA核验", "完整图像门槛"];')
replace('useState(14)','useState(15)')
replace('CUDA局部前向通过；独立构造拒绝精度候选','完整图像拒绝首版packet；异常机制待重建')
replace('复制生产CUDA预处理的诊断扩展在既有1992输入局部通过；两种输入保护方案在新4096构造失败13/11，均拒绝。已有留出TEST仅+.0901dB且SSIM下降；质量、效率与新颖性目标尚未实现。',
        '完整24视角、10834488像素的三臂比较已完成：8视角最大深度差门槛失败，共9像素超阈值；packet/factor完整forward时间比2.60018。当前不扩大GT或训练，先重建实际异常射线。')
r=json.loads((BASE/'results/frozen_exact_render_report_v1.json').read_text(encoding='utf-8'))
rows=[[str(x['scan']),x['camera'],str(x['pixel']),str(x['pixels_gt1e-3']),f"{x['max_abs_depth_error']:.8g}",f"{x['p99_abs_depth_error']:.6g}"] for x in r['failures']]
block='''    {tab === 15 && <Stack gap={18}>
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
      <Table headers={["scan", "camera", "最大误差pixel(x,y)", "超阈值像素数", "depth最大绝对差", "depth p99绝对差"]} rows={FAILROWS} />
      <Text>这些是按失败选出的描述性位置，不能用于估计真实训练发生率。最大例scan55/cam0000/(161,215)：affine2.687563658、packet2.502209425、factor2.681849957。保存割线斜率约2.1103/.00485/.00490，但尚未读实际贡献前缀或重解高精度根，不能宣布已找到小斜率因果机制。</Text>
      <Text>三臂系数求值均FP64标量、输入/solver仍FP32；direct factor额外48B/基元与预处理含在forward成本。每视角5次预热、3轮×10、seed13018串行随机化顺序；像素和计时重复不是独立场景样本。当前版本重复求值与FP64成本可能有优化空间，未实现设想不改变本次拒绝。</Text>
      <H2>决定与后续</H2>
      <Text>按预定门槛不进入GT改善解释或训练扩展。下一项有界失败诊断覆盖全部9异常像素：真实metadata/membership/prefix、两臂系数、稳定高精度根与FP32响应，区分表示误差、条件数、乘积舍入和有限bracket。TGS原文与固定CUDA实现已补齐；sigma是屏幕锐度，不能等同GGGS沿射线厚度。完整CUDA后向仍未实现。</Text>
      <Text size="small" tone="secondary">来源：frozen_exact_render_v1、frozen_exact_render_report_v1.json、frozen_exact_assets_v1.json；2026-10-14。17源/archive、50产物、3binary、51生成文件、652原源与3原PLY核验；这是全图/计时算术重放，非独立rerender。原LaTeX原位v21编译通过。</Text>
    </Stack>}
'''.replace('FAILROWS',json.dumps(rows,ensure_ascii=False))
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
p=BASE/'canvas_frozen_exact_stage.tsx';p.write_text(text,encoding='utf-8')
g={'previous_sha256':PREVIOUS,'staged_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'destination':str(TARGET)}
(BASE/'results/canvas_frozen_exact_guard_v1.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
