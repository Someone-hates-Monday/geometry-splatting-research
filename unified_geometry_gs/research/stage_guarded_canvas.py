"""Append the independently checked CPU guard result to the existing research view."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json

BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
PREVIOUS='677dbc23d18227ca2f0f3b70f4dcc7cde2ff30386ff1ca0734174beeb2045912'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==PREVIOUS
text=raw.decode('utf-8').replace('\r\n','\n')
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)

replace('"一致因子与根别名"];','"一致因子与根别名", "条件精度根证书"];')
replace('useState(18)','useState(19)')
replace('一致QR候选拒绝；F32射线系数的根别名证书完成','条件精度CPU根证书通过；GPU与真实质量尚待验证')
replace('下一步条件精度与质量裕量，当前无GT或训练扩展。','条件精度CPU全7010区间与106有限根/4无根通过，30根触发回退；下一步GPU预算/尾部符号和完整成本，无新GT或训练扩展。')
rootrows=json.loads((BASE/'experiments/guarded_precision_v2/root_rows.json').read_text(encoding='utf-8'))
audit=json.loads((BASE/'results/guarded_precision_audit_v1.json').read_text(encoding='utf-8'));assert audit['status']=='complete' and not audit['failures']
data=defaultdict(list)
for r in rootrows['old']:data['旧真实9' if r['name'].startswith('outlier_') else '旧临界9'].append(r)
names={'ordinary':'新普通四层','near_threshold':'新近阈值双层','exact_critical_two_layer':'新精确临界双层','tiny_offset':'新微小偏移','no_root':'新无有限根'}
for r in rootrows['fresh']:data[names[r['kind']]].append(r)
rows=[]
for group,items in data.items():
    error=max([float(r['absolute_error']) for r in items if r['absolute_error'] is not None],default=None)
    rows.append([group,str(len(items)),str(sum(not r['fallback'] for r in items)),str(sum(r['fallback'] for r in items)),str(sum(not r['gate_pass'] for r in items)),'无有限根' if error is None else f'{error:.6g}'])
block='''    {tab === 19 && <Stack gap={18}>
      <H2>条件精度：先验证数值证书，再进入GPU</H2>
      <Text>缓存仍由一致QR构造，增加可计算的U/mean误差预算；名义56B/基元尚未验证GPU对齐。低精度求根仅提出候选，两侧残差区间给严格符号才认证；否则读取原参数回退，仍需同一证书。区间预算和候选都不读取MP真根。</Text>
      <Table headers={["固定集合", "模型数", "直接认证", "回退", "未过门槛", "max根误差(camera-ray)"]} rows={ROWS} />
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
'''.replace('ROWS',json.dumps(rows,ensure_ascii=False))
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
stage=BASE/'canvas_guarded_stage_v1.tsx';assert not stage.exists();stage.write_text(text,encoding='utf-8')
guard=dict(created_utc=datetime.now(timezone.utc).isoformat(),previous_sha256=PREVIOUS,staged_sha256=hashlib.sha256(stage.read_bytes()).hexdigest(),destination=str(TARGET),rows=rows)
(BASE/'results/canvas_guarded_guard_v1.json').write_text(json.dumps(guard,indent=2)+'\n',encoding='utf-8');print(json.dumps(guard))
