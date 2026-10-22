from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;t=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx');old='6cd96658cab318c6ca60144c1f4c07eb99116b5727e175190c124126c7a947b0';raw=t.read_bytes();assert hashlib.sha256(raw).hexdigest()==old;text=raw.decode('utf-8').replace('\r\n','\n');a=json.loads((R/'results/frozen_certified_mesh_audit_v1.json').read_text(encoding='utf-8'));assert a['audit_gate_pass'] and not a['quality_gate_pass']
def replace(x,y):
    global text
    assert text.count(x)==1,x;text=text.replace(x,y)
replace('"世界参数链VJP"];','"世界参数链VJP", "完整49视角GT"];');replace('useState(30)','useState(31)');replace('CPU连续模型的完整参数链VJP细化通过；完整CUDA与GT待验证','完整147视角GT质量门槛失败；数值修正保留作参照，目标未完成')
rows=[[str(r['scan']),f"{r['means']['control']['overall']:.10f}",f"{r['means']['certified']['overall']:.10f}",f"{r['s2d_delta']:+.9f}"] for r in a['scenes']]
block='''    {tab === 31 && <Stack gap={18}>
      <H2>完整147视角：数值通过，重建质量门槛失败</H2>
      <Text>原scan37/55/63各完整49视角、66,361,239像素、5,346,733,664接受关联。每视角与原renderer的RGB/alpha/normal/radii/depth/slope位一致；CSR write/check与cache状态通过。已有证书52,932,318；3,661,825待修正中3,659,782认证修正、2,043数学无根、未解决0。</Text>
      <Text>同一白背景RGB、gt_mask、相机/视角顺序、TSDF(.002/1/8/trunc4/block50000)与largest cluster1。官方camera索引0000--0048，两支共享同一alignment。每scene/branch seed0/1/2，共18评估；评估采样重复不等于独立训练重复或置信区间。</Text>
      <H2>三seed均值，官方DTU毫米</H2>
      <Table headers={["场景", "原overall（mm）", "修正overall（mm）", "s2d修正−原（mm）"]} rows={ROWS} />
      <Text>三scene平均overall比0.9999937549，仅约0.0006245%下降，未达1%预设门槛。s2d差值三个均正，也全未过≤0门槛。不能把这些微小变化称为显著改善；数值正确不自动保证几何质量。</Text>
      <H2>独立产物审核与覆盖</H2>
      <Text>97来源、实际binary、147全部数组/掩码/保留像素与证书端点计数、6套mesh文件SHA和全部18评估源/结果核对通过。掩码后两支有效像素均24,152,107；无根置0没有改变此覆盖数。完整关联计数不是53亿逐节点MP核验；仍是此前开发场景。</Text>
      <Text>三scene wall约516/299/370s，含render/capture/修正、存储、CPU TSDF/extract/eval等待；单view总wall之和309.54s。单GPU核不是整体成本，不声称加速。原模型/生产源码保持，此后处理保留作数值参照，不作为已验证质量优化或训练采纳。</Text>
      <H2>后续候选单独核验</H2>
      <Text>平方历史差目标及有界伴随/C1候选推导只保存为研究计划：需独立证明、MP/中心差分、真实射线诊断、CUDA链路后才考虑训练。对称平坦间隙可使其梯度指数小的竞争反例也已记录；目前没有该新目标的性能或新颖性结论。</Text>
      <Text size="small" tone="secondary">FROZEN_CERTIFIED_MESH_RESULT.md / frozen_certified_mesh_v1 / frozen_certified_mesh_audit_v1；主session33905、独立29600 exit0。日期2026-10-15 Asia/Shanghai。原LaTeX原位编译；此静态看板仅语法检查，SDK/runtime未验证。</Text>
    </Stack>}
'''.replace('ROWS',json.dumps(rows,ensure_ascii=False));marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker);p=R/'canvas_frozen_certified_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8');g=dict(previous_sha256=old,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(t));(R/'results/canvas_frozen_certified_guard_v1.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
