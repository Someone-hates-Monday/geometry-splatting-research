from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent;TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx');OLD='17260ee2d585ed1df1ff035ae9f8c9d351a198efc8b7d570d5b23c6834b9cd3a'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==OLD;text=raw.decode('utf-8').replace('\r\n','\n')
s=json.loads((BASE/'experiments/dynamic_root_v1/summary.json').read_text(encoding='utf-8'));a=json.loads((BASE/'results/dynamic_root_audit_v1.json').read_text(encoding='utf-8'));rare=json.loads((BASE/'results/dynamic_root_rare_audit_v1.json').read_text(encoding='utf-8'));assert a['gate_pass'] and rare['gate_pass']
def replace(old,new):
    global text
    assert text.count(old)==1,old;text=text.replace(old,new)
replace('"F32原深度证书"];','"F32原深度证书", "动态严格求根"];');replace('useState(28)','useState(29)');replace('原有效深度93.85%获物理端点证书；未认证探针均需修正，完整方法待验证','全部未认证原有效射线完成修正或无根判定；世界梯度与GT仍待验证')
rows=[]
for r in s['rows']:
    ms=sum(sorted([m['first_root_time']['event_ms']]+[x['event_ms'] for x in m['repeat_root_times']])[1] for m in r['modes']);rows.append([f"scan{r['scan']}/{r['camera']}",f"{r['certified']:,}",str(r['no_finite_root']),str(r['finite_high']),f"{ms:.3f}"])
block='''    {tab === 29 && <Stack gap={18}>
      <H2>562,079个认证修正，256个数学无根，剩余未解决0</H2>
      <Text>原24开发视角全部562,335未认证原有效射线、28,196,963源关联处理完成。原完整贡献集合/query/AMP保持，动态int64分段与每warp一射线；320条超过旧192容量，此子集最多372节点。原8,583,586已认证像素未触及，原无候选/不eligible的1,688,567未填补。</Text>
      <Text>稳定prefix/energy shift与scaled tails生成double候选，转为公共F32 z后围绕真实query-norm中心认证。低几何562,050、高几何29；F32产品证书562,053、short scaled-log26。strict回退执行但无最终接受需要该最后分支。所有认证公共深度为正。</Text>
      <H2>独立固定射线与全部稀有分支</H2>
      <Table headers={["审计", "完整范围与结果"]} rows={[
        ["固定891条完整射线 / 51,524节点", "MP80/140 world inverse，206,096系数检查 + MP240 scaled端点通过"],
        ["原9异常", "全部修正，公共F32深度误差带认证通过"],
        ["全部无根/高精度/short并集299射线", "5,920节点 / 23,680系数与所有端点通过"],
        ["全部256无根", "真实beta>0.5，最小超额约4.51860e-5"],
        ["来源/产物/CSR", "主30来源/74产物，补充9来源/16产物，条件选择/AMP/query/root位核查通过"]
      ]} />
      <Text>固定探针是原132未认证探针加每view32等距未认证像素，没有包含无根例；因此另查全部稀有分支，不能用前一个gate冒充该分支证据。数学无根只属于既定物理Gaussian模型与固定名单，不代表GT/物体无表面。全部开发视角与条件子集不是随机留出集。</Text>
      <H2>成本全保留，尚无端到端加速</H2>
      <Text>root的每view三次观测中位数按low/high求和为5.822–204.906ms；decode单次event和1.625–3258.099ms，首view高值原样保留。cache build/CPU compact/原候选证书/CSR capture/full renderer另计；最大增加GPU allocation约774MB，已有驻留另计。下表仅root诊断中位数，单位ms。</Text>
      <Table headers={["场景/相机", "认证修正", "数学无根", "高模式", "root诊断（ms）"]} rows={ROWS} />
      <Text>全局peak/reach区间不同于旧初始数值区间策略，贡献集合与物理T=0.5目标保持。严格根证书不自动证明世界VJP精度、真实重建质量或学术新颖性。下一步用既存源与内部根验证条件数/链式梯度，并做固定模型完整GT对照，再决定训练采纳。</Text>
      <Text size="small" tone="secondary">来源：DYNAMIC_ROOT_RESULT.md / dynamic_root_v1 / dynamic_root_audit_v1 / dynamic_root_rare_audit_v1，2026-10-15 Asia/Shanghai。主session88413、独立23351、稀有36155全部exit0。原LaTeX原位编译；本静态视图仅语法检查，SDK/runtime未验证。</Text>
    </Stack>}
'''.replace('ROWS',json.dumps(rows,ensure_ascii=False));marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker);p=BASE/'canvas_dynamic_root_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8');g=dict(previous_sha256=OLD,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(TARGET));(BASE/'results/canvas_dynamic_root_guard_v1.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
