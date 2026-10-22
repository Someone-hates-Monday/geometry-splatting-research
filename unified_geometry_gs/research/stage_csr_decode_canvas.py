"""Append complete actual CSR decode/probe/cost evidence to the existing view."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
OLD='4ab7ed3b28bb2accc02aec715580da8495a6b9a34fa5975c0de1bc61818af74e'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==OLD
text=raw.decode('utf-8').replace('\r\n','\n')
a=json.loads((BASE/'results/csr_decode_audit_v1.json').read_text(encoding='utf-8'));assert a['gate_pass'] and a['records_accounted']==870178922
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace('"共享几何缓存"];','"共享几何缓存", "实际CSR流式解码"];')
replace('useState(26)','useState(27)')
replace('共享几何缓存保持全部固定输出位；真实射线解码与认证待接入','真实CSR全量解码与固定探针通过；诊断成本过高，根证书待实现')
block='''    {tab === 27 && <Stack gap={18}>
      <H2>原内核实际query与AMP，全量CSR解码完成</H2>
      <Text>只在原kernel新增pixnf.x/y输出，第三分量保持1；幅值来自实际conic_opacity.w，未假定等于输入opacity。原图像/counters/radii位不变，9异常query位均匹配。source/model/camera及每pixel数量与原CSR相同。</Text>
      <Table headers={["范围", "完整核验", "结果"]} rows={[
        ["24视角 / 870,178,922 records", "batch65536 / int64 pixel恢复 / 不截断", "全部processed闭合"],
        ["两mode全部record", "status与有限性", "全部status0，非有限0"],
        ["1,924预定探针", "stream/cache / 原直接preprocess", "全数组位一致"],
        ["15,392系数界", "独立MP80/140 world inverse", "peak/rho/E/alpha全过"]
      ]} />
      <Text>探针为每view48等距record加此前全部9异常射线关联，没有按结果选样。39来源、74产物、生成query/strict源、binary/ZIP复核。不是870M逐条MP验证或每像素根证书；仍属已观测开发输入。</Text>
      <H2>诊断成本和名义下溢必须保留</H2>
      <Text>各view decode event总和597.529–2164.430ms（约0.60–2.16秒），含temporary allocation/fill与kernel，CPU transfer/hash/statistics、capture/build另计。双路径26double全算是昂贵诊断，不采用为完整方法速度或训练收益。</Text>
      <Table headers={["诊断项", "完整记录"]} rows={[
        ["单batch最大输出", "27,262,976B；cache/CSR/query另有驻留"],
        ["每mode名义double alpha=0", "21,577；所有record保留，严格区间有效"],
        ["批输出留存", "SHA/状态/时序全存；约362GB double流未永久存"]
      ]} />
      <Text>名义double AMP×exp(-E_nom/2)=0不证明数学零，统计还未包含F32 candidate转换的信息损失。未保存完整double流，离线可核查hash清单但不能重算所有批hash。不能把probe通过或状态0放大成根/梯度保证。</Text>
      <H2>下一项：便宜充分条件、选择性回退与并行根</H2>
      <Text>在这批真实输入固定普通像素的充分认证，严格解码作为回退参照，使用紧凑输出和射线内并行。保留原贡献集合/物理根/初始范围语义，明确unresolved及公共F32输出舍入。世界VJP、完整GT与跨场景训练/留出图像、新颖性仍未完成。</Text>
      <Text size="small" tone="secondary">来源：CSR_DECODE_RESULT.md、csr_decode_v1、csr_decode_audit_v1.json；独立审核2026-10-15 Asia/Shanghai。主批session35967与独立审核exit0。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
'''
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
p=BASE/'canvas_csr_decode_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8')
g=dict(previous_sha256=OLD,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(TARGET))
(BASE/'results/canvas_csr_decode_guard_v1.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
