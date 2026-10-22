"""Append audited shared-cache parity and real build costs to existing view."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
OLD='d1a060e5f96546dedbabbf69ebcb0d511278c3231d7277a511ed276e28261344'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==OLD
text=raw.decode('utf-8').replace('\r\n','\n')
a=json.loads((BASE/'results/shared_cache_audit_v1.json').read_text(encoding='utf-8'));assert a['all_saved_budget_and_root_bits_match'] and a['local_reuse_gate_pass']
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace('"动态CSR实际名单"];','"动态CSR实际名单", "共享几何缓存"];')
replace('useState(25)','useState(26)')
replace('动态实际名单全量核验通过；共享缓存与认证仍待接入','共享几何缓存保持全部固定输出位；真实射线解码与认证待接入')
block='''    {tab === 26 && <Stack gap={18}>
      <H2>query之前按primitive缓存；query公式完整保留</H2>
      <Text>原inverse/QR/polar/residual与mean预算拆入共享cache；保留真实mean区间以证明aligned分支。low六项U是F32精确提升，high为F64，eta/nu向上F32。query与AMP只在decode输入，不改物理模型、余项、候选、mass、容差或原生产代码。</Text>
      <Table headers={["固定完整检查", "范围", "结果"]} rows={[
        ["全部预算 / 原archive", "7322×2×26double", "完整位一致"],
        ["根/证书", "106finite+4no-root", "含NaN bits全部一致"],
        ["八次query重用", "58576项", "完整预算位一致"],
        ["独立审核", "25来源 / 28产物 / 12generated", "原query代码/缓存字段/数组/hash均过"]
      ]} />
      <H2>固定5对重用计时，CUDA event ms</H2>
      <Table headers={["阶段", "中位数(ms)"]} rows={[
        ["build7322", "0.081056"],
        ["decode58576", "1.160480"],
        ["original repeat8", "1.481312"],
        ["build once+decode", "1.231136"]
      ]} />
      <Text>配对比几何均值0.830169，约16.9831%时间减少；原位一致且比≤0.95局部门槛通过。包括输出allocation/fill和ID范围检查/sync，不含host输入传输或完整renderer/training。重复query不是新场景，继承旧MP240参照而非新增确认。</Text>
      <H2>完整24camera的全部基元cache构造</H2>
      <Table headers={["场景", "P", "每camera cache bytes"]} rows={[
        ["scan37", "1,010,226", "226,290,624"],
        ["scan55", "1,052,385", "235,734,240"],
        ["scan63", "510,660", "114,387,840"]
      ]} />
      <Text>全部24视角两mode status=0、cache有限；相机/PLY/activated输入身份与原CSR一致，不按visibility删primitive。真实build event中位数3.343–37.605ms，全部高值保留。224B/row为诊断格式，另有288B/source row，不是此前56B拟定生产布局。</Text>
      <Text>真实cache hash/eta统计在运行中记录，大数组未永久存为第二次离线/MP全量审核。真实build尚未解码全部870M关联，更不是完整图像根证书。下一项按需CSR解码、射线内并行求根，保留原range语义并明确unresolved/F32输出舍入；世界VJP、完整成本、GT与训练仍未完成。</Text>
      <Text size="small" tone="secondary">来源：SHARED_CACHE_RESULT.md、shared_cache_v1、shared_cache_audit_v1.json；2026-10-14。session55325 exit0。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
'''
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
p=BASE/'canvas_shared_cache_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8')
g=dict(previous_sha256=OLD,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(TARGET))
(BASE/'results/canvas_shared_cache_guard_v1.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
