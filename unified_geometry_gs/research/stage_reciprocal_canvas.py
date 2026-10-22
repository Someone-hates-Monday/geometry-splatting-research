"""Append the audited reciprocal optimization to the existing result view."""
from pathlib import Path
import hashlib, json
BASE = Path(__file__).resolve().parent
TARGET = Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
OLD = '99421f75e32b124acb67fecb22e5cba82b24d3802391d668d6ecc887e9e224c1'
raw = TARGET.read_bytes()
assert hashlib.sha256(raw).hexdigest() == OLD
text = raw.decode('utf-8').replace('\r\n', '\n')
audit = json.loads((BASE/'results/reciprocal_root_audit_v1.json').read_text())
assert audit['status'] == 'complete' and audit['local_gate_pass']
def replace(old, new):
    global text
    assert text.count(old) == 1, old
    text = text.replace(old, new)
replace('"设备预算与自主根"];', '"设备预算与自主根", "精确倒数表优化"];')
replace('useState(21)', 'useState(22)')
replace('设备预算与自主认证根固定开发通过；效率和完整渲染待验证',
        '认证求根局部耗时减少45.4%；完整渲染仍待验证')
block = '''    {tab === 22 && <Stack gap={18}>
      <H2>单一实现优化：同一输出，局部诊断快1.83倍</H2>
      <Text>将严格级数六处整数倒数替换为CUDA constant-memory表，k=1..51。精确有理数验证实际编译表的相邻binary64下/上界，等于原IEEE directed reciprocal；区间乘法、余项、候选、预算、门槛和7322输入不改。原生产CUDA与训练源码未改。</Text>
      <Table headers={["逐位检查", "完整范围", "结果"]} rows={[
        ["control / variant预算", "7322层×2路径×26double", "全部位一致"],
        ["control / variant根", "110模型×12double，包括NaN bits", "全部位一致"],
        ["重编control / 原device_root_v1", "完整预算与根", "全部位一致"],
        ["有限根与无根", "106finite：76low/30high；4no-root", "分类/严格符号/精确宽度保持"]
      ]} />
      <H2>五对交替CUDA event：同进程对照，ms</H2>
      <Table headers={["批次", "control中位数(ms)", "variant中位数(ms)", "配对比几何均值"]} rows={[
        ["7322双路径预算", "0.403200", "0.268448", "0.681301"],
        ["110模型根与证书", "705.381042", "384.888275", "0.545663"],
        ["预算+根合并", "705.739014", "385.180573", "0.545837"]
      ]} />
      <Text>预定所有输出位一致且合并比≤0.95的局部门槛通过：合并耗时减少45.4163%，速度比1.83205。每对合并比0.545781–0.545884；不是独立训练种子或显著性检验。包含输出allocation/fill，排除host输入传输、完整renderer/backward/GT/optimizer。</Text>
      <H2>仍需完成的证据</H2>
      <Text>位一致继承旧MP240世界参照，本轮不是未观测样本确认。整数倒数缓存是数值实现优化，尚无新颖性或完整训练收益。385ms仍是高成本诊断；下一步固定普通像素便宜充分认证及严格回退协议，再核验冻结24视角的选择、初始范围、实际延迟/内存和世界VJP，通过后再扩展配对训练。</Text>
      <Text>v1仅在源码匹配计数断言处失败，未运行CUDA比较；修正r/x循环计数后v2 complete/exit0(session15185)，失败来源与记录保留。当前无新训练卡死证据，整体科研目标未完成。</Text>
      <Text size="small" tone="secondary">来源：RECIPROCAL_ROOT_RESULT.md、reciprocal_root_v2、reciprocal_root_audit_v1.json；2026-10-14。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
'''
marker = '  </Stack>;\n}'
assert text.count(marker) == 1
text = text.replace(marker, block + marker)
p = BASE/'canvas_reciprocal_stage_v1.tsx'
assert not p.exists()
p.write_text(text, encoding='utf-8')
guard = dict(previous_sha256=OLD, staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), destination=str(TARGET))
(BASE/'results/canvas_reciprocal_guard_v1.json').write_text(json.dumps(guard, indent=2)+'\n', encoding='utf-8')
print(json.dumps(guard))
