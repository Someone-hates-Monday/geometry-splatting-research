"""Append measured shared-core host/device results to the existing research view."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
OLD='3c25c1eb954820ed298c257c3d333367bd03b66168dbcd248bc7b138d688d23c'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==OLD
text=raw.decode('utf-8').replace('\r\n','\n')
def replace(old,new):
 global text
 assert text.count(old)==1,old
 text=text.replace(old,new)
replace('"条件精度根证书"];','"条件精度根证书", "严格尾部CUDA组件"];')
replace('useState(19)','useState(20)')
replace('条件精度CPU根证书通过；GPU与真实质量尚待验证','CUDA严格尾部端点组件通过；设备预算与完整求根待完成')
replace('下一步GPU预算/尾部符号和完整成本，无新GT或训练扩展。','严格尾部核心CUDA212端点通过，局部event约223ms；尚待设备预算/完整求根与成本对照，无新GT或训练扩展。')
s=json.loads((BASE/'experiments/strict_interval_cuda_v1/summary.json').read_text(encoding='utf-8'));assert s['endpoint_component_gate_pass']
block='''    {tab === 20 && <Stack gap={18}>
      <H2>共享严格数学核心：host与CUDA端点通过</H2>
      <Text>CUDA12.8原文说明standard数学函数的ULP边界来自非穷尽测试、不是保证界。严格路径采用基本IEEE有向运算、range reduction和级数余项；不调用standard exp/log/log1p/expm1。host用MSVC严格浮点和动态舍入，device用CUDA double intrinsic，所有源与输入归档。</Text>
      <Table headers={["核验范围", "全部项数", "失败", "参照/限定"]} rows={[
        ["Windows scalar", "774", "0", "固定seed13023，独立Decimal"],
        ["Windows端点", "212", "0", "106有限模型，复用独立MP240世界端点"],
        ["CUDA端点", "212", "0", "同一header，CPU区间与CPU选定bracket"],
        ["精确有理常量", "ln2/tail/drop界", "0", "Fraction atanh级数及余项"]
      ]} />
      <H2>在log域保留小于IEEE标量范围的尾部</H2>
      <Text>g=exp(ell)，b=-log(1-g)/2。当ell小于−40，log b介于ell−log2和ell−log2+2⁻⁵⁶；从g小于2⁻⁵⁷及几何级数界推出。正负残差项用共同log尺度有向求和；缩放后低于exp(−700)仍保留正上界2⁻¹⁰⁰⁹。区间重叠或峰位交叉保留unresolved，不删例。</Text>
      <H2>测得开销：仍为参考组件</H2>
      <Table headers={["计时项", "值", "包括", "尚未包括"]} rows={[
        ["212端点event中位数", "222.9916504ms", "输出allocation/fill及kernel", "源参数预算、transfer、选择、求根"],
        ["设备", "RTX5090 Laptop / sm120", "CUDA12.8 gcc/g++11", "完整renderer与backward/GT"]
      ]} />
      <Text>本次设备验证使用保存的CPU coefficient区间和bracket；不是完整GPU条件精度算法。device未单独验证全部774scalar。本局部计时没有完整renderer对照，不能宣布加速。下一步便宜充分认证、设备预算/offset/mass/回退/求根，以及全部成本核验，严格核心保留作难点参照。</Text>
      <Text>之前WSL查询两次自动审批timeout没有执行；用户明确允许后，CUDA编译/运行完成。无GPU死锁证据，当前host和CUDA诊断均终态，没有新训练。原先质量与效率失败仍保留，整体科研目标未完成。</Text>
      <Text size="small" tone="secondary">来源：STRICT_INTERVAL_RESULT.md、strict_interval_host_v1、strict_interval_cuda_v1、strict_interval_audit_v1.json；2026-10-14。15+8源archive、9+5产物、9generated核验；host未执行CUDAwrapper的same-device guard源漂移明确记录。原LaTeX原位v26编译成功，静态视图仅作结果快照。</Text>
    </Stack>}
'''
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
p=BASE/'canvas_strict_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8')
guard=dict(previous_sha256=OLD,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(TARGET))
(BASE/'results/canvas_strict_guard_v1.json').write_text(json.dumps(guard,indent=2)+'\n',encoding='utf-8');print(json.dumps(guard))
