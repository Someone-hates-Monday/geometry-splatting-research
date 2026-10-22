"""Append short certificate/fallback evidence to the existing research view."""
from pathlib import Path
import hashlib, json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
OLD='a405ef4faa4eceb66515498a8f21da6cf3768ec4953dcd8a7e068b294a9d7357'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==OLD
text=raw.decode('utf-8').replace('\r\n','\n')
assert json.loads((BASE/'results/short_certificate_audit_v1.json').read_text())['local_adoption_gate_pass']
f=json.loads((BASE/'experiments/short_fallback_v1/summary.json').read_text())
assert f['no_false_certificate_gate_pass'] and f['strict_success_branch_covered']
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace('"精确倒数表优化"];','"精确倒数表优化", "短证书与回退"];')
replace('useState(22)','useState(23)')
replace('认证求根局部耗时减少45.4%；完整渲染仍待验证','短证书再减少局部耗时36.2%；完整渲染和质量仍待验证')
block='''    {tab === 23 && <Stack gap={18}>
      <H2>便宜的充分认证：同一物理根，失败仍走严格路径</H2>
      <Text>固定log九项、exp十二阶、小范围log1p/expm1八阶。余项由有向IEEE基本运算包围；两端严格相反符号才short返回。否则同候选调用原长证书，再按原逻辑进入high；仍不能认证保留unresolved。参数预算、候选、mass、容差和全部固定输入不变，原生产CUDA/训练未修改。</Text>
      <Table headers={["主批完整核验", "范围", "结果"]} rows={[
        ["预算/重编control", "7322层，完整输出位", "与旧倒数表一致"],
        ["独立scalar", "774保存输入，Decimal240+MP240", "包围全过"],
        ["有限根", "106：76low/30high", "最终106 short；strict曾调用30次"],
        ["世界逆/端点", "106对端点、4408使用系数、4无根", "MP240及精确宽度全过"]
      ]} />
      <H2>五对交替CUDA event：倒数表control到short，ms</H2>
      <Table headers={["批次", "control中位数(ms)", "variant中位数(ms)", "配对比几何均值"]} rows={[
        ["110模型根与证书", "384.782196", "245.457794", "0.637784"],
        ["预算+根合并", "385.132538", "245.741669", "0.638155"]
      ]} />
      <Text>合并耗时减少36.1845%，速度比1.56702；预定全数值通过且比≤0.95的局部门槛通过。含输出allocation/fill，不含host transfer/完整renderer/backward/GT/训练。此前705→385与此次385→246属于不同运行，不能称同进程三臂比较。主批全是已观测开发输入。</Text>
      <H2>另一个固定平坦族：严格成功与未解分支都已实测</H2>
      <Text>单Gaussian amplitude=3/4、peak=1、scale=10^j，j=0..7，解析median=1。只加载主批冻结二进制，解析根不输入设备；这8例不混入主批门槛或计时。</Text>
      <Table headers={["scale", "control", "hybrid"]} rows={[
        ["1、10、100", "3 certified", "3 short"],
        ["1000", "certified", "strict成功救回"],
        ["10000至10000000", "4 unresolved", "4 unresolved"]
      ]} />
      <Text>全部返回证书均过独立MP240单基元公式、scaled包围、精确宽度，无误报成功。未解案例保留，完整renderer必须明确处理；不代表真实像素发生率或统一可认证保证。标准级数界暂不主张学术新颖性。</Text>
      <H2>下一项：完整渲染的可扩展成本与反向门槛</H2>
      <Text>固定24视角的选择/生产初始范围和成本协议，检查缓存与射线内并行计算；独立世界VJP通过后才扩展跨场景配对训练、留出图像及几何评价。极平坦族的精度限制和此前质量失败仍在，整体科研目标未完成。</Text>
      <Text size="small" tone="secondary">来源：SHORT_CERTIFICATE_RESULT.md、short_certificate_v1、short_certificate_audit_v1.json、short_fallback_v1；2026-10-14。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
'''
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
p=BASE/'canvas_short_certificate_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8')
g=dict(previous_sha256=OLD,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(TARGET))
(BASE/'results/canvas_short_certificate_guard_v1.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
