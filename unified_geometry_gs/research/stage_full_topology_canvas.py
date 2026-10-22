"""Append complete-image actual topology evidence to the existing audit view."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
OLD='c56cd7bd2a3e670e1976c94e38757ec52556bca909f8f8cb99826ff200b50dac'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==OLD
text=raw.decode('utf-8').replace('\r\n','\n')
a=json.loads((BASE/'results/full_topology_audit_v1.json').read_text(encoding='utf-8'));assert a['all24_views'] and a['source_isolation_pass']
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace('"短证书与回退"];','"短证书与回退", "完整图像名单容量"];')
replace('useState(23)','useState(24)')
replace('短证书再减少局部耗时36.2%；完整渲染和质量仍待验证','完整图像证实192层容量不足；动态名单与认证仍待接入')
block='''    {tab === 24 && <Stack gap={18}>
      <H2>全部24原分辨率视角：容量192不能用于完整图像</H2>
      <Text>scan37/55/63各8个既定开发视角，共10,834,488像素。只在原affine内核真正RGB接受及median第一次迭代接受位置计数，无容量截断或抽样。新的同源control/计数版本保留全部原RGB/alpha/normal/depth/slope/radii位；原invalid depth=0不变，RGB与median接受数在initial eligible中零不一致。</Text>
      <Table headers={["全部实际名单分布", "initial eligible", "finally valid"]} rows={[
        ["像素数", "9,165,957", "9,145,921"],
        ["min / p50 / p90 / p99 / max", "3 / 88 / 155 / 234 / 461", "3 / 88 / 155 / 234 / 461"],
        ["超过192", "334,083 (3.644824%)", "334,072 (3.652688%)"],
        ["超过256", "41,536 (0.453155%)", "41,536 (0.454148%)"],
        ["超过512", "0", "0"]
      ]} />
      <Text>512覆盖本批并不证明通用容量。后续名单按真实计数动态分配，逐像素检查写入数；不能删去超容量像素后声称全图认证。initial与finally valid分别保留，约870M primitive-ray关联是全部24视角汇总，不是单视角同时常驻内存。该固定名单未观察到非ball贡献项。</Text>
      <H2>容量尾部有场景差异</H2>
      <Table headers={["场景(各8视角)", "initial max", "finally valid超过192像素数"]} rows={[
        ["scan37", "444", "153,403"],
        ["scan55", "382", "33,951"],
        ["scan63", "461", "146,718"]
      ]} />
      <Text>scan63/0041的valid超过192比例10.6250%，不能只看全图平均比例。独立CPU审核669来源、50产物、两臂各16生成源、实际binary/vendor header/ZIP，重算全部保存depth/slope位、计数及直方图；RGB/alpha/normal/radii在运行中完整比较，未留存第二次离线位审核数组。</Text>
      <H2>计数开销与下一步</H2>
      <Text>每视角3次预热、5对交替native render，event配对比的视角几何均值1.025855，约2.59%计数开销。control中位数8.235–21.230ms，instrumented8.521–21.301ms；新增peak-added allocated约6.69–7.95MB。它们与110模型245.7ms不是同一工作量，不构成方法速度比。</Text>
      <Text>下一项捕获动态CSR实际primitive-ID名单，逐像素验证数量与顺序，再接共享参数缓存、射线内并行认证/回退；明确unresolved政策和世界VJP后才扩大训练。当前没有完整认证renderer、GT改善、训练收益或新颖性结论，整体目标仍未完成。</Text>
      <Text size="small" tone="secondary">来源：FULL_TOPOLOGY_RESULT.md、full_topology_v1、full_topology_audit_v1.json；2026-10-14。首个离线审核因默认GBK读UTF8源失败，记录保留，显式UTF8后通过。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
'''
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
p=BASE/'canvas_full_topology_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8')
guard=dict(previous_sha256=OLD,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(TARGET))
(BASE/'results/canvas_full_topology_guard_v1.json').write_text(json.dumps(guard,indent=2)+'\n',encoding='utf-8');print(json.dumps(guard))
