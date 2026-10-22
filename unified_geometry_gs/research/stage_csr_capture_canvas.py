"""Append verified dynamic full-image CSR capture to the existing result view."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
OLD='62c2c8911ab87c5fd29179592a3d66990741a31144865d8a0533a3d3ff4d7664'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==OLD
text=raw.decode('utf-8').replace('\r\n','\n')
a=json.loads((BASE/'results/csr_capture_audit_v1.json').read_text(encoding='utf-8'));assert a['gate_pass'] and a['checked_records']==870178922
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace('"完整图像名单容量"];','"完整图像名单容量", "动态CSR实际名单"];')
replace('useState(24)','useState(25)')
replace('完整图像证实192层容量不足；动态名单与认证仍待接入','动态实际名单全量核验通过；共享缓存与认证仍待接入')
block='''    {tab === 25 && <Stack gap={18}>
      <H2>动态CSR：完整实际ID与位置，不再用192截断</H2>
      <Text>用冻结计数二进制读首轮实际接受数，int64 prefix-sum分配offsets及int32 ID/position pairs；在原FIRST接受位置写入。检查pixel span及总容量，越写尝试置bit1，数量不符置bit2；任何非零状态不算成功。原生产源码未变。</Text>
      <Table headers={["完整范围", "核验", "结果"]} rows={[
        ["24视角 / 10,834,488 pixels", "原native图像与计数", "全部逐位一致"],
        ["870,178,922 associations", "与原initial名单数量", "精确相同"],
        ["全部关联GPU checker", "ID / 递增位置 / 前缀 / tile / point-list", "全部通过，status=0"],
        ["全部关联独立CPU", "offset差 / 每条ID与原位置 / 同pixel顺序", "870,178,922条通过，无抽样"]
      ]} />
      <Text>36来源、54产物、18生成源、binary/vendor headers/ZIP均复核。完整depth/counters/CSR和原point-list离线保存；RGB/alpha/normal/radii在运行中全部比较，未为第二次离线位审核保存数组。全部仍是此前观察的开发视角。</Text>
      <H2>错误分配保护与成本</H2>
      <Table headers={["预定guard probe", "目标pixel状态", "其他状态/原图像"]} rows={[
        ["少分配1项", "3，越写尝试被阻止且数量不符", "其他pixel=0；原图像/计数位一致"],
        ["多分配1项", "2，恰好1个sentinel未填", "其他pixel=0；原图像/计数位一致"]
      ]} />
      <Text>每视角records本体约157–383MB，offsets/status另计；6.96GB是24视角汇总，不是一次常驻。5对交替capture/count使用已分配CSR、含fill/reset，配对比1.074844：这是捕获阶段约7.48%开销。</Text>
      <Text>完整count→scan/总数读取/allocation→capture→checker单次event观察为23.680–50.505ms，同步wall23.815–49.580ms；包含host间隙，CPU传输/压缩/编译另计。没有完整流程配对性能采用试验，不代表最终认证/训练速度。</Text>
      <H2>下一项：共享源参数缓存与射线内并行</H2>
      <Text>原世界因子/误差预算按primitive共享，减少870M关联中的重复构建；用动态名单求候选/证书，保留原物理根与初始范围语义，明确unresolved及公共F32输出舍入。世界VJP、完整成本、跨场景训练、留出RGB/几何及新颖性仍未完成。</Text>
      <Text size="small" tone="secondary">来源：CSR_CAPTURE_RESULT.md、csr_capture_v1、csr_capture_audit_v1.json；2026-10-14。主批session19995与CPU session95619均exit0。现有LaTeX原位更新；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
'''
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
p=BASE/'canvas_csr_capture_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8')
g=dict(previous_sha256=OLD,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(TARGET))
(BASE/'results/canvas_csr_capture_guard_v1.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
