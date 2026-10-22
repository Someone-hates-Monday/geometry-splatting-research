"""Append verified device budgets and autonomous root development to existing view."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
OLD='9f75007e2fa4e29d071bd04961b273fb826df768e9f3b691dde3722de7dc4b1a'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==OLD
text=raw.decode('utf-8').replace('\r\n','\n')
def replace(old,new):
 global text
 assert text.count(old)==1,old
 text=text.replace(old,new)
replace('"严格尾部CUDA组件"];','"严格尾部CUDA组件", "设备预算与自主根"];')
replace('useState(20)','useState(21)')
replace('CUDA严格尾部端点组件通过；设备预算与完整求根待完成','设备预算与自主认证根固定开发通过；效率和完整渲染待验证')
replace('尚待设备预算/完整求根与成本对照，无新GT或训练扩展。','设备7322预算/110mass/自主106根与4无根已通过，合并event约706ms；完整renderer/backward/GT与效率待验证，无新训练扩展。')
s=json.loads((BASE/'experiments/device_root_v1/summary.json').read_text(encoding='utf-8'));assert s['fixed_device_root_gate_pass']
block='''    {tab === 21 && <Stack gap={18}>
      <H2>从原参数到认证根：全部在设备，固定开发通过</H2>
      <Text>输入只有原mean/scales/quaternion或R、camera affine、query、amplitude和110模型start/count。真正F逆用有向算术，two-pass modified Gram–Schmidt提供名义Q/U，polar/residual预算覆盖缓存与构造误差。设备求根未输入CPU候选、bracket或MP根；参照只在返回后核验。</Text>
      <Table headers={["阶段", "完整固定范围", "结果"]} rows={[
        ["设备预算", "7322层×2路径×4系数=58576", "全部参照包围，零失败"],
        ["设备mass", "110模型×2=220", "106finite/4no-root分类正确"],
        ["旧CPU bracket端点组件", "106finite模型", "low79/high106认证"],
        ["设备自主候选/新bracket", "106finite+4no-root", "finite76low/30high，4分类正确"],
        ["自主根max误差", "camera-ray单位", "1.361838888e−5，门槛1e−3不改"]
      ]} />
      <Text>候选可用标准math函数，但输出必须由严格基本算术/级数/log-tail给两端符号。低系数转F32，仅作候选；未认证则高路径读取amplitude/energy并重新求根。tau0严格小于1/1000，向内舍入保证精确宽度。已观测fixed seed属于开发，不再称独立新确认或真实像素发生率。</Text>
      <H2>独立世界复核与局部成本</H2>
      <Text>MP240 covariance逆重算106个新设备端点对，4408使用中系数、scaled包围、精确有理宽度和4无根通过。56+14源/archive、9+7产物、10+11generated校验。原生产CUDA与训练未变。</Text>
      <Table headers={["自主根运行的CUDA event中位数", "规模", "ms"]} rows={[
        ["原参数预算", "7322双路径", ".441504"],
        ["求根与证书", "110数学模型", "705.173584"],
        ["两阶段合并", "上述固定输入", "705.568237"]
      ]} />
      <Text>包括输出allocation/fill与相应kernel，未包括host transfer或完整renderer，诊断26double/路径输出不代表56B生产cache的布局实测。成本尚需降低，未证明加速。正确forward也不说明临界薄层opacity梯度可表示或训练会改善。</Text>
      <H2>下一项：便宜充分认证与完整成本</H2>
      <Text>先让健康情况用便宜充分界，严格核心作难点参照；保持相同物理根和全部临界输入。然后核验完整24视角选择/初始范围/成本与世界backward，通过后再跨场景训练、held-out图像与几何评价。当前诊断均终态，无新训练；整体科研目标未完成。</Text>
      <Text size="small" tone="secondary">来源：DEVICE_ROOT_RESULT.md、device_budget_v1、device_root_v1、device_root_audit_v1.json；2026-10-14。原LaTeX原位v27编译成功；本视图静态，语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
'''
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
p=BASE/'canvas_device_root_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8')
g=dict(previous_sha256=OLD,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(TARGET))
(BASE/'results/canvas_device_root_guard_v1.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
