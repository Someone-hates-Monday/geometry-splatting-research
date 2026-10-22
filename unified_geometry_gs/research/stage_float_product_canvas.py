from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
OLD='110a53d0ac0f4548e86604c5adc573a01a22d7ec7e8919b8ea82d06ac5191201'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==OLD;text=raw.decode('utf-8').replace('\r\n','\n')
a=json.loads((BASE/'results/float_product_audit_v1.json').read_text(encoding='utf-8'));assert a['gate_pass'] and a['coefficient_checks']==583436
s=json.loads((BASE/'experiments/float_product_v1/summary.json').read_text(encoding='utf-8'));f=json.loads((BASE/'results/float_product_failure_analysis_v1.json').read_text(encoding='utf-8'))
def replace(old,new):
    global text
    assert text.count(old)==1,old;text=text.replace(old,new)
replace('"实际CSR流式解码"];','"实际CSR流式解码", "F32原深度证书"];')
replace('useState(27)','useState(28)')
replace('真实CSR全量解码与固定探针通过；诊断成本过高，根证书待实现','原有效深度93.85%获物理端点证书；未认证探针均需修正，完整方法待验证')
rows=[]
for r in s['rows']:
    ms=sorted(t['event_ms'] for t in r['times'])[1];rows.append([f"scan{r['scan']} / {r['camera']}",f"{100*r['certified']/r['original_valid']:.3f}%",f"{r['unresolved']:,}",f"{ms:.3f} ms"])
block='''    {tab === 28 && <Stack gap={18}>
      <H2>公共F32深度：93.85%通过充分条件，剩余像素保留</H2>
      <Text>24开发视角全部10,834,488像素 / 870,178,922关联核对。原有效深度9,145,921，认证8,583,586，未认证562,335；无原候选/不eligible的1,688,567不等于数学无根。bad输入标记0。每warp完整CSR射线，不截断192节点，输出12double全部保存。</Text>
      <Text>真实射线中心是公共F32 z×实际query长度，以有向double包络，再设置严格小于1/1000的左右误差端点。只有左T下界&gt;0.5、右T上界&lt;0.5才认证；原低模式double几何保持，alpha/端点用F32有向exp8与严格余项。FTZ关闭，未知峰侧取包络，失败返回unresolved。</Text>
      <H2>完整固定射线的独立世界数学检查</H2>
      <Table headers={["范围", "独立证据"]} rows={[
        ["4,104 scalar + 有理算术", "exp8/ln2/尾部/reciprocal/tau全部通过"],
        ["1,545完整探针射线", "每view64等距eligible + 原9异常，选择独立于认证结果"],
        ["145,859节点 / 583,436系数界", "MP80/140世界covariance逆，p/rho/E/alpha和端点T通过"],
        ["1,410认证探针", "真实严格符号与公共z舍入界通过"],
        ["25来源 / 76产物", "源隔离、实际binary/生成源/ZIP与全输出计数复核"]
      ]} />
      <Text>上述固定探针不是全部870M逐条MP证明。误差是既定物理Gaussian模型的camera-ray参数，不是GT毫米重建误差，也没有证明新颖性或训练收益。</Text>
      <H2>未认证探针中，原候选确实需要修正</H2>
      <Text>探索性跟进覆盖固定集合内全部132未认证射线：MP80/140在公共中心±精确1/1000判断18根在左侧、114在右侧，原9异常为2左7右。该子集没有仅因界太宽拒绝的例子；按未认证条件选样，不能推算562,335像素总体错误率。</Text>
      <H2>每视角覆盖与诊断成本</H2>
      <Text size="small" tone="secondary">来源：float_product_v1（2026-10-15 Asia/Shanghai）。覆盖分母为原有效深度；延迟是3次重复CUDA event中位数，含紧凑output allocation/fill与kernel，cache build/CSR/query驻留另计。不同诊断职责，不声称端到端加速。</Text>
      <Table headers={["场景 / 相机", "认证 / 原有效", "未认证像素", "诊断延迟（ms）"]} rows={ROWS} />
      <Text>57.714–134.755ms低于上轮0.60–2.16s double诊断，但仍高于原renderer8.235–21.230ms。下一步是动态长度、射线内并行的选择性strict根修正；世界VJP、完整GT、留出训练和新颖性审查未完成。当前训练与GPU诊断均已退出。</Text>
      <Text size="small" tone="secondary">主批session26752、独立session10530均exit0；FLOAT_PRODUCT_RESULT.md / float_product_audit_v1.json / float_product_failure_analysis_v1.json。原LaTeX已原位编译。静态视图语法检查不替代SDK/runtime核验。</Text>
    </Stack>}
'''.replace('ROWS',json.dumps(rows,ensure_ascii=False))
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
p=BASE/'canvas_float_product_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8')
g=dict(previous_sha256=OLD,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(TARGET));(BASE/'results/canvas_float_product_guard_v1.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
