"""Preserve existing canvas pages and append fixed QR rejection and alias theorem."""
from pathlib import Path
import hashlib,json

BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
PREVIOUS='0b4fc82155ef0f970cbe8218ad75b78008aee031d103146fc60b544976bd558d'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==PREVIOUS
text=raw.decode('utf-8').replace('\r\n','\n')


def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)


replace('"有效alpha与世界参照"];','"有效alpha与世界参照", "一致因子与根别名"];')
replace('useState(17)','useState(18)')
replace('72混合根定位有效alpha差；世界参照完成','一致QR候选拒绝；F32射线系数的根别名证书完成')
replace('当前不扩大GT或训练，下一步一致能量构造与薄层裕量理论。',
        '一致QR主候选真实9点改进，但12局部构造/5临界根失败；合法F32世界在正确系数舍入后别名，接口统一误差下界超过.488。下一步条件精度与质量裕量，当前无GT或训练扩展。')
s=json.loads((BASE/'experiments/coherent_factor_v2/summary.json').read_text(encoding='utf-8'))
rows=[[g['group'],str(g['inputs'])]+[str(r['bad']) for r in g['variants']] for g in s['groups']]
block='''    {tab === 18 && <Stack gap={18}>
      <H2>一致构造改进真实异常；固定主候选仍被拒绝</H2>
      <Text>主候选QR_mixed以真正R逆及camera仿射A逆构造precision factor，QR三角U统一给peak/rho/energy。6F32 U+3F64 mean共48B；全F32九项36B、全F64九项72B为预定对照。均先double解码再将(p,rho,alpha)转F32；没有因结果换主候选。</Text>
      <Table headers={["固定组", "全部项数", "QR32局部失败", "mixed局部失败", "QR64局部失败"]} rows={ROWS} />
      <Table headers={["根比较（camera-ray单位）", "QR32", "QR_mixed", "QR64"]} rows={[
        ["9真实异常max误差", "3.89654e−4", "4.99316e−5", "5.61782e−6"],
        ["9真实异常>1e−3", "0", "0", "0"],
        ["9临界双层>1e−3", "5", "5", "4"],
        ["临界max误差", ".48882027", ".48882027", ".48882027"]
      ]} />
      <Text>全7010项保留，primary整体FAILED。旧4096属于开发，不是独立确认。未进入新seed确认、整图计时或训练；16.2s CPU数值检验不是GPU性能。v1最后组参照相减断言失败保留，v2用等价残差二次型，不clamp或删例。</Text>
      <H2>合法F32三维输入，正确舍入后根信息已经丢失</H2>
      <Text>相机/rotation I、query(0,0,1)、epsilon=8589935/2³²，两Gaussian z=.5/1.5，amplitude=.75/.5。第一中心y_A=7735379/2³²，y_B=15470759/2³³，相差一个F32间隔2⁻³³；第二y=0。均是合法正定世界输入。</Text>
      <Table headers={["对象", "世界A", "世界B"]} rows={[
        ["未round有效alpha₁", ".5000000205022779", ".4999999942937956"],
        ["正确F32 alpha₁", ".5", ".5"],
        ["全部6个F32 p/rho/alpha字节", "与B相同", "与A相同"],
        ["量化模型精确median", "1", "1"],
        ["未round物理根区间证书", "r_A<.512", "r_B>1.488"]
      ]} />
      <Text>60/120位有向区间验证两alpha位于.5的正确舍入单元，以及T_A(.512)小于.5、T_B(1.488)大于.5。物理根相距超过.976，因此仅依据相同F32射线系数产生同一z的算法，必有一侧误差超过.488。量化对称模型的根1本身是正确的；不能用更精确求解这个不同模型恢复已丢的信息。</Text>
      <Text>该存在反例在固定fixtures之后选择；不估计发生率，不排除原世界参数/完整metadata含额外信息，也未核验完整renderer选择、初始bracket、CUDA指令、backward或GT。区间证书信任mpmath.iv，并非proof-assistant核验。一般quantile敏感性、新颖性仍待审查。</Text>
      <H2>下一项：条件精度与可计算误差区间</H2>
      <Text>不继续缓存组合筛查。对层间质量裕量不足的像素，需要保留几何输入及有效alpha的必要精度，核验沿root链的误差区间/触发规则。先在全旧输入和锁定新构造检验正确性，再做完整CUDA成本/后向及真实质量验证。原首版packet24视角和2.60倍耗时失败保持。</Text>
      <Text size="small" tone="secondary">来源：coherent_factor_v2、coherent_factor_audit_v1.json、COHERENT_FACTOR_RESULT.md；2026-10-14。42源/archive、12产物、21030误差向量、54根记录核验；9quaternion inverse符号分量通过。原LaTeX原位v24编译成功，静态快照无活任务声明。</Text>
    </Stack>}
'''.replace('ROWS',json.dumps(rows,ensure_ascii=False))
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
p=BASE/'canvas_coherent_stage_v2.tsx';assert not p.exists();p.write_text(text,encoding='utf-8')
r={'previous_sha256':PREVIOUS,'staged_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'destination':str(TARGET)}
(BASE/'results/canvas_coherent_guard_v2.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
