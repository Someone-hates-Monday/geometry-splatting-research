"""Append the completed coefficient/world analysis to the existing canvas."""
from pathlib import Path
import json,hashlib

BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
PREVIOUS='5ccf444bc3555bbaf543eab51eaf9e0b23afbb2f02f82786f0470bb748b9a354'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==PREVIOUS
text=raw.decode('utf-8').replace('\r\n','\n')


def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)


replace('"全部异常根诊断"];','"全部异常根诊断", "有效alpha与世界参照"];')
replace('useState(16)','useState(17)')
replace('全9异常点已重建：大差异主要存在于系数表示','72混合根定位有效alpha差；世界参照完成')
replace('当前不扩大GT或训练，下一步隔离系数并对照raw世界参照。',
        '全72混合根已定位有效alpha差；MP80/140世界参照与共同opacity参照完成。当前不扩大GT或训练，下一步一致能量构造与薄层裕量理论。')
mix=json.loads((BASE/'results/exact_outlier_mix_report_v1.json').read_text(encoding='utf-8'))
world=json.loads((BASE/'experiments/exact_outlier_world_v1/summary.json').read_text(encoding='utf-8'))
audit=json.loads((BASE/'results/exact_outlier_world_audit_v1.json').read_text(encoding='utf-8'))
gap=json.loads((BASE/'experiments/exact_outlier_gap_v1/summary.json').read_text(encoding='utf-8'))
rows=[]
for r,w,g in zip(audit['rows'],world['rows'],gap['rows']):
    assert r['case']==w['case']==g['case']
    c=w['comparisons']['raw_physical']
    rows.append([str(r['case']),f"scan{r['scan']}/{r['camera']}",
                 f"{float(c['packet_mathematical_root_difference_camera_ray_signed']):.8g}",
                 f"{float(c['factor_mathematical_root_difference_camera_ray_signed']):.8g}",r['raw_physical_closer'],
                 '/'.join(a['sufficient_classification'].replace('root_left_of_gap','左').replace('root_right_of_gap','右').replace('bound_inconclusive','不充分') for a in g['arms'].values())])
block='''    {tab === 17 && <Stack gap={18}>
      <H2>有效alpha表示差决定本批大根差</H2>
      <Text>固定全部9点/772 accepted项，factor=mask0、packet=mask7，bit0/1/2替换alpha/peak/rho。全9×8数学模型有限；120位根和180位端点72/72通过，18端点与旧oracle吻合。混合模型不一定对应共同3D Gaussian，六顺序分配只是代数机制分离。</Text>
      <Table headers={["固定混合检查", "最大变化（camera-ray模型单位）", "含义"]} rows={[
        ["全替换根差绝对值", ".1820989237（case2）", "保存模型的差"],
        ["alpha-only与全替换差", "9.58250e−7（全9最大）", "沿射线有效alpha主导"],
        ["只替换peak+rho", "9.58099e−7", "在这些选定模型中较小"],
        ["alpha×几何交互", "9.54527e−7", "保留非线性交互"],
        ["有效alpha含off-ray能量衰减", "非单独训练opacity参数", "不能叫opacity训练因果"]
      ]} />
      <H2>独立世界参照：结果依赖明确的数值对象</H2>
      <Text>原3 PLY逐ID核验772原始F32字段；saved-activated保留F32激活参数且真正逆非正交R covariance，raw-physical重新MP激活和unit quaternion。80位cov逆与140位inverse factor/residual相符，36根端点复查通过。相机用保存F32仿射view精确逆，t为单位camera-ray参数，非假定单位world弧长。参考alpha不先round F32。</Text>
      <Table headers={["case", "场景/相机", "packet−raw根", "factor−raw根", "较近模型", "中央60%区间界 P/F"]} rows={ROWS} />
      <Text>两世界参照及第三共同co.w参照均factor较近8/9、packet较近case0。这是选定异常的描述，非GT、统计显著性或总体排名；factor在case0仍有约.0012097 ray误差，未获完整采用结论。</Text>
      <H2>平台裕量与信息丢失</H2>
      <Text>无peak区间[a,b]中，H=M_L−左tails+右tails，故M_L−左tails(a)≤H(t)≤M_L+右tails(b)。case1/2/5的两臂落在区间相反侧，其余6个单臂界不充分。独立MP180 direct-H核验18模型/126点；不是有向区间证书。最大两例中点H_t约5.584e−13/6.4e−161，根处局部斜率不能界跨层gap。</Text>
      <Text>实际共同co.w与filtered opacity在2307/17581候选、96/772 accepted末位不同，accepted最大5.96046e−8。控制这个共同amplitude后8/9比较仍同，参考根移位最多1.78528e−6 ray；两exact臂共享co.w，不是两臂独立缓存差。</Text>
      <Text>候选信息损失点：packet混用covariance→conic与precision→plane，且从投影xy反算mean的舍入/1e−7分母与factor view mean路径不同。尚不能唯一归因于其中一项。下一步一致precision factor构造b/c/Schur能量及条件误差界，再核验锁定旧输入与新构造。9F32缓存预算不等于速度或稳定性证据。</Text>
      <Text size="small" tone="secondary">来源：exact_outlier_mix_report/check_v1、exact_outlier_world_v1/world_audit_v1、gap_v1/gap_audit_v1；2026-10-14。原LaTeX原位v23编译成功。完整24视角数值及2.60倍耗时失败保持，当前无GT/训练扩展。</Text>
    </Stack>}
'''.replace('ROWS',json.dumps(rows,ensure_ascii=False))
marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
stage=BASE/'canvas_coefficient_stage.tsx';assert not stage.exists();stage.write_text(text,encoding='utf-8')
record={'previous_sha256':PREVIOUS,'staged_sha256':hashlib.sha256(stage.read_bytes()).hexdigest(),'destination':str(TARGET)}
(BASE/'results/canvas_coefficient_guard_v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
