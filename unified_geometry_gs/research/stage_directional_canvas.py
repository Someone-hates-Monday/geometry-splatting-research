"""Stage a data-backed addition to the existing research canvas with a byte guard."""
from pathlib import Path
import hashlib,json

BASE=Path(__file__).resolve().parent
CANVAS=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
EXPECTED='5383793df032ee813aab1d6290d2c824da9c315eb0865c8f5b97e33ebddcba5c'

def main():
    payload=CANVAS.read_bytes();assert hashlib.sha256(payload).hexdigest()==EXPECTED
    source=payload.decode('utf-8').replace('\r\n','\n')
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1
        source=source.replace(old,new)
    replace('"薄层与区间深度"];','"薄层与区间深度", "方向兼容性"];')
    replace('useState(10)','useState(11)')
    replace('质量门槛失败；薄层梯度反例与区间深度定理已核验','质量门槛失败；单法向兼容条件与真实参数证书已核验')
    replace('最新进展为固定射线模型的薄层极限、正则根梯度反例和分位区间的条件稳定性证明；新颖性、真实质量收益与加速仍未证实。',
            '最新进展为世界Gaussian方向衰减的单法向兼容条件、可达谱下界和24个真实参数证书；不代表生产渲染器质量损失，学术新颖性、真实质量收益与加速仍未证实。')
    run=BASE/'experiments/directional_field_world_v1'
    report=json.loads((BASE/'results/directional_field_world_report_v1.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and report['verified']['queries_reconstructed_exactly']==24
    rows=[]
    for i in range(24):
        r=json.loads((run/f'point_{i:02d}.json').read_text(encoding='utf-8'))
        rows.append([f"scan{r['scan']} / {r['camera']}",f"{r['interval_pair_certificate']['conservative_double_lower_bound']:.6g}",
                     f"{100*r['nominal_lower_bound_over_sum_norms']:.4f}",f"{100*r['relative_directional_gap'][5]:.4f}"])
    rowsjs=json.dumps(rows,ensure_ascii=False)
    block='''    {tab === 11 && <Stack gap={18}>
      <H2>固定点：何时能用单个法向表示所有方向衰减</H2>
      <Text>K(w)=Σ|w·bᵢ|，bᵢ=∇log√(1−Gᵢ)。存在任意b使K(w)=|w·b|对所有单位方向成立，当且仅当所有非零bᵢ共线；若指定b=Σbᵢ，即product-v梯度，还必须同向。互易性本身不够，点态兼容也不保证全局可积。</Text>
      <Text>E=inf_b sup_unit_w |K(w)−|w·b|| ≥ √λ₂(Σbᵢbᵢᵀ)。两正交梯度的合法Gaussian构造达到下界。任意两项的√(cross²/trace)仍是全模型绝对下界，添加其他Gaussian不会削弱它。</Text>
      <Table headers={["证据", "已核验", "范围"]} rows={[
        ["数学检查", "精确SymPy + 64预定数值见证", "完整解析证明给出全方向结论"],
        ["真实参数查询", "24/24正区间下界；3完整模型", "8保存视角/scan；不是GT表面点"],
        ["原始来源", "13源/archive、26产物、3PLY、48raw基元", "24点按原相机/射线精确回投影"],
        ["独立复核", "24有理数不等式；144空间导数分量", "120位标量求导落入80/100位区间"],
        ["名义归一化下界", "min .9000%；median 6.7901%；max 21.3081%", "非区间相对界；非像素/几何错误率"]
      ]} />
      <H2>全部24点：保存绝对证书与名义方向统计</H2>
      <Table headers={["查询点 scan / camera", "绝对下界（世界单位⁻¹）", "名义谱下界 / Σ||bᵢ||（%）", "原相机product衰减差（%）"]} rows={ROWS} />
      <Text size="small" tone="secondary">来源：directional_field_world_v1全部point_00..23.json、directional_field_world_report_v1.json；2026-10-13。绝对下界由双基元区间支持；后两列为完整double测量。最小证书3.4193e−10，非零不等于可见效应。所有名义指数underflow计数保留。</Text>
      <H2>文献重叠与下一步</H2>
      <Text>GGGS近侧条件、Gaussian Wrapping方向问题与Objects as Volumes一般法向分布均为已有工作。K是zonotope支持函数，单法向是线段；Siegel2023与Torres2024已有一般逼近理论。因此不能把基础表示或泛泛多视图问题作为原创。本补充的新颖性尚未完成全面审查。</Text>
      <Text>当前世界场不等于生产per-view affine图；24点不是GT表面或世界模型精确median。下一步需区分affine误差、近侧条件和真实跨视图误差，不据此直接加入全模型单法向惩罚或训练。mpmath.iv仍实验性；高精度与有理数重放不等于证明助手认证。</Text>
      <Text size="small" tone="secondary">LaTeX当前原位v16编译成功。holdout健康v4与方向诊断健康v1均complete、匹配进程为空。最新线程记录确认两轮stream disconnected before completion；此前父队列暂停是另一种已定位情况。没有训练完成触发断流或CUDA死锁的证据。</Text>
    </Stack>}
'''.replace('ROWS',rowsjs)
    marker='  </Stack>;\n}'
    assert source.count(marker)==1
    source=source.replace(marker,block+marker)
    stage=BASE/'canvas_directional_stage.tsx';stage.write_text(source,encoding='utf-8')
    guard={'previous_sha256':EXPECTED,'staged_sha256':hashlib.sha256(stage.read_bytes()).hexdigest(),
           'destination':str(CANVAS),'rows':len(rows),'scope':'Existing canvas only; tab11 default, no fetch/new imports.'}
    (BASE/'results/canvas_directional_guard_v1.json').write_text(json.dumps(guard,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(guard,indent=2))

if __name__=='__main__':main()
