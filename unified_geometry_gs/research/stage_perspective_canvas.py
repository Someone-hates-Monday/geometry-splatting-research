"""Stage completed matched-factor and conditional packet evidence in the existing canvas."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
PREVIOUS='c32614fac752d7f5ebbc4a3163a9ebb0f073112ce25d9638b3eab14b977db4e0'

def main():
    raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==PREVIOUS
    text=raw.decode('utf-8').replace('\r\n','\n')
    def replace(old,new):
        nonlocal text
        assert text.count(old)==1
        text=text.replace(old,new)
    replace('"方向兼容性"];','"方向兼容性", "匹配因素与透视系数"];')
    replace('useState(11)','useState(12)')
    replace('质量门槛失败；单法向兼容条件与真实参数证书已核验','条件性透视系数身份通过；真实质量收益仍未证实')
    replace('最新进展为世界Gaussian方向衰减的单法向兼容条件、可达谱下界和24个真实参数证书；不代表生产渲染器质量损失，学术新颖性、真实质量收益与加速仍未证实。',
            '最新进展为匹配基元因素分离和复用affine量的条件性透视系数身份；1946适用真实输入独立原参数复核通过，46个clipped输入保留域外。尚无完整渲染器、效率、新颖性或质量收益结论。')
    replace('LaTeX当前原位v17编译成功。','LaTeX当前原位v18编译成功。')
    factors=json.loads((BASE/'experiments/matched_ray_factors_v1/summary.json').read_text(encoding='utf-8'))
    report=json.loads((BASE/'results/perspective_reparam_report_v1.json').read_text(encoding='utf-8'));assert report['numerical_gate_pass']
    labels={'factor_peak':'peak排列平均','factor_width':'width排列平均','factor_amplitude':'amplitude排列平均',
      'uncap_world_amplitude':'取消world .99 cap','remove_affine_prefix_stop':'取消affine prefix stop',
      'remove_affine_power_alpha_eligibility':'取消power/alpha eligibility','remove_pixel_tile_membership':'取消像素tile',
      'remove_projection_valid_image_extent':'取消projection/image extent','remove_center_near_plane':'取消center near-plane',
      'finite_camera_origin':'改为有限相机起点'}
    rows=[[labels[k],str(v['defined']),f"{100*v['absolute_increment_over_legacy_depth']['median']:.8g}",
           f"{100*v['absolute_increment_over_legacy_depth']['max']:.8g}"] for k,v in factors['component_summaries'].items()]
    block='''    {tab === 12 && <Stack gap={18}>
      <H2>同24条射线：匹配基元和逐项规则</H2>
      <Text>24投影和24射线所有原CPU数组bitwise一致。固定原selected集合的全部8种peak/width/amplitude组合，192根全有限且独立重解通过。再依固定顺序解除筛选并改变起点，24条完整路径均有限，总和误差最大4.34e−19。</Text>
      <Table headers={["因素/固定路径增量", "定义样本数", "绝对增量/旧深度 中位（%）", "最大（%）"]} rows={ROWS} />
      <Text size="small" tone="secondary">来源：matched_ray_factors_v1/summary.json、matched_ray_factors_report_v1.json；2026-10-13。前三项为标准六排列平均分配，其余为预定路径差，不是唯一物理/因果归因。69源/archive、50产物、3PLY、24membership复核；192固定集合根独立重解最大差2.69e−13，大集合阶段根未独立重解。</Text>
      <Text>中位相消比例29.1519%，最大87.0742%。最大尾部scan37/cam0041：peak +.204732767、取消eligibility −.041186829，完整net +.164084657个世界单位。不能把总差全归因某个因素；此处不是GT质量对照。</Text>
      <H2>条件性透视重参数：复用已有affine量</H2>
      <Text>未clip中心、screen kernel=0且P与Sigma互逆时，令g=q₀·δ/||q₀||²，f=||q₀+δ||²/||q₀||²，ell=g−Delta_A/t₀，chi=E_A(s₀/t₀)²，D=(1+ell)²+chi。精确世界ray peak=t₀√f(1+ell)/D，width=s₀√f/√D，alpha=amplitude·exp(−E_A/(2D))。</Text>
      <Text>Schur补给出conic能量身份；Delta_A是直接plane-offset线性项，E_A必须来自cap前conic。该表达是精确代数身份，仍可能继承conic舍入/坏条件，不保证FP32稳定或加速。</Text>
      <Table headers={["核验", "实际结果", "解释边界"]} rows={[
        ["符号与固定SPD", "9身份；64/64构造通过", "原1e−8/floor1e−12门槛"],
        ["真实selected输入", "1992保留；1946适用全部通过", "46 clipped center域外，不能算通过"],
        ["独立原PLY 80位oracle", "全部1946；max p/width/alpha=4.99e−15/1.66e−15/2.63e−12", "数值交叉核验，非严格区间/形式化认证"],
        ["实际来源与产物", "124源/archive、50产物、3PLY", "全体输入、域分类和模型SHA核验"]
      ]} />
      <Text>HTGS和3DGUT已有一般透视正确Gaussian评估，不能将精确ray概念作为原创。此packet形式是否有独立价值仍需新颖性与实现比较。下一步先一般化clip域，再做FP32/极端数值、CUDA forward/VJP、实际开销；通过后再做固定模型质量对照和决定训练。</Text>
      <Text size="small" tone="secondary">来源：PERSPECTIVE_REPARAM_THEORY.md、PERSPECTIVE_REPARAM_RESULT.md、perspective_reparam_theory_v1.json、perspective_reparam_report_v1.json；2026-10-13。当前LaTeX原位v18编译成功。无GPU、梯度、速度、GT、新视角或重建收益声明。</Text>
    </Stack>}
'''.replace('ROWS',json.dumps(rows,ensure_ascii=False))
    marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
    stage=BASE/'canvas_perspective_stage.tsx';stage.write_text(text,encoding='utf-8')
    guard={'previous_sha256':PREVIOUS,'staged_sha256':hashlib.sha256(stage.read_bytes()).hexdigest(),
      'destination':str(TARGET),'scope':'Existing canvas tab12 only, completed evidence and all exclusions.'}
    (BASE/'results/canvas_perspective_guard_v1.json').write_text(json.dumps(guard,indent=2)+'\n',encoding='utf-8');print(json.dumps(guard))

if __name__=='__main__':main()
