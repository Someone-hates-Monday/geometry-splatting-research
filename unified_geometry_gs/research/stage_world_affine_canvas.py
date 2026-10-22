"""Add the completed same-ray mechanism check to the existing evidence canvas."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
PREVIOUS='ef4b966064eae565d4afb7d6b285e83bd31655130e39d4369c51e0057af4406a'

def main():
    raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==PREVIOUS
    text=raw.decode('utf-8').replace('\r\n','\n')
    report=json.loads((BASE/'results/world_affine_rays_report_v1.json').read_text(encoding='utf-8'));assert report['status']=='complete'
    summary=json.loads((BASE/'experiments/world_affine_rays_v1/summary.json').read_text(encoding='utf-8'))
    rows=[[f"scan{g['scan']}",f"{100*g['abs_difference_over_legacy_depth']['median']:.6f}",
           f"{100*g['abs_difference_over_legacy_depth']['max']:.6f}",f"{g['old_point_world_camera_T']['median']:.6f}"] for g in summary['per_scan']]
    marker='      <H2>文献重叠与下一步</H2>'
    assert text.count(marker)==1
    block='''      <H2>同24条射线：方向衰减差不自动对应中位深度位移</H2>
      <Text>完整世界Gaussian限制到射线后，用T_camera(t)=T_minus(t)/T_minus(0)求相机中位根。24/24有限；点态logV与射线图最大差1.98e−13，attenuation最大比例差8.47e−13；根log residual最大3.04e−11。无严格区间根界。</Text>
      <Table headers={["完整世界模型", "绝对深度位移/旧深度 中位（%）", "最大（%）", "旧affine根处 world T_camera中位"]} rows={ROWS} />
      <Text>全部24点的相对位移中位.0126254%。最大例scan37/camera0041/pixel(601,131)：6.089402428→6.253487084，+2.69459%；世界坐标单位，未换算mm。保留全部点与尾部。</Text>
      <Text size="small" tone="secondary">来源：world_affine_rays_v1全部24行、world_affine_rays_report_v1.json；2026-10-13。40源/archive、26产物、3PLY与全部算术重放通过，但未独立重算所有根。比较同时改变affine形状、near-plane/tile/alpha/prefix选择及clamp，无法单独归因为透视近似。没有GT或质量收益结论。</Text>
'''.replace('ROWS',json.dumps(rows,ensure_ascii=False))
    text=text.replace(marker,block+marker)
    old='LaTeX当前原位v16编译成功。'
    assert text.count(old)==1;text=text.replace(old,'LaTeX当前原位v17编译成功。同射线检查亦complete、匹配进程为空。')
    stage=BASE/'canvas_world_affine_stage.tsx';stage.write_text(text,encoding='utf-8')
    guard={'previous_sha256':PREVIOUS,'staged_sha256':hashlib.sha256(stage.read_bytes()).hexdigest(),
           'destination':str(TARGET),'scope':'Existing tab11, completed same-ray evidence and limitations.'}
    (BASE/'results/canvas_world_affine_guard_v1.json').write_text(json.dumps(guard,indent=2)+'\n',encoding='utf-8');print(json.dumps(guard))

if __name__=='__main__':main()
