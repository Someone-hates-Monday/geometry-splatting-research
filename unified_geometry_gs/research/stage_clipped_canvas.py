"""Append completed clip recovery and the retained FP32 failure to the existing canvas."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
PREVIOUS='25419a9b1c46c890d644c42a38bb7652fb589ee8111ffdf3f9ede91abe99630a'

def main():
    raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==PREVIOUS
    text=raw.decode('utf-8').replace('\r\n','\n')
    def replace(old,new):
        nonlocal text
        assert text.count(old)==1;text=text.replace(old,new)
    replace('"匹配因素与透视系数"];','"匹配因素与透视系数", "裁剪扩展与精度负例"];')
    replace('useState(12)','useState(13)')
    replace('条件性透视系数身份通过；真实质量收益仍未证实','含裁剪透视恢复通过；FP32构造门槛仍失败')
    replace('最新进展为匹配基元因素分离和复用affine量的条件性透视系数身份；1946适用真实输入独立原参数复核通过，46个clipped输入保留域外。尚无完整渲染器、效率、新颖性或质量收益结论。',
            '最新表达覆盖全部1992个原selected输入，包含旧46个clip；独立80位forward和71712个raw FP64梯度分量通过。但128个FP32构造有1例峰位失败，整体开发门槛保持失败；真实CUDA预处理、速度、新颖性与质量收益未证实。')
    replace('LaTeX当前原位v18编译成功。','LaTeX当前原位v19编译成功。')
    report=json.loads((BASE/'results/clipped_packet_precision_report_v1.json').read_text(encoding='utf-8'))
    assert not report['development_gate_pass'] and report['synthetic_fp32_bad_cases']==1
    d=report['failed_cases'][0]
    rows=[['原double packet / 80位求值',f"{d['original_double_packet_80digit_scaled_errors'][0]:.8g}",'通过'],
          ['已round FP32 packet / 80位求值',f"{d['rounded_float32_packet_80digit_scaled_errors'][0]:.8g}",'失败'],
          ['已round FP32 packet / FP32求值',f"{d['saved_float32_errors'][0]:.8g}",'失败']]
    block='''    {tab === 13 && <Stack gap={18}>
      <H2>参考中心可以被clip：完整二次型仍可恢复</H2>
      <Text>kernel0且Sigma=P⁻¹时，现有conic、plane两项、width提供参考基底中的Schur参数(b,c,K)。P在该基底的Gram为b[[K+ccᵀ,c],[cᵀ,1]]，六个自由度可唯一恢复；这是标准块分解的Gaussian应用，不是新的矩阵分解。</Text>
      <Text>令a=qray_xy−qbar_xy、m=qmean_xy−qbar_xy，A=1+c·a、B=1+c·m，den=A²+aᵀKa、cross=AB+aᵀKm。peak=z||qray||cross/den，width=||qray||/√(b·den)。能量用Gram非负分解det(K)(a×m)²+(Am−Ba)ᵀK(Am−Ba)计算，仍可能继承浮点det/cross敏感性。</Text>
      <Table headers={["核验范围", "已完成", "限制"]} rows={[
        ["符号/固定SPD", "6身份；128/128，其中112 clip", "一般精确算术身份，不保证浮点稳定"],
        ["同24射线真实输入", "1992/1992，包括原46 clip", "旧选择集合不变，无fallback"],
        ["独立原PLY80位", "全部1992；max p/s/alpha=4.72e−15/1.52e−15/2.61e−12", "180源/archive、50产物、3PLY核验"],
        ["raw FP64 AD图", "71712/71712分量通过；max4.19e−7", "1e−4/floor1e−8；共用raw变换与Torch AD"],
        ["真实FP32 packet", "1992/1992通过；p/s相对3.13e−7/2.71e−7，alpha绝对5.59e−6", "仅round double数据，不是实际FP32/CUDA预处理"],
        ["合成FP32开发守护", "127/128通过；case2峰位失败", "development_gate_pass=false，不能删例或放宽门槛"]
      ]} />
      <H2>保留的FP32失败：输入舍入与求值误差都存在</H2>
      <Table headers={["同一case2的计算方式", "峰位相对误差", "原1e−5门槛"]} rows={ROWS} />
      <Text>真peak≈−.02839817738，FP32≈−.02839967981。独立80位decode已经round过的packet仍超阈值，单提高求值精度不足。该例真alpha≈2.3005e−5712，double下为0，物理响应极小；仍保留原失败判定，不据此证明任意decoder不可能。</Text>
      <Text size="small" tone="secondary">来源：clipped_packet_theory_v1.json、clipped_packet_report_v1.json、clipped_packet_precision_v1/summary.json、clipped_packet_precision_report_v1.json；2026-10-13。233源/archive、51产物、所有行/梯度算术复核；失败例误差分离是事后分析。逐输入参数复制不测试共享基元atomic累加，非完整CUDA backward。</Text>
      <H2>下一项：输入精度保护与实际生产数据</H2>
      <Text>先处理输入舍入敏感性，任何新方法/门槛另立方案并保留本失败；再核验实际FP32预处理、CUDA forward/backward与资源/开销。当前不训练，无GT质量、速度或新颖性结论。LaTeX原位v19已编译通过，原文档保持打开。</Text>
    </Stack>}
'''.replace('ROWS',json.dumps(rows,ensure_ascii=False))
    marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
    stage=BASE/'canvas_clipped_stage.tsx';stage.write_text(text,encoding='utf-8')
    guard={'previous_sha256':PREVIOUS,'staged_sha256':hashlib.sha256(stage.read_bytes()).hexdigest(),
      'destination':str(TARGET),'scope':'Existing tab13; all clip inputs and full precision failure retained.'}
    (BASE/'results/canvas_clipped_guard_v1.json').write_text(json.dumps(guard,indent=2)+'\n',encoding='utf-8');print(json.dumps(guard))

if __name__=='__main__':main()
