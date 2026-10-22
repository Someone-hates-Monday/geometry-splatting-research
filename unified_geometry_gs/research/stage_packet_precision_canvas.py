"""Guarded append of finished precision rejection and production CUDA evidence."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
TARGET=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
PREVIOUS='64e21440c013bdd1c80b7c471fbe3b4614342188bfdea7a2381ec14c5327ca97'
raw=TARGET.read_bytes();assert hashlib.sha256(raw).hexdigest()==PREVIOUS
text=raw.decode('utf-8').replace('\r\n','\n')
def replace(old,new):
    global text
    assert text.count(old)==1,old;text=text.replace(old,new)
replace('"裁剪扩展与精度负例"];','"裁剪扩展与精度负例", "精度候选与CUDA核验"];')
replace('useState(13)','useState(14)')
replace('含裁剪透视恢复通过；FP32构造门槛仍失败','CUDA局部前向通过；独立构造拒绝精度候选')
replace('最新表达覆盖全部1992个原selected输入，包含旧46个clip；独立80位forward和71712个raw FP64梯度分量通过。但128个FP32构造有1例峰位失败，整体开发门槛保持失败；真实CUDA预处理、速度、新颖性与质量收益未证实。',
        '复制生产CUDA预处理的诊断扩展在既有1992输入局部通过；两种输入保护方案在新4096构造失败13/11，均拒绝。已有留出TEST仅+.0901dB且SSIM下降；质量、效率与新颖性目标尚未实现。')
c=json.loads((BASE/'experiments/packet_input_confirmation_v1/summary.json').read_text(encoding='utf-8'))
p=json.loads((BASE/'experiments/production_packet_v3/summary.json').read_text(encoding='utf-8'))
rows=[[r['name'],str(r['bad_inputs'])+'/4096',f"{r['max_errors'][0]:.7g}",str(r['all_inputs_pass'])] for r in c['variants']]
cuda=[[name,str(r['failed_inputs'])+'/1992',*[f'{v:.7g}' for v in r['max_errors_finite_only']]] for name,r in p['arms'].items()]
block='''    {tab === 14 && <Stack gap={18}>
      <H2>独立构造拒绝开发集通过的两种输入保护</H2>
      <Text>64种保护组合覆盖1992真实+128旧构造后，最小通过mask8只保护qmean，mask21保护plane/conic/camera_geometry。锁定后，以seed13017生成4096同分布新构造，两者均失败；不在看到结果后重新搜索组合。</Text>
      <Table headers={["固定方法", "失败数/输入数", "peak最大相对误差", "全部通过"]} rows={ROWS} />
      <Text>原门槛peak≤1e−5、width≤1e−4（floor1e−8）、alpha绝对≤1e−4。所有保护组合使用double求值，不是实际FP32预处理。选择阶段旧6360系数逐位一致；新失败并集24例独立80位因子参照保持全部分类。不是独立真实场景质量实验。</Text>
      <H2>实际生产CUDA操作：原1992真实输入局部通过</H2>
      <Table headers={["诊断扩展的decode臂", "失败数/输入数", "peak相对max", "width相对max", "alpha绝对max"]} rows={CUDA_ROWS} />
      <Text>复制完整生产preprocessCUDA及其依赖函数直接启动；46clip保留、0cull。676源/archive、50产物、生成文件与二进制、3原PLY核验。5976 CUDA输出重放逐位一致，实际CUDA den/det/energy无效0。非已安装renderer二进制、完整反向或效率实验；两额外geometry臂不是仅有packet的部署方案。v1/v2环境编译失败日志保留。</Text>
      <H2>理论范围与下一步</H2>
      <Text>有限扰动界保留cross接近零的相对敏感性；5符号身份与2120数值界检查通过。同一binary32字节的两个合法SPD输入，其peak接受区间分离下界约4.92775e−7，仅约束直接round qmean的八输入表示。mpmath.iv是实验性可信依赖；不扩写成全部生产布局不可能。</Text>
      <Text>贡献审计确认IFT/exact ray/Schur工具已有明确重合。薄层两种裕量与梯度反例仍为理论候选，新颖性未证实；Triangle原论文技术核验尚缺。下一项完整CUDA/VJP与实际成本，然后冻结模型affine/exact packet/optimized direct factor三臂GT对照；未过门槛不扩大训练。</Text>
      <Text size="small" tone="secondary">来源：packet_input_protection_v1、packet_input_confirmation_v1、packet_input_protection_report_v1.json、packet_conditioning_v1.json、production_packet_v3、production_packet_report_v1.json；2026-10-13。原LaTeX原位v20编译成功；用户要求先简报，当前没有新训练。</Text>
    </Stack>}
'''.replace('CUDA_ROWS',json.dumps(cuda,ensure_ascii=False)).replace('ROWS',json.dumps(rows,ensure_ascii=False))
# Only replace the table token; the English CUDA words in prose remain intact.
marker='  </Stack>;\n}';assert text.count(marker)==1
text=text.replace(marker,block+marker)
stage=BASE/'canvas_packet_precision_stage.tsx';stage.write_text(text,encoding='utf-8')
guard={'previous_sha256':PREVIOUS,'staged_sha256':hashlib.sha256(stage.read_bytes()).hexdigest(),'destination':str(TARGET)}
(BASE/'results/canvas_packet_precision_guard_v1.json').write_text(json.dumps(guard,indent=2)+'\n',encoding='utf-8')
print(json.dumps(guard))
