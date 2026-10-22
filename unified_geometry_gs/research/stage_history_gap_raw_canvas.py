from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');install=parser.parse_args().install;guard=R/'results/canvas_history_gap_raw_guard_v1.json';stage=R/'canvas_history_gap_raw_stage_v1.tsx'
    if install:
        g=json.loads(guard.read_text(encoding='utf-8'));s=json.loads((R/'results/canvas_history_gap_raw_syntax_v1.json').read_text(encoding='utf-8'));assert not s['errors'];t=Path(g['destination']);assert sha(t)==g['previous_sha256'] and sha(stage)==g['staged_sha256'];t.write_bytes(stage.read_bytes());assert sha(t)==g['staged_sha256'];print(json.dumps(dict(status='updated_existing',sha256=sha(t),new_tabs_opened=0)));return
    t=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx');old='1ef2c5048ce9a33c4f710d093694ebf8e84f8ce3343b8b0981043c0971f5485b';assert sha(t)==old;text=t.read_text(encoding='utf-8').replace('\r\n','\n');assert not stage.exists();audit=json.loads((R/'results/history_gap_raw_audit_v1.json').read_text(encoding='utf-8'));assert audit['audit_gate_pass'];ext=json.loads((R/'results/history_gap_extensions_audit_v2.json').read_text(encoding='utf-8'));assert ext['audit_gate_pass']
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,a;text=text.replace(a,b)
    replace('"平方历史差证明"];','"平方历史差证明", "原参数链与无根延拓"];');replace('useState(32)','useState(33)');replace('平方历史差C1证明与CPU梯度核验完成；高loss弱梯度反例成立，训练价值未证','原参数链与共享下降完成；无根时C1延拓证明/核验通过，GPU和质量仍待验证')
    block='''    {tab === 33 && <Stack gap={18}>
      <H2>从系数到可训练参数：独立MP链与真实共享ID更新</H2>
      <Text>仍是固定891射线/51,524节点、原9异常。位置/过滤尺度/归一化四元数/opacity/cap全链：factor891/891过，整射线相对L2最大4.54e−11、各块≤2.54e−10。dense890/891过，原组合gate保留失败。原42方向差分35过；7失败的独立240位完整前向微步核验全过，不覆盖原失败。</Text>
      <Text>24,007唯一Gaussian中6,278重复，最大137次。按(scan,ID)真实共享梯度，raw/缩放一致，891个zero-control损失逐位一致。下表仅固定源平均loss、等坐标步长.001；不代表GT、独立训练或全可见性更新。</Text>
      <Table headers={["共享更新", "平均loss", "原0.2891820143"]} rows={[["全部11参数", "0.2889393258", "约下降0.0839224%"], ["仅几何", "0.2889393275", "约下降0.0839218%"], ["仅opacity", "0.2891811147", "约下降0.0003111%"]]} />
      <H2>原失败与精度局限全部保留</H2>
      <Text>共享h=.01的full/geometry各1射线无有限根，原平均值未定义，没有删除后重新平均。共享原方向h=1e−5差分约5.35%失败，主因一个旧异常；后续全891微步1e−7/1e−9/1e−11分别误差3.877%/8.58e−6/2.23e−4，仅中间过。主点独立MP支持截断与舍入限制，不等于全部891重做MP。</Text>
      <Text>真实561个loss≥.25者在预设缩放下没有梯度范数≤1e−8者；3条零loss零梯度。理论分离薄表面梯度指数消失反例仍成立，不能从开发集计数推断总体率。缩放梯度能量约99.14%在quaternion，不是训练因果效果。</Text>
      <H2>无有限根时仍有定义的C1延拓</H2>
      <Text>Ω=sum[−log(1−α)]。Ω＞log2保留原平方历史差；Ω≤log2定义(log4−Ω)²。边界附近H(max peak)＜log2，有根一侧已精确等于相同公式，因此整个正系数开域C1。不会把缺失射线奖励成0。cap/visibility/IEEE离散输入不在C1声明内。</Text>
      <Text>16随机模型MP差分最大相对3.34e−56、16双侧边界及精确单层点通过。两个原缺根更新的MP质量不足裕量约.0025415，延拓loss .48398274/.48398332；raw方向导数与MP前向误差≤2.4e−14。原失败结果未回写；未声称全共享branch的MP重验。</Text>
      <H2>下一阶段</H2>
      <Text>实现包含无根延拓的GPU批处理loss/伴随，独立MP/CPU核验既有全批和新确认输入，记录全部成本与失败；正确性通过后才配对训练和GT/留出图像评价。当前没有训练、质量、加速或新颖性成功结论，整体目标未达成。</Text>
      <Text size="small" tone="secondary">HISTORY_GAP_RAW_RESULT.md / raw_v1、raw_micro_v1、shared_v1、shared_micro_v1、extension_v1及两个独立审核。2026-10-15 Asia/Shanghai。原LaTeX原位编译；看板仅TypeScript语法检查，非SDK/runtime验证。</Text>
    </Stack>}
'''
    marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker);stage.write_text(text,encoding='utf-8');g=dict(previous_sha256=old,staged_sha256=sha(stage),destination=str(t));guard.write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
if __name__=='__main__':main()
