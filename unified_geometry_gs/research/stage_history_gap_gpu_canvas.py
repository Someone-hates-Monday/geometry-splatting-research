from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');install=parser.parse_args().install;guard=R/'results/canvas_history_gap_gpu_guard_v1.json';stage=R/'canvas_history_gap_gpu_stage_v1.tsx'
    if install:
        g=json.loads(guard.read_text(encoding='utf-8'));s=json.loads((R/'results/canvas_history_gap_gpu_syntax_v1.json').read_text(encoding='utf-8'));assert not s['errors'];t=Path(g['destination']);assert sha(t)==g['previous_sha256'] and sha(stage)==g['staged_sha256'];t.write_bytes(stage.read_bytes());assert sha(t)==g['staged_sha256'];print(json.dumps(dict(status='updated_existing',sha256=sha(t),new_tabs_opened=0)));return
    t=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx');old='0f83bc56f46e4ad14ada355c8e1eac29251e911cc9e875900a7e93c70b0617b0';assert sha(t)==old;text=t.read_text(encoding='utf-8').replace('\r\n','\n');assert not stage.exists();audit=json.loads((R/'results/history_gap_gpu_audit_v1.json').read_text(encoding='utf-8'));assert audit['audit_gate_pass']
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,a;text=text.replace(a,b)
    replace('"原参数链与无根延拓"];','"原参数链与无根延拓", "GPU损失与原参数伴随"];');replace('useState(33)','useState(34)');replace('原参数链与共享下降完成；无根时C1延拓证明/核验通过，GPU和质量仍待验证','GPU损失/完整原参数梯度通过891真实射线与64合成确认；串行核成本仍高，训练待验证')
    block='''    {tab === 34 && <Stack gap={18}>
      <H2>GPU正确性完成：独立求损失与完整原参数伴随</H2>
      <Text>原source/raw/camera输入，GPU自行前/尾/中间判断、内部二分、稳定density伴随，Torch Float64图链回11原参数；没有CPU/MP根或adjoint输入。真实891射线/51,524节点/9异常全保留：877中间二分、11尾部公式、3零loss，全过原loss/整体及各块梯度门槛。普通Float64符号不是区间认证。</Text>
      <Table headers={["核验", "结果", "相对梯度误差"]} rows={[["真实完整891", "891 / 891", "整体max9.15e−11；块max1.73e−10"], ["原32合成回归", "32 / 32", "与独立MP链比较"], ["另一seed13019新合成32", "32 / 32", "整体max2.63e−14"], ["原真实无根更新", "2 / 2", "raw方向误差≤1.42e−14"]]} />
      <H2>保留初版28/32失败后修正</H2>
      <Text>初版真实891过，但新合成32只有28过，4个理论零几何块的舍入残差未过块门槛。V2用通用叉积恒等式计算横向能量，并在screen kernel0采用精确零补偿/导数。数学目标/门槛不改，原32变为开发回归，再使用另一固定seed确认。恒等式不是新颖性主张。</Text>
      <Text>另10个系数测试涵盖单/双奇异点、质量边界、无根、前根、严格cap和rho5/10/13.333/20薄间隙，全部过MP。rho0无效输入明确NaN/负状态拒绝。已知大loss弱梯度反例保持，不称根本解决优化问题。</Text>
      <H2>当前成本与下一阶段</H2>
      <Text>RTX5090Laptop上V2完整几何+loss+raw反传warm三次79.164/79.067/79.066ms，首次cold1045.543ms；compile37.305s、准备/transfer .840s单列。process peak allocated153,932,288B，不是训练增量内存。初版阶段拆分loss/seeds72.70ms，是串行per-ray核主要成本；这不是整体训练速度。</Text>
      <Text>独立全数组/MP引用映射、source ZIP与实际binary审核通过；初版数值失败和JSON标量写入/打印失败都保留，恢复计算产物不假装重跑通过。原LaTeX原位编译，当前无训练启动。</Text>
      <Text>下一步warp并行化损失核，维持所有独立梯度/无根/薄层门槛并计完整成本；之后固定采样、真实共享ID、零权重对照的训练adapter及配对GT/留出图像评价。没有训练、重建、整体效率或新颖性成功结论。</Text>
      <Text size="small" tone="secondary">HISTORY_GAP_GPU_RESULT.md / history_gap_gpu_v1-v2、confirm_v1-v4、history_gap_gpu_audit_v1。2026-10-15 Asia/Shanghai。合成64不是64新真实场景。看板仅TypeScript语法检查，非SDK/runtime验证。</Text>
    </Stack>}
'''
    marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker);stage.write_text(text,encoding='utf-8');g=dict(previous_sha256=old,staged_sha256=sha(stage),destination=str(t));guard.write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
if __name__=='__main__':main()
