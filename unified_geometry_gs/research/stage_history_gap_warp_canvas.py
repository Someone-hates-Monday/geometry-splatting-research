from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');install=parser.parse_args().install;guard=R/'results/canvas_history_gap_warp_guard_v1.json';stage=R/'canvas_history_gap_warp_stage_v1.tsx'
    if install:
        g=json.loads(guard.read_text(encoding='utf-8'));s=json.loads((R/'results/canvas_history_gap_warp_syntax_v1.json').read_text(encoding='utf-8'));assert not s['errors'];t=Path(g['destination']);assert sha(t)==g['previous_sha256'] and sha(stage)==g['staged_sha256'];t.write_bytes(stage.read_bytes());assert sha(t)==g['staged_sha256'];print(json.dumps(dict(status='updated_existing',sha256=sha(t),new_tabs_opened=0)));return
    t=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx');old='e553f53bdca33a7fce5c8e75c8bc601fd60383616d9dcf5dc36773c1991e3814';assert sha(t)==old;text=t.read_text(encoding='utf-8').replace('\r\n','\n');assert not stage.exists();audit=json.loads((R/'results/history_gap_warp_audit_v1.json').read_text(encoding='utf-8'));assert audit['audit_gate_pass'] and audit['adoption_gate_pass']
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,a;text=text.replace(a,b)
    replace('"GPU损失与原参数伴随"];','"GPU损失与原参数伴随", "Warp损失核对照"];');replace('useState(34)','useState(35)');replace('GPU损失/完整原参数梯度通过891真实射线与64合成确认；串行核成本仍高，训练待验证','Warp完整loss批次11.99ms，原梯度门槛全过；下一步真实GPU训练adapter，质量仍未证')
    block='''    {tab === 35 && <Stack gap={18}>
      <H2>同一loss与原参数链：warp完整批次约6.60倍加速</H2>
      <Text>32-lane warp协作每条射线，strided全部贡献项、补偿local sums、double-double prefix乘积、固定warp归约树；loss/独立根/cap/伴随不变，原串行source/binary未改，没有隐藏串行fallback。</Text>
      <Table headers={["同进程GPU事件中位数", "串行（ms）", "Warp（ms）"]} rows={[["完整geometry + loss + raw backward", "79.1189", "11.9931"], ["仅loss/seeds核", "72.7598", "5.9131"]]} />
      <Text>10轮每轮交替先后顺序，前2轮每臂warmup原样保留，后8次全部计中位数。full warp/serial=.151583（约降84.84%），达到预设≤.95门槛；allocation计入，setup/transfer/compile单列。只是本loss GPU批次对串行参照的速度，不是原renderer或完整训练加速，时序不是置信区间。</Text>
      <H2>保持全部正确性门槛</H2>
      <Text>两臂各891/891真实开发射线、原9异常/3零loss全保留，整射线max9.15e−11、块max1.73e−10；64合成和2原真实缺根更新全部过。10特殊系数＋新增513/1025长射线全部过；exact有理数确认最后基元决定有限根，节点数完整。invalidrho0明确拒绝，不改零块门槛。</Text>
      <Text>独立审核111来源、17产物、sourceZIP、实际binary、全梯度/loss/原MP引用、12特殊数组、长射线exact证据和所有交替时序复算通过。Float64分支不是区间认证；理论薄间隙弱梯度反例没有改变。</Text>
      <H2>保留失败和下一步</H2>
      <Text>首次runner编译/计时后因汇总dict重复键退出，原source/manifest/traceback保留；第二批复用同一核并重测固定协议，未改数值算法。第二批compile_wall字段实为已有binary加载，不把它称为编译加速。</Text>
      <Text>采用warp作为下一研究adapter的loss核。接实际GaussianModel共享IDs、直接GPU capture/query/AMP，先disabled/零权重对照、正确scatter/有限梯度/optimizer绑定、短步和完整迭代成本，再配对训练及GT/留出图像/容量评价。当前未启动训练、未改原模型或生产renderer，质量/整体训练效率/新颖性仍未证。</Text>
      <Text size="small" tone="secondary">HISTORY_GAP_WARP_RESULT.md / history_gap_warp_v1失败、v2完成、history_gap_warp_audit_v1。2026-10-15 Asia/Shanghai。原LaTeX原位编译；看板仅TypeScript语法检查，非SDK/runtime验证。</Text>
    </Stack>}
'''
    marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker);stage.write_text(text,encoding='utf-8');g=dict(previous_sha256=old,staged_sha256=sha(stage),destination=str(t));guard.write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
if __name__=='__main__':main()
