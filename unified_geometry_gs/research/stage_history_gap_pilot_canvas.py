from pathlib import Path
import argparse,hashlib,json,re
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--install',action='store_true');install=p.parse_args().install
    stage=R/'canvas_history_gap_pilot_stage_v1.tsx';guard=R/'results/canvas_history_gap_pilot_guard_v1.json'
    if install:
        g=json.loads(guard.read_text(encoding='utf-8'));s=json.loads((R/'results/canvas_history_gap_pilot_syntax_v1.json').read_text(encoding='utf-8'))
        assert not s['errors'];t=Path(g['destination']);assert sha(t)==g['previous_sha256'] and sha(stage)==g['staged_sha256']
        t.write_bytes(stage.read_bytes());assert sha(t)==g['staged_sha256'];print(json.dumps(dict(status='updated_existing',sha256=sha(t),new_tabs_opened=0)));return
    t=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
    old='62940c3d00b5bee610292cbfb21ee834e1d61c22fe49671040d37a2b7fb12d65';assert sha(t)==old and not stage.exists()
    assert json.loads((R/'results/history_gap_pilot_audit_v1.json').read_text(encoding='utf-8'))['audit_gate_pass']
    assert json.loads((R/'results/history_gap_posthoc_audit_v1.json').read_text(encoding='utf-8'))['audit_gate_pass']
    text=t.read_text(encoding='utf-8').replace('\r\n','\n')
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,a;text=text.replace(a,b)
    replace('"原完整训练接入"];','"原完整训练接入", "完整30k负结果"];');replace('useState(37)','useState(38)')
    replace('原完整训练24步与零控制通过；30k配对试验运行中，质量结论待定','完整30k几何门槛失败；损失下降47.6%未转化为GT收益')
    replace('3DGS / GGGS / 2DGS / 三角形 TGS · 2026-10-13 审查快照','3DGS / GGGS / 2DGS / 三角形 TGS · 2026-10-16累计证据')
    text,n=re.subn(r'<Text>factor64全49视角几何pilot.*?</Text>', '<Text>数学补充、数值核验及所有先前负结果保留。本轮严格42/7、同完整状态的30k pair及全部评价已结束：GT overall恶化.264%，TEST PSNR仅+.0045dB，资源耗时+32%。实际目标在共同非空射线下降47.6%，说明本pair的损失改善未转化为GT收益。停止该正则自动扩大和lambda搜索；下一独立问题是TGS细分是否保持实际光学量，尚无新结论。其余页为各阶段历史快照。</Text>',text,count=1,flags=re.S)
    assert n==1
    block='''    {tab === 38 && <Stack gap={18}>
      <H2>固定30k pair完成：几何采用门槛失败</H2>
      <Text>scan55，严格42 TRAIN／7 TEST，共同6999 checkpoint＋完整runtime sidecar起跑。control disabled与candidate .01各23000次Adam，最终30000反传/report/save而不做Adam。原NCC/geo、增密/reset/filter均保留；16个训练/提取/评价阶段全部exit0，不再有这些任务在等待。</Text>
      <Table headers={["指标", "Control", "Weight .01", "变化／判定"]} rows={[["DTU overall，3评价seed均值（mm，低更好）", ".3307163221", ".3315905070", "+.26433%，未达≤.99 ratio"], ["d2s／s2d ratio", "1 / 1", "1.00209755 / 1.00294454", "两方向均未过≤1"], ["TEST7 global RGB PSNR（dB）", "28.07059016", "28.07511820", "+.00452805，仅非劣守护通过"], ["TEST7 SSIM", ".899857896", ".900708420", "+.000850524"], ["TRAIN42 PSNR（dB）", "35.54241317", "35.46370216", "−.07871101"], ["基元数", "928975", "931140", "+.23305%，容量守护过"], ["全训练process wall（秒）", "2387.70", "3151.18", "+31.98%，≤2资源守护过"]]} />
      <Text>所有门槛都必须通过，因此总pilot_gate=false。不是显著图像改善，也不能据一对共享warmup声称普遍有害。三个GT seed只是评价采样重复，七相机相关；没有独立训练重复。资源wall包括full CSR及检查，是实际资源记账，不是控制严密的速度benchmark。PSNR主指标事前固定为global RGB；repository channel-mean TEST在此pair略降，不能混用。</Text>
      <H2>只读事后检查：目标下降与GT收益脱节</H2>
      <Text>所有42 train camera，每图512相同uniform像素，21504 ray/model；不训练、不改模型或质量判据。共同非空18344 ray的D²均值.177667541→.093169198，下降47.5598%。全均值.428462253→.349853801，其中empty常数贡献.274633949→.268914271；非空部分仍按全分母为.153828304→.080939530。不能只用全loss变化推断几何优化，也不能把这些条件均值当因果解释。</Text>
      <Text>空比例14.2904%→13.9927%；empty→nonempty151、反向87。84份NPZ的实际像素ID、counts、empty常数、info-D²与均值独立复算通过；code2只tail，不是无根证书。真实PNG是裁剪尺寸，已按实际尺寸核验。这个事后诊断不替换失败门槛，不用于选权重。</Text>
      <H2>保留证据，停止此正则扩大，检查新的TGS问题</H2>
      <Text>完整audit核290来源/319输入、46000逐步记录、两最终checkpoint与runtime sidecar digest、相机/采样序列和原始质量。审计通过是结果可信，不是方法通过。原C1证明/MP/warp数值结果仍有效；现在没有新的可采用质量方法。</Text>
      <Text>下一步先pin官方Triangle Splatting及native版本，对照面积/支持域保持与alpha场/渲染保持，阅读TGS+/MeshSplatting/MCMC守恒增密。官方main中点四分复制opacity/sigma已定向读到，但未pin/编译/实验；不据简单面积代数声称新颖或反驳Markov正确性。原公式及实际renderer对照通过后才考虑新方法训练。</Text>
      <Text size="small" tone="secondary">来源：HISTORY_GAP_PILOT_RESULT.md / results/history_gap_pilot_audit_v1.json / history_gap_posthoc_audit_v1.json。2026-10-16客户端日期，原UTC日志保留。同一LaTeX v44内置编译成功；本看板仅TypeScript语法检查，非SDK类型或app渲染认证。</Text>
    </Stack>}
'''
    marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
    stage.write_text(text,encoding='utf-8');g=dict(previous_sha256=old,staged_sha256=sha(stage),destination=str(t))
    guard.write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
if __name__=='__main__':main()
