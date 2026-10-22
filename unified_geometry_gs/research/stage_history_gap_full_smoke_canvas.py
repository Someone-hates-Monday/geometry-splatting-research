from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');install=parser.parse_args().install
    stage=R/'canvas_history_gap_full_smoke_stage_v1.tsx';guard=R/'results/canvas_history_gap_full_smoke_guard_v1.json'
    if install:
        g=json.loads(guard.read_text(encoding='utf-8'));s=json.loads((R/'results/canvas_history_gap_full_smoke_syntax_v1.json').read_text(encoding='utf-8'))
        assert not s['errors'];t=Path(g['destination']);assert sha(t)==g['previous_sha256'] and sha(stage)==g['staged_sha256']
        t.write_bytes(stage.read_bytes());assert sha(t)==g['staged_sha256'];print(json.dumps(dict(status='updated_existing',sha256=sha(t),new_tabs_opened=0)));return
    t=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx')
    old='af588ae0af57713985136b90e3caaa2dcf5c6d5b0a2f79f63d04f209011ab886';assert sha(t)==old and not stage.exists()
    text=t.read_text(encoding='utf-8').replace('\r\n','\n')
    assert json.loads((R/'results/history_gap_full_smoke_audit_v1.json').read_text(encoding='utf-8'))['audit_gate_pass']
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,a;text=text.replace(a,b)
    replace('"真实共享参数接入"];','"真实共享参数接入", "原完整训练接入"];')
    replace('useState(36)','useState(37)')
    replace('真实共享参数接入与36个RGB短步完成；正则下降、RGB略高，完整训练及质量仍待验证',
        '原完整训练24步与零控制通过；30k配对试验运行中，质量结论待定')
    block='''    {tab === 37 && <Stack gap={18}>
      <H2>原train.py完整损失与实际增密：四组各6步</H2>
      <Text>scan55严格42 train／7 test，common6999 checkpoint＋runtime sidecar恢复参数、优化器、filter、增密统计、camera queue与RNG。四进程完整状态digest一致；初始676277基元，原有752个+∞ opacity保留。原NCC .6／geo .02，depth-normal权重按既定recipe为0；训练render、NCC邻居和filter池均有实际相机guard。</Text>
      <Text>disabled两次、computed-zero与weight.01各7000–7005六次Adam，共24更新。未冻结拓扑/filter；原7000增密后各677291基元。首步完整梯度在增密前保存；增密替换参数可清除相应梯度，因此不能说每个反传贡献都进入了该次Adam。</Text>
      <H2>首步完整共享梯度的独立复算</H2>
      <Table headers={["参数组", "disabled重复相对L2", "zero对disabled相对L2", ".01对disabled相对L2"]} rows={[["位置", "1.7804e−7", "1.5717e−7", "32.8867%"], ["opacity", "1.3514e−7", "1.5020e−7", "15.2524%"], ["scale", "6.9716e−5", "4.6291e−5", "14.8272%"], ["rotation", "2.4800e−4", "3.0048e−4", "22.7230%"]]} />
      <Text>全部参数组zero≤max(1e−5,5×disabled重复)，是预设工程控制，非统计等价。color只有约2e−7量级重复噪声，SG/appearance组零差。相机序列0022/0033/0011/0019/0020/0005完全一致，weighted两组128个均匀像素ID相同，额外采样不消耗训练RNG。所有其他参数/梯度/Adam有限，当前opaque mask在普通Adam前后不变。</Text>
      <H2>已启动固定30k配对pilot</H2>
      <Text>两组均从common6999重新起跑：control关闭正则，history_gap权重.01；原始schedule与增密/reset/filter保留。2026-10-15此快照时控制组正在运行，之后队列自动推进候选训练、42-view mesh、官方GT seeds0/1/2以及TRAIN42／TEST7图像评价。当前没有新质量结果。</Text>
      <Text>预设推进门槛全需通过：GT overall ratio≤.99，两方向距离各≤1，TEST PSNR下降≤.10dB、SSIM下降≤.001，基元ratio≤1.05，全训练资源耗时ratio≤2。单一共享warmup配对是探索，场景有先前研究暴露；GT采样种子和相关视图不是独立训练重复。失败保留，不自动扩权重搜索。</Text>
      <Text size="small" tone="secondary">来源：history_gap_full_smoke_v1/summary.json；独立audit复核63项来源检查与46产物，重算原始梯度。首区间省略forward且含冷开销，不能用来声称完整训练加速。HISTORY_GAP_PILOT_PROTOCOL.md／history_gap_pilot_v1/manifest.json为固定协议与动态运行真值；此看板为静态快照。仅TS语法检查，非SDK/runtime验证。</Text>
    </Stack>}
'''
    marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker)
    stage.write_text(text,encoding='utf-8');g=dict(previous_sha256=old,staged_sha256=sha(stage),destination=str(t))
    guard.write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
if __name__=='__main__':main()
