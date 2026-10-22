from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');install=parser.parse_args().install;guard=R/'results/canvas_history_gap_adapter_guard_v1.json';stage=R/'canvas_history_gap_adapter_stage_v1.tsx'
    if install:
        g=json.loads(guard.read_text(encoding='utf-8'));s=json.loads((R/'results/canvas_history_gap_adapter_syntax_v1.json').read_text(encoding='utf-8'));assert not s['errors'];t=Path(g['destination']);assert sha(t)==g['previous_sha256'] and sha(stage)==g['staged_sha256'];t.write_bytes(stage.read_bytes());assert sha(t)==g['staged_sha256'];print(json.dumps(dict(status='updated_existing',sha256=sha(t),new_tabs_opened=0)));return
    t=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx');old='e4c5379b532b15dd6e2fb43c7b2fcf8c6cd6a9b57bd3d9cf2604b5794bac08fa';assert sha(t)==old;text=t.read_text(encoding='utf-8').replace('\r\n','\n');assert not stage.exists();audit=json.loads((R/'results/history_gap_adapter_audit_v1.json').read_text(encoding='utf-8'));assert audit['audit_gate_pass']
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,a;text=text.replace(a,b)
    replace('"Warp损失核对照"];','"Warp损失核对照", "真实共享参数接入"];');replace('useState(35)','useState(36)');replace('Warp完整loss批次11.99ms，原梯度门槛全过；下一步真实GPU训练adapter，质量仍未证','真实共享参数接入与36个RGB短步完成；正则下降、RGB略高，完整训练及质量仍待验证')
    block='''    {tab === 36 && <Stack gap={18}>
      <H2>真实GaussianModel接入：共享Float32梯度与短步控制</H2>
      <Text>scan37/55/63原30k PLY各camera0000。public renderer与private query/CSR的RGB/alpha/normal/depth/slope/radii逐位一致；实际query/AMP/IDs/source/raw对上旧探针。仅采样行GPU打包到Float64图，梯度回写原共享Float32参数；MP loss梯度聚合误差分别4.69e−8/2.57e−8/3.85e−8，各块全过，没有非目标color/appearance/SG正则梯度。</Text>
      <H2>原模型边界与空射线明确处理</H2>
      <Text>原PLY已含3431/1908/2443个+∞ opacity logits，未替换或删除。初始all-finite gate失败保留；兼容修订将这些坐标固定为sigmoid1，梯度严格0、mask/值不变，其余参数/所有梯度/Adam状态有限。不在无穷logit处主张C1，也未证明原值导致过去失败。</Text>
      <Text>三批样本164/162/167，实际empty0；scan37另有现场全空ray。空列表loss=(log4)²≈1.921812、梯度0，保留分母。另36真实输入＋1空的重放fixture核验mixed loss误差5.56e−17、梯度缩放误差8.29e−8，全空梯度精确0；fixture不是新capture场景。</Text>
      <H2>36次fresh-Adam RGB-only内存更新</H2>
      <Text>每场景两次disabled、computed-zero、weight.01各3步，每臂恢复同一参数对象/fresh state，固定拓扑/filter/camera/采样。optimizer绑定/RNG/finite检查通过，原参数和PLY恢复。零控制按repeat-noise工程界通过，非bit或统计等价；scan37零组差1.054%，disabled重复.678%。三步zero参数max差≤8.43e−6。</Text>
      <Table headers={["场景", "正则项首→末观察", "disabled／positive诊断wall中位（ms）"]} rows={[["37", ".163931 → .138692", "70.71 / 111.31"], ["55", ".244935 → .224625", "47.25 / 85.66"], ["63", ".123730 → .104279", "31.30 / 64.38"]]} />
      <Text>positive RGB混合损失最后略高于disabled：.0243716 vs .0243528、.0161611 vs .0161410、.0110433 vs .0110286。只能说明正则生效与局部tradeoff，不是图像/几何改善。capture单独42.32/30.51/24.20ms，prepare/pack/loss约5.4–6.4ms；包括同步与finite/binding检查，另存冷观测，不含产物写盘/状态恢复。不能把12ms组件计时当训练开销。</Text>
      <H2>下一步完整训练recipe</H2>
      <Text>接原完整NCC/normal等loss与6999 checkpoint/runtime sidecar，严格42 train/7 test，先完整恢复/disabled/zero及短步，再预设配对pilot、GT/heldout/容量评价。当前49-trained PLY只用于开发接入，不能作新heldout证据。原147view GT负结果不变，整体目标未完成。</Text>
      <Text size="small" tone="secondary">HISTORY_GAP_ADAPTER_RESULT.md / adapter_v1严格finite失败、v2参考选错、v3完成；mixed empty fixture和独立audit通过。2026-10-15 Asia/Shanghai。原LaTeX原位编译；看板仅语法检查，非SDK/runtime验证。</Text>
    </Stack>}
'''
    marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker);stage.write_text(text,encoding='utf-8');g=dict(previous_sha256=old,staged_sha256=sha(stage),destination=str(t));guard.write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
if __name__=='__main__':main()
