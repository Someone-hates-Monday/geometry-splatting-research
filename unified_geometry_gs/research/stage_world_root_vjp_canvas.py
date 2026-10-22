from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;target=Path('C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx');old='8fd28af48537363102b3eb9dd32483b34497c7e15a1de55477e6f2867ce4b66e';raw=target.read_bytes();assert hashlib.sha256(raw).hexdigest()==old;text=raw.decode('utf-8').replace('\r\n','\n');audit=json.loads((R/'results/world_root_vjp_audit_v1.json').read_text(encoding='utf-8'));v2=json.loads((R/'results/world_root_vjp_v2_checks.json').read_text(encoding='utf-8'));assert audit['gate_pass'] and v2['gate_pass']
def replace(a,b):
    global text
    assert text.count(a)==1,a;text=text.replace(a,b)
replace('"动态严格求根"];','"动态严格求根", "世界参数链VJP"];');replace('useState(29)','useState(30)');replace('全部未认证原有效射线完成修正或无根判定；世界梯度与GT仍待验证','CPU连续模型的完整参数链VJP细化通过；完整CUDA与GT待验证')
block='''    {tab === 30 && <Stack gap={18}>
      <H2>深度证书不自动给出梯度精度：内部根需保留并细化</H2>
      <Text>全部原固定891完整射线 / 51,524源节点，每个基元位置3、原log-scale3、quaternion4、opacity-logit1，共11参数。含3D滤波激活、归一化及屏幕补偿。每分支566,764梯度元素保存，原9异常均在集合中。MP140 world covariance逆+解析矩阵链，MP240稳定根/IFT参考；实现自己细化，不接受参考根输入。</Text>
      <Text>连续模型经过实际捕获F32值：尺度用激活基点比值、q用归一化差、AMP用sigmoid/滤波/补偿基点比值。不是字面IEEE离散程序的微分，也不是悄悄换成未舍入raw_physical模型；相机/query/filter/名单固定，不含拓扑变化导数。</Text>
      <Table headers={["adjoint评价点/算法", "通过 / 全891", "最大完整射线相对L2"]} rows={[
        ["既存GPU内部候选 / factor", "858 / 891；33失败", "4.47157e-3"],
        ["公共F32深度中心 / factor", "345 / 891；546失败", "4.48923e-3"],
        ["自行细化内部根 / factor", "891 / 891", "1.86370e-7"],
        ["自行细化内部根 / dense inverse", "891 / 891", "9.86379e-7"]
      ]} />
      <Text size="small" tone="secondary">来源world_root_vjp_v1，2026-10-15 Asia/Shanghai。完整射线相对L2门槛1e-5，xyz/log_scale/quaternion/opacity-logit各block门槛1e-4。全部参考规则可微、梯度有限；公共分支含原候选误差与输出舍入，不把全部差异仅归因于一次舍入。</Text>
      <H2>来源、数组和通用辅助函数核验</H2>
      <Text>13来源/51产物、ZIP、全部source/root匹配和逐ray/block统计独立重算通过；8项SymPy层partials通过。真实数据所有screen kernel=0，不能据此证明非零屏幕补偿分支的实测有效性。</Text>
      <Text>V2令补偿基点独立于requires_grad标志，避免无梯度参数前向与VJP使用不同函数。V1真实结果与源保留；V2全部891精细梯度位不变。另以明确合成的kernel0/.3/1做完整根中心差分，末级相对误差≤5.74e-10；合成检查不是GT/训练成果。</Text>
      <H2>当前门槛与下一项</H2>
      <Text>两个精细分支都通过，尚无一般性factor优势、训练收益或学术新颖性证明。CPU完整参数链不是完整CUDA训练后向；必须接好内部状态与真实梯度链，检查完整成本。固定scan37/55/63各完整49视角、固定TSDF与官方GT seed0/1/2的质量对照协议已写，状态pending。</Text>
      <Text size="small" tone="secondary">WORLD_ROOT_VJP_RESULT.md / world_root_vjp_audit_v1.json / world_root_vjp_v2_checks.json / FROZEN_CERTIFIED_MESH_PROTOCOL.md。主session16634与V2核验exit0。原LaTeX原位编译；此静态看板只做语法检查，SDK/runtime未验证。</Text>
    </Stack>}
''';marker='  </Stack>;\n}';assert text.count(marker)==1;text=text.replace(marker,block+marker);p=R/'canvas_world_root_vjp_stage_v1.tsx';assert not p.exists();p.write_text(text,encoding='utf-8');g=dict(previous_sha256=old,staged_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination=str(target));(R/'results/canvas_world_root_vjp_guard_v1.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8');print(json.dumps(g))
