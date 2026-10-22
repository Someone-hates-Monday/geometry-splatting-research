"""Freeze source intervention evidence and old/fresh cameras before fresh payload/output."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_sh_isolation_v12';SOURCE=ROOT/'sources/tgs_sh_isolation_v12'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);SOURCE.mkdir(exist_ok=True);target=OUT/'input_plan.json';assert not target.exists();original=ROOT/'sources/tgs_native_diagnostic_v1/native';generated=ROOT/'experiments/tgs_subdivision_v1/sh_direction_v1/fixed_src';receipts={v:json.loads((ROOT/p).read_text()) for v,p in {'archived_native':'experiments/tgs_subdivision_v1/native_build/receipt_9d974f09b721.json','sh_only':'experiments/tgs_subdivision_v1/sh_direction_v1/fixed_build_receipt.json'}.items()};edits=[]
    old='(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)';new='glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)'
    for name in ['forward.cu','backward.cu']:
        a=original/'cuda_rasterizer'/name;b=generated/name;text=a.read_text();assert text.count(old)==1 and b.read_text()==text.replace(old,new);edits.append(dict(original=a.relative_to(ROOT).as_posix(),generated=b.relative_to(ROOT).as_posix(),original_sha256=sha(a),generated_sha256=sha(b),replacement_count=1))
    for key in ['native_commit','glm_commit','torch','cuda','python','flags']:assert receipts['archived_native'][key]==receipts['sh_only'][key]
    assert receipts['archived_native']['source_edits'] is False and len(receipts['sh_only']['source_edits'])==2
    v10=json.loads((ROOT/'experiments/tgs_cross_scene_v10/input_audit.json').read_text());oldsel=json.loads((ROOT/'sources/tgs_gt_identity_v11/selection.json').read_text());rows=[]
    for scene in ['flowers','room']:
        cameras=json.loads((ROOT/'sources/tgs_cross_scene_v10'/scene/'cameras.json').read_text());test=sorted(cameras,key=lambda c:c['img_name'])[::8];oldrow=next(r for r in oldsel['rows'] if r['scene']==scene);used={c['img_name'] for sc in v10['scenes'] if sc['scene']==scene for c in sc['selected_cameras']};fresh=min([c for c in test if c['img_name'] not in used],key=lambda c:hashlib.sha256(('tgs_sh_isolation_v12/'+scene+'/'+c['img_name']).encode()).hexdigest())
        for label,c in [('old_posthoc',oldrow['camera']),('fresh',dict(fresh,split='test'))]:rows.append(dict(scene=scene,cohort=label,camera=c,test_index=[x['img_name'] for x in test].index(c['img_name'])))
    out=dict(rows=rows,source_edits=edits,receipt_comparison_keys=['native_commit','glm_commit','torch','cuda','python','flags'],protocol_sha256=sha(ROOT/'TGS_SH_ISOLATION_PROTOCOL_V12.md'),script_sha256=sha(Path(__file__)),old_v11_manifest_sha256=sha(ROOT/'experiments/tgs_gt_quality_v11/evidence_manifest.json'),old_v10_manifest_sha256=sha(ROOT/'experiments/tgs_cross_scene_v10/evidence_manifest.json'),camera_selection_rule='Minimum SHA256(tgs_sh_isolation_v12/scene/name), excluding v10 selected cameras',training=False,fresh_output_inspected=False);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cameras=[dict(scene=r['scene'],cohort=r['cohort'],name=r['camera']['img_name'],test_index=r['test_index']) for r in rows],source_only_edit_verified=True)),flush=True)
if __name__=='__main__':main()
