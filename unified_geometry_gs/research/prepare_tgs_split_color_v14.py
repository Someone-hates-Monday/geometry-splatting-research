"""Freeze model/camera/source provenance before all new v14 GPU outputs."""
import datetime,json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,OUT,HELPER,RECEIPTS,EXPECTED,sha,write_new

def main():
    OUT.mkdir(exist_ok=True);target=OUT/'plan.json';assert not target.exists()
    old=ROOT/'experiments/tgs_cross_scene_v10/input_audit.json';audit=json.loads(old.read_text());assert audit['passed']
    metadata=json.loads((ROOT/'sources/tgs_cross_scene_v10/manifest.json').read_text())['files']
    models=[];modules={}
    for scene in ['flowers','room']:
        path=f'sources/tgs_cross_scene_v10/{scene}/point_cloud/iteration_7000/point_cloud_state_dict.pt'
        en=next(x for x in metadata if x['local']==path);assert sha(ROOT/path)==en['sha256']
        models.append(dict(scene=scene,path=path,sha256=en['sha256']))
    for name,rel in RECEIPTS.items():
        r=json.loads((ROOT/rel).read_text());assert sha(Path(r['module_file']))==r['module_sha256']==EXPECTED[name]
        modules[name]=dict(path=r['module_file'],sha256=r['module_sha256'],receipt=rel,receipt_sha256=sha(ROOT/rel))
    helpermanifest=json.loads((ROOT/'sources/tgs_real_checkpoint_v1/source_manifest.json').read_text())
    assert sha(HELPER)==next(e['sha256'] for e in helpermanifest['files'] if e['path']=='utils/sh_utils.py')
    scripts=['tgs_split_color_common_v14.py','prepare_tgs_split_color_v14.py','render_tgs_split_color_v14.py','audit_tgs_split_color_v14.py',
             'run_tgs_real_events.py','tgs_cross_scene_common_v10.py','tgs_sh_isolation_common_v12.py']
    plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(ROOT/'TGS_SPLIT_COLOR_PROTOCOL_V14.md'),
        models=models,modules=modules,helper_sha256=sha(HELPER),camera_audit_sha256=sha(old),
        scenes=[dict(scene=s['scene'],resolution=s['actual_resolution'],cameras=s['selected_cameras']) for s in audit['scenes']],
        scripts={n:sha(ROOT/n) for n in scripts},events_per_scene=8,forward_cap=76,
        model_camera_output_selection_frozen=True,training=False,benchmark=False)
    write_new(target,plan);print(json.dumps(dict(plan_frozen=True,path=str(target))),flush=True)

if __name__=='__main__':main()
