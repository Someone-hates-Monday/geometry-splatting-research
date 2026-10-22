"""Freeze v7 cameras/pixel probes before running native kernels."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_real_work_v7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not OUT.exists();OUT.mkdir()
    v6=ROOT/'experiments/tgs_compensated_edge_v6/evidence_manifest.json';assert sha(v6)=='95114628796d6dc005335732b11e63df450c91593d3e6a0369b2e4cfe0bae5c2'
    for e in json.loads(v6.read_text())['files']:assert sha(ROOT/e['path'])==e['sha256']
    v5=ROOT/'experiments/tgs_phase_cost_v5';m=v5/'evidence_manifest.json';assert sha(m)=='c8ab7c32e93b5aca6a830d41fdc04b2bc1e8205b852a9a0a39f6bfdd8a9af08d'
    hashes={e['path']:e['sha256'] for e in json.loads(m.read_text())['files']}
    for p in ['oracle/build_receipt.json','generated/native/cuda_rasterizer/stable_window.cuh']:
        pp=v5/p;assert sha(pp)==hashes[pp.relative_to(ROOT).as_posix()]
    auditp=ROOT/'experiments/tgs_real_event_v1/input_audit.json';audit=json.loads(auditp.read_text());old=[audit['selected_cameras'][i] for i in [2,3,0,1]]
    cp=ROOT/'sources/tgs_real_checkpoint_v1/stump/cameras.json';allcam=json.loads(cp.read_text());test=sorted(allcam,key=lambda c:c['img_name'])[::8]
    unused=[c for c in test if c['img_name'] not in {x['img_name'] for x in old}]
    chosen=min(unused,key=lambda c:hashlib.sha256(('tgs_real_work_v7/'+c['img_name']).encode()).hexdigest());cams=old+[dict(chosen,split='test',heldout_for_v7=True)]
    w,h=audit['actual_resolution'];rng=np.random.default_rng(202610209)
    pixels=[np.sort(rng.choice(w*h,512,replace=False)).tolist() for _ in cams]
    plan=dict(protocol_sha256=sha(ROOT/'TGS_REAL_WORK_PROTOCOL_V7.md'),source_sha256=sha(ROOT/'tgs_real_work_probe.cu'),prepare_sha256=sha(Path(__file__)),v6_manifest_sha256=sha(v6),v5_manifest_sha256=sha(m),camera_source_sha256=sha(cp),audit_source_sha256=sha(auditp),resolution=[w,h],cameras=cams,sample_pixels=pixels,pixel_seed=202610209,cache_selection='SHA256(tgs_real_work_v7/cache/<img_name>/<triangle_id>) smallest512 visible plus32 largest Bx*By (stable ID ties)',native_receipt_sha256=sha(v5/'oracle/build_receipt.json'),native_module_expected='e8aee33c860f346247a1e48a9fc61f70a7b035ef2ea55b800c5960280ad24ab8',training=False,timing=False,renderer_candidate=False)
    (OUT/'input_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(cameras=[c['img_name'] for c in cams],new=chosen['img_name'],plan_sha256=sha(OUT/'input_plan.json'))),flush=True)
if __name__=='__main__':main()
