"""Seal loss-module correctness with integration, timing and quality claims unset."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_local_image_loss_v24'

def main():
    assert not (OUT/'evidence_manifest.json').exists() and not any(OUT.glob('*failure*'))
    r=json.loads((OUT/'results.json').read_text());a=json.loads((OUT/'receipt_audit.json').read_text());fixture=json.loads((OUT/'fixtures.json').read_text())
    assert r['all_passed'] and len(r['rows'])==32 and len(r['batches'])==4 and fixture['all_passed'] and len(fixture['batch']['rows'])==8
    assert r['new_native_forwards']==0 and not r['training'] and not r['renderer_integration'] and not r['timing']
    assert a['all_verified'] and a['rows']==32 and a['CPU_GPU_blend_sign_agreements']==64 and a['blend_comparisons']==64
    assert a['maxima']['CPU_map_error']<1e-10 and a['maxima']['score_error']<1e-13
    dep=set()
    for name in ['plan.json','receipt_audit.json']:
        p=json.loads((OUT/name).read_text())
        for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
        dep.update(p['inputs'])
    b=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(b['path']))==b['sha256'] and sha(ROOT/'tgs_local_image_loss_v24.cu')==b['source_sha256']
    dep.update(['TGS_LOCAL_IMAGE_LOSS_RESULT_2026_10_21.md','seal_tgs_local_image_loss_v24.py'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file());entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    dependencies=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(dep)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependencies,real_image_cases=32,synthetic_cases=8,new_native_forwards=0,
        local_full_GPU_maps_literal=True,CPU_GPU_values_within_frozen_tolerances=True,training=False,backward_exercised=False,
        renderer_integration=False,performance_measured=False,quality_gain_established=False,novelty_established=False,goal_complete=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependencies),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
