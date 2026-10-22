"""Append-only provenance seal, retaining original ABI load failure and repair."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_ray_order_v16a'

def main():
    plan=json.loads((OUT/'plan.json').read_text())
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
    result=json.loads((OUT/'results.json').read_text());audit=json.loads((OUT/'quality_audit.json').read_text())
    assert result['full_forwards']==4 and result['all_correctness_gates'] and audit['all_passed']
    assert audit['descriptive_engineering_lead']=={'physical':False,'legacy':False}
    assert (ROOT/'experiments/tgs_ray_order_v16/fixtures_failure.txt').exists()
    sources=['TGS_RAY_ORDER_PROTOCOL_V16.md','TGS_RAY_ORDER_RESULT_2026_10_21.md',
        'run_tgs_ray_order_v16.py','tgs_ray_order_v16.cu','run_tgs_ray_order_v16a.py','tgs_ray_order_v16a.cu',
        'audit_tgs_ray_order_v16a.py','plot_tgs_ray_order_v16a.py','seal_tgs_ray_order_v16a.py',
        'tgs_image_metrics_v11.py','TGS_PLANAR_SPLIT_ORDER_NOTE_V15.md','TGS_WINDOW_PRIOR_WORK_V15.md']
    deps=set(sources)|set(plan['inputs'])
    files=sorted(p for d in [ROOT/'experiments/tgs_ray_order_v16',OUT] for p in d.rglob('*') if p.is_file())
    assert not (OUT/'evidence_manifest.json').exists()
    entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    dependency=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p)) for p in sorted(deps)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependency,
        native_module=plan['native'],training=False,quality_gain=False,goal_complete=False,
        original_failure_retained=True,abi_repair='long long tensor interface -> int64_t; independent new directory'))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependency),manifest_sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
