"""Seal coverage/appearance stress and preserve the setup-inclusive negative case."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_local_stress_v20'

def main():
    result=json.loads((OUT/'results.json').read_text());audit=json.loads((OUT/'independent_audit.json').read_text());timing=json.loads((OUT/'timing/independent_analysis.json').read_text())
    assert result['all_gates_passed'] and result['full_forwards']==26 and len(result['rows'])==16
    assert audit['all_passed'] and audit['max_coverage']>.23 and timing['all_verified'] and timing['pair_count']==128
    assert next(v for v in timing['views'] if v['scene']=='flowers')['estimated_setup_inclusive_ratio']<1
    deps=set()
    for name in ['plan.json','timing/plan.json']:
        p=json.loads((OUT/name).read_text())
        for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
        deps.update(p['inputs'])
    p=json.loads((OUT/'plan.json').read_text())
    for name in ['native','local']:assert sha(Path(p[name]['path']))==p[name]['sha256']
    deps.update(['TGS_LOCAL_STRESS_RESULT_2026_10_21.md','audit_tgs_local_stress_v20.py','analyze_tgs_local_stress_timing_v20.py','plot_tgs_local_stress_v20.py','seal_tgs_local_stress_v20.py'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file());assert not (OUT/'evidence_manifest.json').exists()
    entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    dependencies=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(deps)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependencies,training=False,goal_complete=False,
        stress_RGB_alpha_paths_literal=True,all_prepared_case_medians_faster=True,flowers_setup_inclusive8query_estimate_slower=True,
        output_matched_full_scene_control_missing=True,novelty_established=False,GT_or_geometry_gain=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependencies),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
