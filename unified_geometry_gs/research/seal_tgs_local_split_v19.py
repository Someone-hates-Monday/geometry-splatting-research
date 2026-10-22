"""Seal literal query correctness and narrow prepared-query cost measurements."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_local_split_v19'

def main():
    r=json.loads((OUT/'results.json').read_text());a=json.loads((OUT/'independent_audit.json').read_text());t=json.loads((OUT/'timing/independent_analysis.json').read_text())
    assert r['all_gates_passed'] and r['full_forwards']==68 and len(r['rows'])==32
    assert a['all_passed'] and a['zero_affected']==13 and t['all_verified'] and t['pair_count']==256
    dep=set()
    for name in ['plan.json','timing/plan.json']:
        p=json.loads((OUT/name).read_text())
        for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
        dep.update(p['inputs'])
    p=json.loads((OUT/'plan.json').read_text());assert sha(Path(p['native']['path']))==p['native']['sha256']
    build=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(build['path']))==build['sha256']
    dep.update(['TGS_LOCAL_SPLIT_RESULT_2026_10_21.md','TGS_LOCAL_SPLIT_CORRECTNESS_NOTE_V19.md',
        'audit_tgs_local_split_v19.py','analyze_tgs_local_split_timing_v19.py','plot_tgs_local_split_v19.py','seal_tgs_local_split_v19.py'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file());assert not (OUT/'evidence_manifest.json').exists()
    entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    dependencies=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(dep)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependencies,training=False,goal_complete=False,
        literal_fixed_input_query_outputs=True,prepared_cached_query_latency_reduced_on_all32_cases=True,
        full_training_speedup=False,GT_or_geometry_improvement=False,novelty_established=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependencies),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
