"""Seal second-view counterexamples and finite-halo loss math without quality claims."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_view_transfer_v23'

def main():
    assert not (OUT/'evidence_manifest.json').exists()
    r=json.loads((OUT/'results.json').read_text());a=json.loads((OUT/'analysis.json').read_text());m=json.loads((OUT/'loss_locality_results.json').read_text())
    assert r['all_passed'] and r['native_forwards']==20 and len(r['rows'])==16 and r['accepted_mutations']==0
    assert a['all_verified'] and len(a['rows'])==32 and m['all_passed'] and len(m['rows'])==32 and m['new_GPU_forwards']==0
    assert sha(ROOT/'run_tgs_view_transfer_v23.py')==r['source_sha256'] and sha(ROOT/'analyze_tgs_view_transfer_v23.py')==a['source_sha256']
    assert sha(OUT/'results.json')==a['render_results_sha256'] and sha(ROOT/'verify_tgs_image_loss_locality_v23.py')==m['source_sha256']
    assert sha(OUT/'loss_locality_plan.json')==m['plan_sha256']
    assert not any(OUT.glob('*failure*'))
    assert len({(x['scene'],x['event']) for x in r['rows']})==16 and len({(x['scene'],x['view'],x['event']) for x in a['rows']})==32
    for row in r['rows']:
        assert all(row['checks'].values()) and row['visible_cache_fields_literal'] and row['all_tile_lists_literal']
        assert row['delta_SSE']==row['batch_score']==row['full_gpu_score']
    for row in m['rows']:
        assert row['local_maps_literal'] and row['zero_difference_outside_output_region'] and row['PNG8_SSE_exact']
        assert all(v['absolute_residual']<=v['reduction_roundoff_bound'] for v in row['reductions'].values())
    dep=set()
    for name in ['plan.json','loss_locality_plan.json']:
        p=json.loads((OUT/name).read_text())
        for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
        dep.update(p['inputs'])
    for phase in ['tgs_batch_score_v22','tgs_aux_ablation_v21']:
        b=json.loads((ROOT/f'experiments/{phase}/build_receipt.json').read_text());assert sha(Path(b['path']))==b['sha256'];dep.add(str(Path(b['path']).relative_to(ROOT)))
    static=json.loads((ROOT/'sources/tgs_subdivision_static_v1/manifest.json').read_text())
    for name in ['train.py','utils/loss_utils.py','arguments/__init__.py']:
        en=next(x for x in static['files'] if x['repo']=='triangle-splatting' and x['path']==name)
        assert sha(ROOT/'sources/tgs_subdivision_static_v1/triangle-splatting'/name)==en['sha256']
    f=json.loads((OUT/'figure_provenance.json').read_text());assert sha(ROOT/'plot_tgs_view_transfer_v23.py')==f['script_sha256'] and sha(OUT/'analysis.json')==f['analysis_sha256']
    dep.update(['TGS_VIEW_TRANSFER_RESULT_2026_10_21.md','plot_tgs_view_transfer_v23.py','seal_tgs_view_transfer_v23.py'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file());entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    dependencies=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(dep)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependencies,new_native_forwards=20,old_view_proposals=16,total_case_view_metrics=32,
        finite_halo_map_checks=32,training=False,backward_exercised=False,accepted_mutations=0,independent_scene_test=False,
        model_quality_gain_established=False,novelty_established=False,goal_complete=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependencies),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
