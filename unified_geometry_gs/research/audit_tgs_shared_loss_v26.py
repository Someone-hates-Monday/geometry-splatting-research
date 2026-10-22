"""CPU audit of cached/unshared score identities, cache bytes and persisted receipts."""
import json
import numpy as np
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new,exact
OUT=ROOT/'experiments/tgs_shared_loss_v26';V24=ROOT/'experiments/tgs_local_image_loss_v24'

def main():
    p=json.loads((OUT/'plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    r=json.loads((OUT/'results.json').read_text());old=json.loads((V24/'results.json').read_text());assert r['all_passed'] and r['new_native_forwards']==0 and len(r['rows'])==32
    assert r['source_sha256']==sha(ROOT/'run_tgs_shared_loss_v26.py') and json.loads((OUT/'fixtures.json').read_text())['all_passed']
    lookup={(x['scene'],x['view'],x['event']):x for x in old['rows']};assert len(lookup)==32
    max_score=max(max(x['absolute_error']) for x in r['rows']);max_map=max(x['max_CPU_map_error'] for x in r['rows']);batches=[]
    for row in r['rows']:
        ref=lookup[(row['scene'],row['view'],row['event'])];assert exact(np.array(row['score'],dtype=np.float64),np.array(ref['score'],dtype=np.float64))
        assert row['trial_hash']==ref['trial_hash'] and row['cached_unshared_scores_and_maps_literal'] and row['local_full_maps_literal'] and row['normal_debug_scalars_literal']
        assert row['max_CPU_map_error']<=p['map_tolerance'] and all(e<=p['score_atol']+p['score_rtol']*abs(v) for e,v in zip(row['absolute_error'],row['expected']))
    for item in r['batches']:
        cell=json.loads((OUT/f"{item['scene']}_{item['view']}.json").read_text());assert len(cell['rows'])==8 and cell['cache_preps']==1 and len(cell['calls'])==3
        assert all(c['scores_literal_unshared'] and c['maps_literal_unshared'] for c in cell['calls'])
        assert cell['temporary_vertical_bytes']*8==cell['unshared_vertical_bytes']*3
        pixels=1256*828 if item['scene']=='flowers' else 1557*1038;assert cell['cache_bytes']==9*pixels*8==item['cache_bytes']
        batches.append(dict(scene=item['scene'],view=item['view'],shared_cache_bytes=cell['cache_bytes'],trial_vertical_bytes=cell['temporary_vertical_bytes'],unshared_trial_vertical_bytes=cell['unshared_vertical_bytes']))
    out=dict(all_verified=True,real_cases=32,cached_GPU_calls=12,reference_GPU_calls=12,cache_preps=4,new_native_forwards=0,
        max_CPU_score_error=max_score,max_CPU_map_error=max_map,cached_unshared_scores_literal=True,cached_unshared_maps_runtime_gate_passed=True,
        independent_map_GPU_rerun=False,training=False,batches=batches,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'plan.json',OUT/'results.json',V24/'results.json',Path(__file__)]})
    write_new(OUT/'receipt_audit.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':main()
