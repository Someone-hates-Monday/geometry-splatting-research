"""Validate balanced timings and compute descriptive paired bootstrap intervals."""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_backward_cost_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def local(path):
    prefix='/mnt/c/Users/Someone/Desktop/SuGaR-main/'
    return ROOT.parents[1]/path[len(prefix):] if path.startswith(prefix) else Path(path)


def main():
    plan=json.loads((OUT/'protocol.json').read_text());data=json.loads((OUT/'results.json').read_text())
    assert data['protocol_sha256']==sha(OUT/'protocol.json')
    assert plan['script_sha256']==sha(ROOT/'benchmark_tgs_backward.py')
    assert not plan['debug'] and not plan['training'] and not plan['instrumentation']
    for entry in plan['modules'].values():assert sha(local(entry['path']))==entry['sha256']
    rows=data['rows'];assert len(rows)==108
    variants=plan['variants'];assert len(plan['balanced_blocks'])==36
    for block_id,block in enumerate(plan['balanced_blocks']):
        subset=[r for r in rows if r['block']==block_id];assert len(subset)==3
        for position,r in enumerate(subset):
            assert r['camera']==block['camera'] and r['position']==position and r['variant']==block['variants'][position]
            for key in ['forward_event_ms','backward_event_ms','forward_wall_ms','backward_wall_ms','pair_wall_ms']:
                assert np.isfinite(r[key]) and r[key]>0
            assert r['pair_wall_ms']>=r['forward_wall_ms']+r['backward_wall_ms']
            assert r['peak_allocated_bytes']-r['resident_bytes']==r['incremental_peak_bytes']
    grouped=[];comparisons=[];rng=np.random.default_rng(plan['bootstrap_seed'])
    for camera in [0,1]:
        blocks=[i for i,b in enumerate(plan['balanced_blocks']) if b['camera']==camera]
        orders=collections.Counter(tuple(plan['balanced_blocks'][i]['variants']) for i in blocks)
        assert set(orders)==set(itertools.permutations(variants)) and set(orders.values())=={3}
        assert len(set(r['tile_instances'] for r in rows if r['camera']==camera))==1
        for variant in variants:
            selected=[r for r in rows if r['camera']==camera and r['variant']==variant]
            assert len(selected)==18
            assert collections.Counter(r['position'] for r in selected)=={0:6,1:6,2:6}
            metrics={}
            for key in ['forward_event_ms','backward_event_ms','forward_wall_ms','backward_wall_ms','pair_wall_ms',
                        'peak_allocated_bytes','incremental_peak_bytes','peak_reserved_bytes']:
                values=np.array([r[key] for r in selected]);q=np.quantile(values,[.25,.5,.75])
                metrics[key]=dict(q25=float(q[0]),median=float(q[1]),q75=float(q[2]),min=float(values.min()),max=float(values.max()))
            grouped.append(dict(camera=camera,image=plan['cameras'][camera]['img_name'],variant=variant,n=18,metrics=metrics))
        for variant in ['geometry','piecewise']:
            metrics={}
            for key in ['backward_event_ms','backward_wall_ms','pair_wall_ms']:
                ratios=[];differences=[]
                for block in blocks:
                    match={r['variant']:r for r in rows if r['block']==block}
                    ratios.append(match[variant][key]/match['control'][key])
                    differences.append(match[variant][key]-match['control'][key])
                ratios=np.array(ratios);differences=np.array(differences)
                samples=rng.integers(0,len(ratios),size=(10000,len(ratios)))
                boot=np.median(ratios[samples],axis=1)
                metrics[key]=dict(paired_median_ratio=float(np.median(ratios)),
                    descriptive_bootstrap_95_ratio=np.quantile(boot,[.025,.975]).tolist(),
                    paired_median_difference_ms=float(np.median(differences)),
                    raw_paired_ratios=ratios.tolist())
            comparisons.append(dict(camera=camera,variant=variant,relative_to='control',metrics=metrics))
    assert all(g['forward_rgb_aux_radii_identical'] for g in data['guards'])
    result=dict(evidence_audit_passed=True,groups=grouped,comparisons=comparisons,full_scene_guards=data['guards'],
                measured_pairs=108,telemetry_start=plan['telemetry_start'],telemetry_end=data['telemetry_end'],
                limitations=['one session, one GPU, one checkpoint and two train views; intervals are descriptive conditional within-session intervals',
                             'events span native API allocations, internal item synchronizations and host-induced GPU gaps; not pure kernel throughput',
                             'no loss computation, optimizer, data loader, densification or quality comparison',
                             'Torch allocator peak excludes registers, local memory and non-Torch driver allocations'],
                training=False,quality_or_training_speed_gain_established=False)
    (OUT/'summary_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(evidence_audit_passed=True,measured_pairs=108,
        medians=[dict(camera=g['camera'],variant=g['variant'],backward_wall_ms=g['metrics']['backward_wall_ms']['median'],
                      pair_wall_ms=g['metrics']['pair_wall_ms']['median'],peak_allocated=g['metrics']['peak_allocated_bytes']['median']) for g in grouped],
        ratios=[dict(camera=c['camera'],variant=c['variant'],backward=c['metrics']['backward_wall_ms']['paired_median_ratio'],
                     backward_ci=c['metrics']['backward_wall_ms']['descriptive_bootstrap_95_ratio'],
                     pair=c['metrics']['pair_wall_ms']['paired_median_ratio']) for c in comparisons]),indent=2))


if __name__=='__main__':main()
