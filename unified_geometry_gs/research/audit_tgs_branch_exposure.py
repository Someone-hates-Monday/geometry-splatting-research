"""Audit raw per-triangle integer counters, support and calibration contracts."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_branch_exposure_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def local(s):
    prefix='/mnt/c/Users/Someone/Desktop/SuGaR-main/'
    return ROOT.parents[1]/s[len(prefix):] if s.startswith(prefix) else Path(s)


def main():
    plan=json.loads((OUT/'protocol.json').read_text());data=json.loads((OUT/'results.json').read_text())
    build=json.loads((OUT/'build_receipt.json').read_text())
    assert sha(OUT/'protocol.json')==data['protocol_sha256']
    assert sha(ROOT/'measure_tgs_branch_exposure.py')==plan['script_sha256']
    assert sha(ROOT/'tgs_branch_counter_backend.py')==build['generator_sha256']
    assert sha(local(build['module_file']))==build['module_sha256']==plan['module_sha256']==data['counter_module_sha256']
    for e in build['sources']:assert sha(local(e['path']))==e['sha256']
    assert sha(OUT/'generated/rasterize_points.h')==build['generated_header_sha256']
    assert not build['timing_eligible'] and not plan['performance_claim']
    benchmark=json.loads((ROOT/'experiments/tgs_backward_cost_v1/protocol.json').read_text())
    assert plan['workload']==benchmark['workload']
    calibrations=json.loads((OUT/'calibration.json').read_text())
    assert calibrations==data['calibrations'] and [c['name'] for c in calibrations]==plan['selftests']
    for c in calibrations:
        with np.load(OUT/('calibration_'+c['name']+'.npz')) as z:
            counts=z['counts'];assert counts.tolist()==c['counts']
            assert np.array_equal(z['control_rgb'],z['trace_rgb'])
            assert np.array_equal(counts,np.rint(counts))
            if c['name']=='untied':assert counts.tolist()==[[1,0,0,0]]
            if c['name']=='tie':assert counts.tolist()==[[1,0,1,1]]
            if c['name']=='cap':assert counts[0,:2].tolist()==[1,1]
            if c['name']=='early':assert counts[:,0].tolist()==[1,1,1,0,0] and counts[:,1].sum()==0
    rows=[];total=np.zeros(4,dtype=np.int64)
    w,h=plan['workload']['resolution']
    assert w*h<2**24
    for row in data['rows']:
        ci=row['camera'];columns=plan['counter_columns']
        with np.load(OUT/f'camera_{ci}_counts.npz') as z:
            counts=z['counts'];current=z['current_forward_contributing'];cached=z['historical_cached_contributing']
        assert counts.shape==(plan['workload']['triangles'],4) and counts.dtype==np.int32
        assert counts.min()>=0 and counts.max()<=w*h
        assert (counts[:,1:]<=counts[:,0,None]).all() and (counts[:,3]<=counts[:,2]).all()
        assert np.array_equal(counts[:,0]>0,current)
        path=ROOT/f'experiments/tgs_real_event_v1/baseline_{ci+2}.npz'
        assert sha(path)==row['historical_baseline_sha256']
        with np.load(path) as z:assert np.array_equal(z['max_blend']>0,cached)
        assert int((current!=cached).sum())==row['historical_cached_support_disagreements']
        totals=counts.sum(0,dtype=np.int64);total+=totals
        assert {key:int(v) for key,v in zip(columns,totals)}==row['totals']
        assert {key:int((counts[:,j]>0).sum()) for j,key in enumerate(columns)}==row['triangles_with_each_event']
        assert row['forward_rgb_aux_radii_equal']
        assert all(g['relative']<3e-4 and np.isfinite(g['relative']) for g in row['gradient_checks'])
        rows.append(dict(image=row['image'],contributing_pairs=int(totals[0]),saturated_pairs=int(totals[1]),
            legacy_multiple_pairs=int(totals[2]),exact_multiple_pairs=int(totals[3]),
            saturated_per_million_pairs=float(1e6*totals[1]/totals[0]),
            multiple_per_million_pairs=float(1e6*totals[2]/totals[0]),
            maximum_trace_control_gradient_relative_difference=max(g['relative'] for g in row['gradient_checks']),
            historical_support_matches=bool(np.array_equal(current,cached))))
    result=dict(evidence_audit_passed=True,rows=rows,
        combined=dict(contributing_pairs=int(total[0]),saturated_pairs=int(total[1]),legacy_multiple_pairs=int(total[2]),exact_multiple_pairs=int(total[3]),
                      saturated_per_million_pairs=float(1e6*total[1]/total[0]),multiple_per_million_pairs=float(1e6*total[2]/total[0])),
        interpretation='counts of traversed (triangle,pixel) pairs; not unique pixels, nonzero-gradient counts, dataset frequency or quality impact',
        timing_eligible=False,training=False)
    (OUT/'summary_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
