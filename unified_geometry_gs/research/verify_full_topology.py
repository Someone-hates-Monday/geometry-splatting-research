"""CPU audit of all24 saved pixel topology fields and compiled source isolation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
BASE=R/'experiments/full_topology_v1';OUT=R/'results/full_topology_audit_v1.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
def stats(a):
    assert a.size
    hist=np.bincount(a)
    return dict(pixels=int(a.size),min=int(a.min()),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),max=int(a.max()),
        total_accepted=int(a.sum(dtype=np.int64)),histogram={str(i):int(n) for i,n in enumerate(hist) if n},
        capacity_exceed={str(k):dict(pixels=int((a>k).sum()),fraction=float((a>k).mean())) for k in [192,256,512]})
def main():
    assert not OUT.exists();m=read(BASE/'manifest.json');s=read(BASE/'summary.json')
    assert m['status']==s['status']=='complete' and s['phase']=='full24' and s['views']==24
    counts={}
    for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest,name
    counts['sources']=len(m['source_sha256'])
    with zipfile.ZipFile(BASE/'sources.zip') as z:
        assert set(z.namelist())==set(m['source_sha256'])
        for name,digest in m['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    for name,digest in m['artifacts_sha256'].items():assert sha(BASE/name)==digest,name
    counts['artifacts']=len(m['artifacts_sha256'])
    for arm,meta in m['backends'].items():
        assert sha(ROOT/meta['binary_relative_path'])==meta['binary_sha256']
        for name,digest in meta['generated_sha256'].items():assert sha(BASE/(arm+'_generated/src')/name)==digest,name
        for name,digest in meta['vendor_headers_sha256'].items():assert sha(ROOT/'submodules/diff-gaussian-rasterization'/name)==digest,name
        counts[arm+'_generated']=len(meta['generated_sha256'])
    # Check actual built code changed only counters and extra output allocation.
    control=BASE/'control_generated/src';variant=BASE/'instrumented_generated/src'
    forward=(variant/'cuda_rasterizer/render_forward.cu').read_text(encoding='utf-8')
    forward=once(forward,'uint32_t last_contributor          = 0;\n    uint32_t topology_rgb_accepted=0, topology_depth_accepted=0, topology_nonball=0;',
                 'uint32_t last_contributor          = 0;')
    forward=once(forward,'topology_rgb_accepted++;\n            const float aT = alpha * T;','const float aT = alpha * T;')
    forward=once(forward,'const bool ball    = rsigma > 0;\n                    if constexpr(FIRST){topology_depth_accepted++;if(!ball)topology_nonball++;}',
                 'const bool ball    = rsigma > 0;')
    forward=once(forward,'out_mdepth[H * W + pix_id]     = tslope;\n'
                 '            out_mdepth[2 * H * W + pix_id] = float(topology_rgb_accepted);\n'
                 '            out_mdepth[3 * H * W + pix_id] = float(topology_depth_accepted);\n'
                 '            out_mdepth[4 * H * W + pix_id] = float(topology_nonball);\n'
                 '            out_mdepth[5 * H * W + pix_id] = T<=MIN_TRANSMITTANCE?1.f:0.f;',
                 'out_mdepth[H * W + pix_id]     = tslope;')
    assert forward==(control/'cuda_rasterizer/render_forward.cu').read_text(encoding='utf-8')
    cpp=once((variant/'rasterize_points.cu').read_text(encoding='utf-8'),'torch::full({6, H, W}, 0.0, float_opts)','torch::full({2, H, W}, 0.0, float_opts)')
    assert cpp==(control/'rasterize_points.cu').read_text(encoding='utf-8')
    for name in m['backends']['control']['generated_sha256']:
        if name not in ['cuda_rasterizer/render_forward.cu','rasterize_points.cu']:assert (control/name).read_bytes()==(variant/name).read_bytes(),name
    identities=[];initial_arrays=[];valid_arrays=[];rows=[]
    for scan in [37,55,63]:
        ref=read(R/f'results/median_cpu_scan{scan}.json');names=list(dict.fromkeys(r['camera'] for r in ref['rows']))
        identities.extend((scan,n) for n in names)
        model=m['model_inputs'][str(scan)];assert sha(ROOT/model['ply_relative_path'])==model['ply_sha256']
    assert identities==[(r['scan'],r['camera']) for r in s['rows']]
    pixels=0;mismatches=0
    for row in s['rows']:
        scan,name,H,W=row['scan'],row['camera'],row['height'],row['width'];path=BASE/f'scan{scan}_{name}.npz'
        with np.load(path) as d:
            dc=d['control_depth_slope'];dv=d['instrumented_depth_slope'];rgb=d['rgb_accepted'];depth=d['depth_accepted'];nonball=d['nonball_accepted'];initial=d['initial_eligible'];valid=d['final_valid'];contrib=d['n_contrib'];mismatch=d['rgb_depth_count_mismatch']
            assert dc.shape==dv.shape==(2,H,W) and np.isfinite(dc).all() and np.isfinite(dv).all()
            assert np.array_equal(dc.view(np.uint32),dv.view(np.uint32))
            assert all(a.shape==(H,W) for a in [rgb,depth,nonball,initial,valid,contrib,mismatch])
            assert rgb.dtype==depth.dtype==nonball.dtype==contrib.dtype==np.int32
            assert initial.dtype==valid.dtype==mismatch.dtype==np.bool_
            assert (rgb>=0).all() and (depth>=0).all() and (nonball>=0).all() and (rgb<=row['primitives']).all() and (depth<=row['primitives']).all()
            assert (rgb<=contrib).all() and (nonball<=depth).all() and (depth[~initial]==0).all()
            assert np.array_equal(valid,dc[0]>0) and np.array_equal(mismatch,initial&(rgb!=depth))
            assert stats(depth[initial])==row['initial_depth_counts'] and stats(depth[valid])==row['valid_depth_counts']
            assert stats(rgb.ravel())==row['all_rgb_counts']
            assert int((initial&(nonball>0)).sum())==row['initial_nonball_pixels']
            assert int((valid&(nonball>0)).sum())==row['valid_nonball_pixels']
            assert int(initial.sum())==row['initial_eligible_pixels'] and int(valid.sum())==row['final_valid_pixels']
            assert int(mismatch.sum())==row['rgb_depth_count_mismatches'];mismatches+=int(mismatch.sum());pixels+=H*W
            initial_arrays.append(depth[initial].copy());valid_arrays.append(depth[valid].copy())
        assert all(row['original_output_bitwise_checks'].values()) and row['numerical_gate_pass']
        pairs=row['timing_pairs'];assert len(pairs)==5
        for i,p in enumerate(pairs):
            assert p['pair']==i and p['order']==(['control','instrumented'] if i%2==0 else ['instrumented','control'])
            ratio=p['results']['instrumented']['event_ms']/p['results']['control']['event_ms']
            assert math.isclose(ratio,p['event_ratio'],rel_tol=1e-15)
        gm=math.exp(sum(math.log(p['event_ratio']) for p in pairs)/5)
        assert math.isclose(gm,row['paired_event_ratio_geometric_mean'],rel_tol=1e-15)
        rows.append(dict(scan=scan,camera=name,initial_pixels=row['initial_eligible_pixels'],valid_pixels=row['final_valid_pixels'],
            initial_max=row['initial_depth_counts']['max'],valid_max=row['valid_depth_counts']['max'],
            valid_exceed192=row['valid_depth_counts']['capacity_exceed']['192'],
            median_control_ms=row['event_median_ms']['control'],median_instrumented_ms=row['event_median_ms']['instrumented']))
    all_initial=np.concatenate(initial_arrays);all_valid=np.concatenate(valid_arrays)
    assert stats(all_initial)==s['initial_depth_counts'] and stats(all_valid)==s['valid_depth_counts']
    assert pixels==s['pixels']==10834488 and mismatches==s['rgb_depth_count_mismatches']==0
    record=dict(status='complete',checked_utc=datetime.now(timezone.utc).isoformat(),archive_checks=counts,all24_views=True,pixels=pixels,
        saved_full_depth_slope_bitwise=True,runtime_rgb_alpha_normal_radii_bitwise_flags_all_true=True,
        source_isolation_pass=True,all_counts_and_histograms_recomputed=True,rgb_depth_count_mismatches=mismatches,
        initial_depth_counts=s['initial_depth_counts'],valid_depth_counts=s['valid_depth_counts'],
        capacity192_covers_all_initial=bool((all_initial<=192).all()),capacity512_covers_this_fixed_cohort=bool((all_initial<=512).all()),
        per_scan={str(scan):dict(views=8,initial_pixels=sum(r['initial_pixels'] for r in rows if r['scan']==scan),valid_pixels=sum(r['valid_pixels'] for r in rows if r['scan']==scan),
                    initial_max=max(r['initial_max'] for r in rows if r['scan']==scan),valid_exceed192=sum(r['valid_exceed192']['pixels'] for r in rows if r['scan']==scan)) for scan in [37,55,63]},
        rows=rows,scope='All saved topology/median arrays, source isolation and archives audited. RGB/alpha/normal/radii were fully compared in runtime producer; arrays not retained for a second offline bit check. Not certified full render, backward/GT or unseen prevalence.')
    OUT.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k not in ['rows','initial_depth_counts','valid_depth_counts','scope']},indent=2))
    print(json.dumps(dict(initial={k:v for k,v in s['initial_depth_counts'].items() if k!='histogram'},valid={k:v for k,v in s['valid_depth_counts'].items() if k!='histogram'}),indent=2))
if __name__=='__main__':main()
