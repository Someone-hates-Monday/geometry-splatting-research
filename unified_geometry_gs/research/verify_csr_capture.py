"""Independent all-record CPU CSR order/provenance and complete archive audit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research';BASE=R/'experiments/csr_capture_v1';OLD=R/'experiments/full_topology_v1';OUT=R/'results/csr_capture_audit_v1.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert not OUT.exists();m=read(BASE/'manifest.json');s=read(BASE/'summary.json');assert m['status']==s['status']=='complete' and s['views']==24
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in m['artifacts_sha256'].items():assert sha(BASE/n)==h,n
    with zipfile.ZipFile(BASE/'sources.zip') as z:
        assert set(z.namelist())==set(m['source_sha256'])
        for n,h in m['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    for key in ['control_backend','csr_backend']:
        b=m[key];assert sha(ROOT/b['binary_relative_path'])==b['binary_sha256']
        tree=OLD/'instrumented_generated/src' if key=='control_backend' else BASE/'csr_generated/src'
        for n,h in b['generated_sha256'].items():assert sha(tree/n)==h,n
        for n,h in b['vendor_headers_sha256'].items():assert sha(ROOT/'submodules/diff-gaussian-rasterization'/n)==h,n
    prior=read(OLD/'summary.json');assert [(r['scan'],r['camera']) for r in s['rows']]==[(r['scan'],r['camera']) for r in prior['rows']]
    rows=[];total=0;pixels=0
    for row in s['rows']:
        name=f'scan{row["scan"]}_{row["camera"]}';H,W,P=row['height'],row['width'],row['primitives'];L=row['records']
        with np.load(BASE/(name+'.npz')) as d:
            offsets=d['offsets'];records=d['records'];counts=d['expected_counts'];write=d['write_status'];check=d['checker_status'];control=d['control_depth_counters'];capture=d['capture_depth_counters'];contrib=d['n_contrib'];ranges=d['tile_ranges'];points=d['point_list']
            assert offsets.dtype==np.int64 and records.dtype==np.int32 and counts.dtype==np.int32
            assert offsets.shape==(H*W+1,) and records.shape==(L,2) and counts.shape==(H,W)
            assert write.shape==check.shape==contrib.shape==(H,W) and ranges.shape==(((W+15)//16)*((H+15)//16),2) and points.ndim==1
            assert offsets[0]==0 and offsets[-1]==L and np.array_equal(np.diff(offsets),counts.ravel()) and (counts>=0).all()
            assert not np.any(write) and not np.any(check)
            assert control.shape==capture.shape==(6,H,W) and np.array_equal(control.view(np.uint32),capture.view(np.uint32))
            assert np.array_equal(counts,control[3])
            with np.load(OLD/(name+'.npz')) as old:
                assert np.array_equal(counts,old['depth_accepted']) and np.array_equal(control[:2].view(np.uint32),old['control_depth_slope'].view(np.uint32))
            assert (records[:,0]>=0).all() and (records[:,0]<P).all() and (records[:,1]>=0).all()
            assert (points>=0).all() and (points<P).all() and (ranges[:,0]>=0).all() and (ranges[:,1]>=ranges[:,0]).all() and (ranges[:,1]<=points.size).all()
            # Every non-increasing adjacent position must be a pixel boundary.
            bad_order=np.flatnonzero(records[1:,1]<=records[:-1,1])+1
            indices=np.searchsorted(offsets,bad_order,side='left')
            assert np.array_equal(offsets[indices],bad_order)
            checked_records=0;flat=counts.ravel();cn=contrib.ravel()
            # Check all provenance in bounded memory, not sampled pixels.
            for begin in range(0,H*W,4096):
                end=min(begin+4096,H*W);lo=int(offsets[begin]);hi=int(offsets[end])
                if hi==lo:continue
                pixel=np.repeat(np.arange(begin,end,dtype=np.int32),flat[begin:end])
                position=records[lo:hi,1];ids=records[lo:hi,0]
                tile=((pixel//W)//16)*((W+15)//16)+(pixel%W)//16
                base=ranges[tile,0];limit=ranges[tile,1]
                assert (position<cn[pixel]).all() and (position<limit-base).all()
                assert np.array_equal(ids,points[base+position]);checked_records+=hi-lo
            assert checked_records==L
        assert row['numerical_gate_pass'] and all(row['original_output_bitwise_checks'].values())
        pairs=row['timing_pairs'];assert len(pairs)==5
        for i,p in enumerate(pairs):
            assert p['pair']==i and p['order']==(['count','capture'] if i%2==0 else ['capture','count'])
            ratio=p['results']['capture']['event_ms']/p['results']['count']['event_ms'];assert math.isclose(ratio,p['event_ratio'],rel_tol=1e-15)
        ratio=math.exp(sum(math.log(p['event_ratio']) for p in pairs)/5);assert math.isclose(ratio,row['paired_capture_count_ratio_geometric_mean'],rel_tol=1e-15)
        rows.append(dict(scan=row['scan'],camera=row['camera'],pixels=H*W,records=L,all_positions_and_source_IDs_checked=True));total+=L;pixels+=H*W
        progress=dict(utc=datetime.now(timezone.utc).isoformat(),completed_views=len(rows),checked_records=total,last=name)
        (R/'results/csr_capture_audit_progress_v1.json').write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8')
        print(f'{len(rows)}/24 independent CPU checked {total} records',flush=True)
        del offsets,records,counts,control,capture,ranges,points,contrib,write,check,bad_order,indices
    guards=[]
    for which,expected,change,holes in [('under',3,-1,0),('over',2,1,1)]:
        g=read(BASE/f'guard_{which}.json');assert g['gate_pass'] and g['change']==change and g['target_actual_status']==expected and g['unfilled_slots']==holes
        with np.load(BASE/f'guard_{which}.npz') as d:
            st=d['write_status'].ravel();assert st[g['target_flat_pixel']]==expected and np.count_nonzero(st)==1
            assert (np.diff(d['offsets'])>=0).all();span=d['target_span_records'];assert int((span[:,0]==-1).sum())==holes
        guards.append(dict(case=which,status=expected,unfilled_slots=holes,pass_gate=True))
    assert total==s['records']==870178922 and pixels==s['pixels']==10834488
    result=dict(status='complete',checked_utc=datetime.now(timezone.utc).isoformat(),views=24,pixels=pixels,checked_records=total,
        archive_checks=dict(sources=len(m['source_sha256']),artifacts=len(m['artifacts_sha256']),csr_generated=len(m['csr_backend']['generated_sha256'])),
        all_record_ID_range_order_prefix_and_source_matches=True,all_saved_depth_counters_bitwise=True,all_status_zero=True,guard_probes=guards,gate_pass=True,rows=rows,
        scope='Every saved CSR record checked independently on CPU; original runtime RGB/alpha/normal/radii flags/source verified but those arrays not retained for a second offline bit audit. No full certified root/backward/GT claim.')
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','scope']},indent=2))
if __name__=='__main__':main()
