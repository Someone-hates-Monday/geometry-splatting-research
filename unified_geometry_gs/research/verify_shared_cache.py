"""Independent saved-array/source-split audit of shared-cache results."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research';BASE=R/'experiments/shared_cache_v1';OLD=R/'experiments/reciprocal_root_v2';OUT=R/'results/shared_cache_audit_v1.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def same(a,b):return bool(a.shape==b.shape and a.dtype==b.dtype==np.dtype('float64') and np.array_equal(a.view(np.uint64),b.view(np.uint64)))
def main():
    assert not OUT.exists();m=read(BASE/'manifest.json');s=read(BASE/'summary.json');f=read(BASE/'fixed_summary.json');assert m['status']==s['status']=='complete'
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in m['artifacts_sha256'].items():assert sha(BASE/n)==h,n
    for n,h in m['generated_sha256'].items():assert sha(BASE/'generated'/n)==h,n
    with zipfile.ZipFile(BASE/'sources.zip') as z:
        assert set(z.namelist())==set(m['source_sha256'])
        for n,h in m['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    original=(OLD/'variant_generated/guarded_preprocess.cuh').read_text(encoding='utf-8');generated=(BASE/'generated/shared_cache.cuh').read_text(encoding='utf-8')
    start=original.index('__device__ __forceinline__ void decode(');cut=original.index(' I rawq[3]=',start);end=original.index('\n__global__ void preprocess(',cut)
    prefix=original[start:cut].replace('void decode(','void budget(')
    assert prefix in generated
    suffix=original[cut:end];assert generated.count(suffix)==1
    for name in ['guarded_interval.cuh','guarded_preprocess.cuh','guarded_root.cuh']:
        assert (BASE/'generated'/name).read_bytes()==(OLD/'variant_generated'/name).read_bytes()
    with np.load(BASE/'fixed_outputs.npz') as d,np.load(OLD/'outputs.npz') as p:
        cache=d['cache'];original_packets=d['original_packets'];decoded=d['decoded_packets'];roots=d['roots']
        assert cache.shape==(7322,28) and np.isfinite(cache).all() and np.all(cache[:,:2]==0)
        assert same(original_packets,p['variant_packets']) and same(decoded,original_packets) and same(roots,p['variant_roots'])
        # Check actual cached fields match both ray diagnostic paths exactly.
        for mode in [0,1]:
            packet=decoded[:,mode]
            assert np.array_equal(cache[:,2],packet[:,23]) and np.array_equal(cache[:,3+mode],packet[:,11]) and np.array_equal(cache[:,5],packet[:,12])
            assert np.array_equal(cache[:,6:9],packet[:,20:23]) and np.array_equal(cache[:,15+6*mode:21+6*mode],packet[:,14:20])
        assert np.all(cache[:,9:15:2]<=cache[:,10:15:2]) and np.all(cache[:,27]==0)
    assert f['all_bitwise_pass'] and all(f['checks'].values())
    pairs=f['pairs'];assert len(pairs)==5
    for i,p in enumerate(pairs):
        assert p['pair']==i and p['order']==(['original','cached'] if i%2==0 else ['cached','original'])
        ratio=p['results']['cached']['event_ms']/p['results']['original']['event_ms'];assert math.isclose(ratio,p['ratio'],rel_tol=1e-15)
    ratio=math.exp(sum(math.log(p['ratio']) for p in pairs)/5);assert math.isclose(ratio,f['paired_ratio_geometric_mean'],rel_tol=1e-15)
    assert f['local_reuse_gate_pass']==(ratio<=.95)
    expected=[]
    for scan in [37,55,63]:
        names=list(dict.fromkeys(r['camera'] for r in read(R/f'results/median_cpu_scan{scan}.json')['rows']));expected.extend((scan,n) for n in names)
    assert expected==[(r['scan'],r['camera']) for r in s['rows']] and len(s['rows'])==24
    for row in s['rows']:
        P=row['primitives'];assert row['all_cache_constructible'] and row['nonfinite_cache_rows']==0
        assert row['cache_shape']==[P,28] and row['memory']['cache_row_bytes']==224 and row['memory']['cache_bytes']==224*P
        assert row['statuses']=={'0':{'0':P},'1':{'0':P}}
        assert len(row['event_ms'])==len(row['wall_ms'])==5 and min(row['event_ms'])>0 and row['median_event_ms']==float(np.median(row['event_ms']))
    result=dict(status='complete',checked_utc=datetime.now(timezone.utc).isoformat(),archive_checks=dict(sources=len(m['source_sha256']),artifacts=len(m['artifacts_sha256']),generated=len(m['generated_sha256'])),
        frozen_core_and_query_body_unchanged=True,all_saved_budget_and_root_bits_match=True,cached_fields_match_original=True,source_rows=7322,repeat_queries=58576,
        ratio_geometric_mean=ratio,local_reuse_gate_pass=f['local_reuse_gate_pass'],real_views=24,all_real_build_metadata_pass=True,cache_row_bytes=224,
        source_real_arrays_offline_limitation='Real cache content hashes/statistics are recorded at runtime; real arrays were not retained for a second offline/full MP audit.',
        scope='Independent archive/code split/saved local arrays/timing/real build metadata audit. No complete CSR ray decode, certified full images, source backward, GT or training evidence.')
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
