"""Independent complete batch accounting and MP80/140 fixed world probes."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,json,math,zipfile
import numpy as np
import mpmath as mp
from research.exact_outlier_world_reference import matrix,vector,dot,rotation
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research';BASE=R/'experiments/csr_decode_v1';OUT=R/'results/csr_decode_audit_v1.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def reference(row):
    A=matrix(row[20:29].reshape(3,3));Ai=A**-1;t=vector(row[29:32]);origin=-Ai*t;mean=vector(row[:3]);q=vector(row[32:35]);qn=mp.sqrt(dot(q,q));direction=Ai*q/qn
    rot=rotation([mp.mpf(float(v)) for v in row[6:10]]) if row[10] else matrix(row[11:20].reshape(3,3))
    cov=rot*mp.diag([mp.mpf(float(v))**2 for v in row[3:6]])*rot.T;precision=cov**-1;v=mean-origin
    a=dot(direction,precision*direction);p=dot(direction,precision*v)/a;res=v-p*direction;energy=dot(res,precision*res)
    exact_cam=[sum((Fraction(float(row[20+3*i+j]))*Fraction(float(row[j])) for j in range(3)),Fraction(float(row[29+i]))) for i in range(3)]
    qr=[Fraction(float(v)) for v in row[32:35]]
    if all(exact_cam[i]*qr[2]==exact_cam[2]*qr[i] for i in [0,1]):p=mp.mpf((exact_cam[2]/qr[2]).numerator)/(exact_cam[2]/qr[2]).denominator*qn;energy=mp.mpf(0)
    alpha=min(mp.mpf(float(np.float32(.99))),mp.mpf(float(row[35]))*mp.exp(-energy/2))
    return dict(p=p,rho=mp.sqrt(a),E=energy,alpha=alpha)
def main():
    assert not OUT.exists();m=read(BASE/'manifest.json');s=read(BASE/'summary.json');assert m['status']==s['status']=='complete' and s['all24_views']
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in m['artifacts_sha256'].items():assert sha(BASE/n)==h,n
    with zipfile.ZipFile(BASE/'sources.zip') as z:
        for n,h in m['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    for key,folder in [('capture_backend','query_generated'),('stream_backend','stream_generated')]:
        b=m[key];p=Path(b['binary_path']);p=ROOT/p.relative_to('/mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs') if str(p).startswith('/mnt/c/') else p
        assert sha(p)==b['binary_sha256']
        for n,h in b['generated_sha256'].items():assert sha(BASE/folder/n)==h,n
    # Actual query capture differs from the frozen renderer only in new channels.
    old=R/'experiments/full_topology_v1/instrumented_generated/src';new=BASE/'query_generated'
    f=(new/'cuda_rasterizer/render_forward.cu').read_text(encoding='utf-8').replace('\n            out_mdepth[6 * H * W + pix_id] = pixnf.x;\n            out_mdepth[7 * H * W + pix_id] = pixnf.y;','')
    assert f==(old/'cuda_rasterizer/render_forward.cu').read_text(encoding='utf-8')
    cpp=(new/'rasterize_points.cu').read_text(encoding='utf-8').replace('torch::full({8, H, W}, 0.0, float_opts)','torch::full({6, H, W}, 0.0, float_opts)')
    assert cpp==(old/'rasterize_points.cu').read_text(encoding='utf-8')
    assert (BASE/'stream_generated/shared_cache.cuh').read_bytes()==(R/'experiments/shared_cache_v1/generated/shared_cache.cuh').read_bytes()
    total=0;probe_count=0;fieldchecks=0;failures=[];rows=[];qfail=[]
    for r in s['rows']:
        name=f'scan{r["scan"]}_{r["camera"]}';cursor=0
        for batch in r['batches']:
            assert batch['start']==cursor and 0<batch['count']<=65536 and len(batch['output_sha256'])==64 and batch['event_ms']>0
            cursor+=batch['count']
        assert cursor==r['records']==r['processed_records']
        for mode in [0,1]:assert sum(r['status_counts'][mode].values())==cursor
        assert r['status0_nonfinite']==[0,0] and r['status_counts'][1]=={'0':cursor}
        assert all(r['original_image_bitwise'].values()) and r['probe_cached_direct_bitwise']
        qfail += [dict(scan=r['scan'],camera=r['camera'],case=q['case']) for q in r['old_anomaly_queries'] if not q['query_bits_match']]
        with np.load(BASE/(name+'_query_amp.npz')) as d:q=d['query_xy'];amp=d['amplitude'];counters=d['counters'];assert q.shape==(r['height']*r['width'],2) and amp.shape==(r['primitives'],) and np.isfinite(q).all() and np.isfinite(amp).all()
        with np.load(BASE/(name+'_probes.npz')) as d:
            source=d['source'];packet=d['packets'];direct=d['direct_packets'];indices=d['record_indices'];pixels=d['pixels'];ids=d['primitive_ids']
            assert np.array_equal(packet.view(np.uint64),direct.view(np.uint64)) and len(source)==r['probe_count']
            assert np.array_equal(source[:,32:34],q[pixels].astype(np.float64)) and np.all(source[:,34]==1) and np.array_equal(source[:,35],amp[ids].astype(np.float64))
            assert set(np.linspace(0,cursor-1,48,dtype=np.int64).tolist())<=set(indices.tolist())
            for i,srow in enumerate(source):
                mp.mp.dps=80;v80=reference(srow);mp.mp.dps=140;v=reference(srow)
                for key in ['p','rho','E','alpha']:
                    assert abs(v[key]-v80[key])<=mp.mpf('1e-65')*max(1,abs(v[key]))
                for mode in [0,1]:
                    p=packet[i,mode]
                    if p[0]!=0:continue
                    for key,a,b in [('p',1,2),('rho',3,4),('E',5,6),('alpha',24,25)]:
                        value=v[key];slack=mp.mpf('1e-120')*max(1,abs(value));fieldchecks+=1
                        if not mp.mpf(float(p[a]))-slack<=value<=mp.mpf(float(p[b]))+slack:failures.append(dict(scan=r['scan'],camera=r['camera'],probe=i,mode=mode,field=key,value=mp.nstr(value,70)))
                probe_count+=1
        total+=cursor;rows.append(dict(scan=r['scan'],camera=r['camera'],records=cursor,probes=len(source)))
        print(f'{len(rows)}/24 MP world probes done; total records accounting={total}',flush=True)
    assert total==870178922 and not failures
    result=dict(status='complete',checked_utc=datetime.now(timezone.utc).isoformat(),source_checks=len(m['source_sha256']),artifact_checks=len(m['artifacts_sha256']),
        views=24,records_accounted=total,complete_stream_hashes_and_states_accounted=True,all_high_valid=True,actual_query_source_isolation_pass=True,
        world_probe_count=probe_count,used_coefficient_checks=fieldchecks,independent_precisions=[80,140],probe_failures=failures,old_anomaly_query_mismatches=qfail,gate_pass=True,rows=rows,
        scope='All record accounting/status/hash manifests; full double streams not retained to recompute hashes offline. Predeclared48/view plus all9 anomaly-ray associations independently checked, not MP verification of all870M. Any copied-query mismatch reported separately; no root/backward/GT claim.')
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','scope']},indent=2))
if __name__=='__main__':main()
