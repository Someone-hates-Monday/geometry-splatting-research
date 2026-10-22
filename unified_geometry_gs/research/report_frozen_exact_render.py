"""Replay complete images, every gate, source archives and paired timing arithmetic."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def quantiles(x):
    return {k:float(v) for k,v in zip(('min','p50','p90','p99','max'),np.quantile(x,(0,.5,.9,.99,1)))} if len(x) else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    m=read(args.run/'manifest.json');s=read(args.run/'summary.json');assert m['status']=='complete' and s['views']==24
    with zipfile.ZipFile(args.run/'sources.zip') as z:
        assert set(z.namelist())==set(m['source_sha256'])
        for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest and hashlib.sha256(z.read(name)).hexdigest()==digest
    for name,digest in m['artifacts_sha256'].items():assert sha(args.run/name)==digest
    # Linux generated/binary paths are checked separately on their host, not silently remapped here.
    failures=[];ratios=[];wallratios=[];pixels=0;errorpixels=0;all_finite=True;preserved=True;row_gates=[]
    counts={v:[] for v in ('affine','exact_packet','exact_factor')}
    for row in s['rows']:
        with np.load(args.run/f"scan{row['scan']}_{row['camera']}.npz",allow_pickle=False) as d:
            a=d['affine_depth'].squeeze();b=d['exact_packet_depth'].squeeze();c=d['exact_factor_depth'].squeeze()
            assert a.shape==(row['height'],row['width']);pixels+=a.size
            common=(b>0)&(c>0);error=abs(b[common]-c[common]);q=quantiles(error);fraction=float(np.mean((b>0)!=(c>0)))
            assert q==row['packet_factor_depth_abs_error'] and fraction==row['packet_factor_validity_mismatch_fraction']
            finite=all(np.all(np.isfinite(d[k])) for k in d.files);all_finite &= finite
            for v,depth in zip(('affine','exact_packet','exact_factor'),(a,b,c)):
                stat=row['variants'][v];counts[v].append(int(sum((depth>0).ravel())));assert counts[v][-1]==stat['valid_pixels']
                va=(depth>0)&(a>0);assert quantiles(abs(depth[va]-a[va]))==stat['depth_abs_difference_vs_affine']
                assert int(sum(((depth>0)!=(a>0)).ravel()))==stat['validity_mismatches_vs_affine']
                if v!='affine':
                    for key,field in (('rgb','render'),('alpha','mask'),('normal','normal')):
                        actual=float(np.max(abs(d[v+'_'+field+'_delta'])))
                        assert actual==stat[key+'_max_abs_difference_vs_affine'];preserved &= actual==0
                assert float(np.median([r['gpu_ms_per_render'] for r in stat['timing_rounds']]))==stat['gpu_ms_median']
                assert float(np.median([r['wall_ms_per_render'] for r in stat['timing_rounds']]))==stat['wall_ms_median']
            gate=(finite and all(x['negative_depth_pixels']==x['negative_alpha_pixels']==0 for x in row['variants'].values())
                and all(x[k]<=1e-5 for x in row['variants'].values() for k in ('rgb_max_abs_difference_vs_affine','alpha_max_abs_difference_vs_affine','normal_max_abs_difference_vs_affine'))
                and fraction<=1e-4 and q is not None and q['p99']<=1e-4 and q['max']<=1e-3)
            assert bool(gate)==row['numerical_gate_pass'];row_gates.append(bool(gate))
            e=np.where(common,abs(b-c),0);n=int(sum((e>1e-3).ravel()));errorpixels+=n
            if not gate:
                y,x=np.unravel_index(np.argmax(e),e.shape)
                failures.append({'scan':row['scan'],'camera':row['camera'],'pixel':[int(x),int(y)],'pixels_gt1e-3':n,
                    'max_abs_depth_error':float(e[y,x]),'depths':{v:float(d[v+'_depth'].squeeze()[y,x]) for v in counts},
                    'slopes':{v:float(d[v+'_slope'].squeeze()[y,x]) for v in counts},
                    'p99_abs_depth_error':q['p99'],'mask_mismatch_fraction':fraction})
            ratios.append(row['variants']['exact_packet']['gpu_ms_median']/row['variants']['exact_factor']['gpu_ms_median'])
            wallratios.append(row['variants']['exact_packet']['wall_ms_median']/row['variants']['exact_factor']['wall_ms_median'])
    gm=float(np.exp(np.mean(np.log(ratios))));eff=all(row_gates) and gm<=.90 and max(ratios)<=1.05
    assert gm==s['packet_factor_render_gpu_ratio_geomean'] and max(ratios)==s['packet_factor_render_gpu_ratio_max']
    assert all(row_gates)==s['numerical_gate_pass'] and eff==s['efficiency_gate_pass']
    result={'status':'complete','checked_utc':datetime.now(timezone.utc).isoformat(),'verified_sources_and_zip':len(m['source_sha256']),
      'verified_artifacts':len(m['artifacts_sha256']),'views':24,'total_image_pixels':pixels,
      'numerical_gate_pass':all(row_gates),'failed_views':len(failures),'pixels_gt1e-3':errorpixels,
      'all_saved_arrays_finite':bool(all_finite),'rgb_alpha_normal_bitwise_preserved_all_views':bool(preserved),
      'packet_factor_gpu_ratio_geomean':gm,'packet_factor_gpu_ratio_max':max(ratios),
      'packet_factor_wall_ratio_geomean':float(np.exp(np.mean(np.log(wallratios)))),
      'efficiency_gate_pass':eff,'failures':failures,
      'interpretation':'Whole-image negative gate preserved; no GT/training expansion. Sparse outliers and low saved secant slopes warrant reconstruction, not an unproven causal attribution.',
      'source_sha256':sha(Path(__file__)),
      'limits':['Stored complete-image/gate/timing arithmetic replay, not an independent rerender or independent pixel root oracle.',
        'Native replay does not verify host-specific generated source/binary paths; a separate WSL hash check is required.',
        'Exact packet and factor both use FP64 scalar evaluation with FP32 input metadata and the legacy solver.',
        'No complete normal/NCC/backward, GT quality or novel-scene result.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('failures','limits')},indent=2),flush=True)
if __name__=='__main__':main()
