"""Artifact and arithmetic replay of the fixed world/affine ray comparison."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def stats(x):return dict(zip(('min','median','max'),map(float,np.quantile(x,[0,.5,1]))))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    m=read(args.run/'manifest.json');assert m['status']=='complete'
    with zipfile.ZipFile(args.run/'sources.zip') as archive:
        assert set(archive.namelist())==set(m['source_sha256'])
        for name,digest in m['source_sha256'].items():
            assert sha(ROOT/name)==digest and hashlib.sha256(archive.read(name)).hexdigest()==digest
    for name,digest in m['artifacts_sha256'].items():assert sha(args.run/name)==digest
    rows=[]
    for i in range(24):
        r=read(args.run/f'ray_{i:02d}.json');prior=read(ROOT/f'research/experiments/directional_field_world_v1/point_{i:02d}.json')
        assert r['scan']==prior['scan'] and r['camera']==prior['camera'] and r['pixel']==prior['pixel']
        assert r['legacy_affine_median']==prior['saved_affine_median_depth'] and r['gaussians_evaluated']==prior['gaussians_evaluated']
        assert r['camera_median_finite'] and r['remaining_optical_depth']>np.log(2)
        delta=r['world_camera_median']-r['legacy_affine_median']
        assert delta==r['signed_depth_difference']
        assert abs(delta)/abs(r['legacy_affine_median'])==r['absolute_depth_difference_over_legacy_depth']
        at0=r['origin_logT_minus']
        for key in ('world_at_legacy_median','world_at_camera_median'):
            a=r[key]
            assert np.exp(a['logV'])==a['V']
            assert np.exp(a['logT_minus']-at0)==a['T_camera']
            assert a['logV']-a['logT_minus']==a['logV_minus_logT_minus']
            assert a['logV']-a['logT_minus']+at0==a['logV_minus_logT_camera']
        assert r['root_log_residual']<=1e-8 and abs(r['world_at_camera_median']['T_camera']-.5)<=1e-8
        assert r['point_attenuation_scaled_error']<=1e-7 and r['point_logV_check_error']<=1e-8*max(1.,abs(prior['log_product_vacancy']))
        rows.append(r)
    summary=read(args.run/'summary.json');assert summary['points']==summary['camera_median_finite']==24
    for scan,group in zip((37,55,63),summary['per_scan']):
        selected=[r for r in rows if r['scan']==scan];assert len(selected)==group['finite']==8
        assert group['signed_depth_difference']==stats([r['signed_depth_difference'] for r in selected])
        assert group['abs_difference_over_legacy_depth']==stats([r['absolute_depth_difference_over_legacy_depth'] for r in selected])
        assert group['old_point_world_camera_T']==stats([r['world_at_legacy_median']['T_camera'] for r in selected])
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json')
        ply=ROOT/f'output_compare/ggs_pipeline/scan{scan}/point_cloud/iteration_30000/point_cloud.ply'
        assert sha(ply)==next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
    outlier=max(rows,key=lambda x:x['absolute_depth_difference_over_legacy_depth'])
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','source_sha256':sha(Path(__file__)),
      'verified_sources_and_zip':len(m['source_sha256']),'verified_artifacts':len(m['artifacts_sha256']),
      'verified_PLYs':3,'verified_rows_and_arithmetic':24,'all_original_points_retained':True,
      'overall_abs_difference_over_legacy_depth':stats([r['absolute_depth_difference_over_legacy_depth'] for r in rows]),
      'largest_relative_difference':{'scan':outlier['scan'],'camera':outlier['camera'],'pixel':outlier['pixel'],
        'legacy_depth':outlier['legacy_affine_median'],'world_depth':outlier['world_camera_median'],
        'relative_difference':outlier['absolute_depth_difference_over_legacy_depth']},
      'limits':['Archive and arithmetic replay does not independently re-evaluate full-world quantile roots.',
        'Pointwise logV and slope checks use a different full-world evaluation graph; root residual uses the same ray function.',
        'No interval root bound, GT comparison or causal decomposition.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
