"""Replay raw-model witnesses, rational endpoints and independent scalar derivatives."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import mpmath as mp
import numpy as np
from plyfile import PlyData
from research.directional_field import interval_pair_certificate,interval_gaussian_gradient

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def exact(x):return Fraction(int(x['numerator']),int(x['denominator']))
def stats(x):return dict(zip(('min','median','max'),map(float,np.quantile(x,[0,.5,1]))))

def scalar_logv(raw,point):
    # Scalar expression differentiated by mpmath, not the explicit b_i formula.
    delta=[point[i]-mp.mpf(raw['mean'][i]) for i in range(3)]
    q=[mp.mpf(x) for x in raw['raw_quaternion']]
    qnorm=mp.sqrt(sum(x*x for x in q));w,x,y,z=[x/qnorm for x in q]
    rotation=((1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y),
              (2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x),
              (2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y))
    variance=[mp.exp(2*mp.mpf(x)) for x in raw['log_scale']]
    smooth=[x+mp.mpf(raw['filter_3D'])**2 for x in variance]
    volume_ratio=mp.fprod(a/b for a,b in zip(variance,smooth))
    alpha=mp.sqrt(volume_ratio)/(1+mp.exp(-mp.mpf(raw['opacity_logit'])))
    coordinates=[sum(rotation[i][j]*delta[i] for i in range(3)) for j in range(3)]
    density=alpha*mp.exp(-sum(x*x/v for x,v in zip(coordinates,smooth))/2)
    return mp.log1p(-density)/2

def raw_matches(raw,row):
    assert raw['mean']==[float(row[k]) for k in ('x','y','z')]
    assert raw['log_scale']==[float(row[f'scale_{i}']) for i in range(3)]
    assert raw['raw_quaternion']==[float(row[f'rot_{i}']) for i in range(4)]
    assert raw['opacity_logit']==float(row['opacity'])
    assert raw['filter_3D']==float(row['filter_3D'])

def replay(run):
    manifest=read(run/'manifest.json');assert manifest['status']=='complete'
    with zipfile.ZipFile(run/'sources.zip') as archive:
        assert set(archive.namelist())==set(manifest['source_sha256'])
        for name,digest in manifest['source_sha256'].items():
            assert sha(ROOT/name)==digest
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    for name,digest in manifest['artifacts_sha256'].items():assert sha(run/name)==digest
    theoretical=read(ROOT/'research/results/directional_field_theory_v1.json')
    for name,digest in theoretical['source_sha256'].items():assert sha(ROOT/'research'/name)==digest
    assert all(theoretical['symbolic_checks']['world_gradient'])
    assert len(theoretical['numeric_cases'])==64
    records=[];derivatives=[];models=[]
    for scan in (37,55,63):
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json')
        modelpath=ROOT/f'output_compare/ggs_pipeline/scan{scan}'
        ply=modelpath/'point_cloud/iteration_30000/point_cloud.ply'
        expected=next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
        assert sha(ply)==expected
        data=PlyData.read(ply)['vertex'].data
        cameras={c['img_name']:c for c in read(modelpath/'cameras.json')}
        selected=[];seen=set()
        for row_index,row in enumerate(old['rows']):
            if row['camera'] not in seen and row['finite_root']:
                seen.add(row['camera']);selected.append((row_index,row))
        assert len(selected)==8
        for row_index,row in selected:
            idx=len(records);record=read(run/f'point_{idx:02d}.json')
            assert record['scan']==scan and record['legacy_row']==row_index
            assert record['camera']==row['camera'] and record['pixel']==row['pixel']
            camera=cameras[row['camera']];width,height=round(camera['width']/2),round(camera['height']/2)
            fx=camera['fx']*width/camera['width'];fy=camera['fy']*height/camera['height']
            px,py=row['pixel'];d=np.array([(px-(width-1)/2)/fx,(py-(height-1)/2)/fy,1.])
            d/=np.linalg.norm(d);worlddir=np.asarray(camera['rotation'])@d
            point=np.asarray(camera['position'])+worlddir*row['median_ray_depth']
            assert np.array_equal(point,record['world_point'])
            assert np.array_equal(worlddir,record['query_world_direction'])
            assert record['gaussians_evaluated']==len(data) and record['invalid_gradient_count']==0
            for raw in record['witness_pair']:raw_matches(raw,data[raw['gaussian_index']])
            stored=record['interval_pair_certificate'];assert stored['positive_certified']
            repeat=interval_pair_certificate(record['witness_pair'],point,dps=80)
            for key in ('cross_squared_lower_exact','trace_upper_exact','certificate_lower_exact'):
                assert repeat[key]==stored[key]
            lower=exact(stored['certificate_lower_exact']);det=exact(stored['cross_squared_lower_exact'])
            trace=exact(stored['trace_upper_exact'])
            assert lower>0 and det>0 and trace>0 and lower*lower*trace<=det
            assert Fraction.from_float(stored['conservative_double_lower_bound'])<lower
            # Preserve actual interval endpoints at two precisions, not their rounded text.
            gradients={}
            for precision in (80,100):
                mp.iv.dps=precision
                gradients[precision]=[interval_gaussian_gradient(raw,point) for raw in record['witness_pair']]
            mp.mp.dps=120
            for j,raw in enumerate(record['witness_pair']):
                exactpoint=[mp.mpf(float(x)) for x in point]
                errors=[]
                for axis in range(3):
                    def f(value):
                        position=exactpoint.copy();position[axis]=value
                        return scalar_logv(raw,position)
                    derivative=mp.diff(f,exactpoint[axis])
                    for precision in (80,100):
                        ivvalue=gradients[precision][j][axis]
                        assert mp.mpf(ivvalue.a)<=derivative<=mp.mpf(ivvalue.b)
                    oldiv,newiv=gradients[80][j][axis],gradients[100][j][axis]
                    assert mp.mpf(oldiv.a)<=mp.mpf(newiv.a)<=mp.mpf(newiv.b)<=mp.mpf(oldiv.b)
                    nominal=mp.mpf(record['witness_nominal_gradients'][j][axis])
                    errors.append(float(abs(nominal-derivative)/max(mp.mpf('1e-300'),abs(derivative))))
                derivatives.append({'point':idx,'gaussian_index':raw['gaussian_index'],
                                    'three_components_enclosed_80_and_100':True,'double_relative_errors':errors})
            S=np.asarray(record['gradient_gram']);eigen=np.linalg.eigvalsh(S)
            assert np.allclose(eigen,record['gram_eigenvalues_ascending'],rtol=1e-12,atol=1e-20)
            projected=abs(np.asarray(record['directions'])@np.asarray(record['sum_gradient']))
            assert np.array_equal(projected,record['product_field_attenuation'])
            gaps=np.asarray(record['summed_attenuation'])-projected
            assert np.array_equal(gaps,record['directional_gap'])
            assert min(gaps)>=-1e-12*max(record['summed_attenuation'])
            records.append(record)
        models.append({'scan':scan,'sha256':expected,'gaussian_count':len(data)})
    summary=read(run/'summary.json');assert len(records)==summary['points']==24
    for key,values in [('nominal_spectral_lower_bounds',[r['nominal_spectral_error_lower_bound'] for r in records]),
                       ('interval_absolute_lower_bounds',[r['interval_pair_certificate']['conservative_double_lower_bound'] for r in records]),
                       ('nominal_normalized_bounds',[r['nominal_lower_bound_over_sum_norms'] for r in records])]:
        assert values==summary[key]
    scans=[]
    for scan in (37,55,63):
        rows=[r for r in records if r['scan']==scan]
        scans.append({'scan':scan,'queries':len(rows),
          'pair_absolute_lower_bound':stats([r['interval_pair_certificate']['conservative_double_lower_bound'] for r in rows]),
          'nominal_spectral_bound_over_sum_norms':stats([r['nominal_lower_bound_over_sum_norms'] for r in rows]),
          'nominal_camera_relative_gap':stats([r['relative_directional_gap'][5] for r in rows]),
          'underflow_zero_G_count':stats([r['nominal_zero_G_count'] for r in rows])})
    return {'verified':{'current_sources_and_zip':len(manifest['source_sha256']),
                        'artifacts':len(manifest['artifacts_sha256']),'models_and_all_raw_witnesses':True,
                        'queries_reconstructed_exactly':len(records),'rational_inequalities':24,
                        'interval_endpoint_replays':24,'independent_scalar_gradient_vectors':len(derivatives),
                        'independent_scalar_components_80_and_100_enclosed':len(derivatives)*3,
                        'nominal_summaries':True},'models':models,'per_scan':scans,
            'overall_nominal_spectral_bound_over_sum_norms':stats(summary['nominal_normalized_bounds']),
            'independent_derivatives':derivatives,'run_manifest_sha256':sha(run/'manifest.json'),
            'limits':['mpmath.iv is experimental; rational inequalities do not audit every transcendental implementation.',
                      'High-precision scalar differentiation is an independent numerical check, not formal verification.',
                      'Absolute bounds have inverse world-coordinate units; no cross-dataset physical-unit interpretation.',
                      'Nominal normalized bounds and camera gaps are not interval-certified relative bounds.',
                      '24 selected backprojections are not GT surface points, random surface prevalence or pixel errors.',
                      'No new rendering method, quality gain, training benefit or novelty established.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    result=replay(args.run)
    result.update(created_utc=datetime.now(timezone.utc).isoformat(),status='complete',
                  source_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'research/DIRECTIONAL_FIELD_REPLAY_PROTOCOL.md']})
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','verified','models','per_scan','overall_nominal_spectral_bound_over_sum_norms')},indent=2),flush=True)

if __name__=='__main__':main()
