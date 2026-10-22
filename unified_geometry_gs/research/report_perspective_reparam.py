"""Independent raw-PLY high-precision forward oracle for all applicable real packets."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import mpmath as mp
import numpy as np
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def oracle(row,origin,direction):
    mean=[mp.mpf(float(row[k])) for k in ('x','y','z')]
    q=[mp.mpf(float(row[f'rot_{i}'])) for i in range(4)]
    qnorm=mp.sqrt(sum(x*x for x in q));w,x,y,z=[x/qnorm for x in q]
    R=((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),
       (2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),
       (2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)))
    variance=[mp.exp(2*mp.mpf(float(row[f'scale_{i}']))) for i in range(3)]
    smooth=[v+mp.mpf(float(row['filter_3D']))**2 for v in variance]
    amp=mp.sqrt(mp.fprod(a/b for a,b in zip(variance,smooth)))/(1+mp.exp(-mp.mpf(float(row['opacity']))))
    displacement=[mp.mpf(float(origin[i]))-mean[i] for i in range(3)]
    d=[mp.mpf(float(v)) for v in direction]
    local_o=[sum(R[i][j]*displacement[i] for i in range(3)) for j in range(3)]
    local_d=[sum(R[i][j]*d[i] for i in range(3)) for j in range(3)]
    A=sum(d*d/v for d,v in zip(local_d,smooth))
    peak=-sum(a*b/v for a,b,v in zip(local_o,local_d,smooth))/A
    energy=sum((a+peak*b)**2/v for a,b,v in zip(local_o,local_d,smooth))
    return (peak,1/mp.sqrt(A),amp*mp.exp(-energy/2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    m=read(args.run/'manifest.json');assert m['status']=='complete' and m['numerical_gate_pass']
    with zipfile.ZipFile(args.run/'sources.zip') as archive:
        assert set(archive.namelist())==set(m['source_sha256'])
        for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest and hashlib.sha256(archive.read(name)).hexdigest()==digest
    for name,digest in m['artifacts_sha256'].items():assert sha(args.run/name)==digest
    mp.mp.dps=80;checks=[];models=[];selected=0;applicable=0;clipped=0;maxerr=[0.,0.,0.]
    for scan in (37,55,63):
        modelpath=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=modelpath/'point_cloud/iteration_30000/point_cloud.ply'
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json')
        expected=next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
        assert sha(ply)==expected;raw=PlyData.read(ply)['vertex'].data
        cameras={c['img_name']:c for c in read(modelpath/'cameras.json')}
        for _ in range(8):
            index=len(checks);r=read(args.run/f'packet_{index:02d}.json');assert r['scan']==scan
            point=read(ROOT/f'research/experiments/directional_field_world_v1/point_{index:02d}.json')
            assert r['camera']==point['camera'] and r['pixel']==point['pixel']
            camera=cameras[r['camera']];per_input=[]
            with np.load(args.run/f'packet_{index:02d}.npz',allow_pickle=False) as data:
                ids=data['selected_ids'];domain=data['domain'];positions=np.flatnonzero(domain)
                assert np.array_equal(positions,data['applicable_indices'])
                assert len(ids)==r['selected_count'] and sum(domain)==r['applicable_count']
                assert np.array_equal(~domain,data['clipped_center']|data['invalid_input'])
                assert ids[~domain].tolist()==r['excluded_ids']
                target=data['expected'];candidate=data['candidate']
                assert np.array_equal(abs(candidate-target)/np.maximum(1e-12,abs(target)),data['errors'])
                if len(positions):assert np.max(data['errors'],axis=0).tolist()==r['max_scaled_errors']
                for j,pos in enumerate(positions):
                    gaussian_id=int(ids[pos]);exact=oracle(raw[gaussian_id],camera['position'],point['query_world_direction'])
                    errors=[float(abs(mp.mpf(float(candidate[j,k]))-exact[k])/max(mp.mpf('1e-12'),abs(exact[k]))) for k in range(3)]
                    maxerr=[max(a,b) for a,b in zip(maxerr,errors)]
                    per_input.append({'gaussian_id':gaussian_id,'oracle':[mp.nstr(x,35) for x in exact],
                                      'scaled_errors':errors,'pass':max(errors)<=1e-8})
                selected+=len(ids);applicable+=len(positions);clipped+=int(sum(data['clipped_center']))
            checks.append({'point':index,'scan':scan,'camera':r['camera'],'pixel':r['pixel'],'inputs':per_input,
                           'excluded_ids':r['excluded_ids']})
            print(f'{len(checks)}/24 independently recomputed',flush=True)
        models.append({'scan':scan,'gaussian_count':len(raw),'sha256':expected});del raw
    summary=read(args.run/'summary.json')
    assert selected==summary['selected_inputs'] and applicable==summary['applicable_inputs'] and clipped==summary['excluded_clipped_center']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','source_sha256':sha(Path(__file__)),
      'verified_sources_and_zip':len(m['source_sha256']),'verified_artifacts':len(m['artifacts_sha256']),
      'models':models,'selected_inputs':selected,'independent_raw_80digit_inputs':applicable,'excluded_clipped_center':clipped,
      'max_scaled_errors_to_raw_oracle':maxerr,'numerical_gate_pass':all(x['pass'] for r in checks for x in r['inputs']),
      'checks':checks,'limits':['The oracle evaluates the exact binary saved world direction; it does not renormalize or orthogonalize camera calibration.',
        'High precision is numerical cross-checking, not interval or proof-assistant certification.',
        'Clipped-domain inputs remain excluded, not passed. No CUDA, backward, timing, GT, quality or novelty claim.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','verified_sources_and_zip','verified_artifacts','selected_inputs','independent_raw_80digit_inputs','excluded_clipped_center','max_scaled_errors_to_raw_oracle','numerical_gate_pass')},indent=2),flush=True)
    if not result['numerical_gate_pass']:raise ArithmeticError('Independent raw-oracle gate failed; all inputs preserved')

if __name__=='__main__':main()
