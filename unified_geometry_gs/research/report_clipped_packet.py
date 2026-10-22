"""Independent original-PLY oracle for the complete clipped/unclipped selected scope."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import mpmath as mp
import numpy as np
from plyfile import PlyData
from research.report_perspective_reparam import oracle

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    manifest=read(args.run/'manifest.json');assert manifest['status']=='complete' and manifest['numerical_gate_pass']
    with zipfile.ZipFile(args.run/'sources.zip') as archive:
        assert set(archive.namelist())==set(manifest['source_sha256'])
        for name,digest in manifest['source_sha256'].items():assert sha(ROOT/name)==digest and hashlib.sha256(archive.read(name)).hexdigest()==digest
    for name,digest in manifest['artifacts_sha256'].items():assert sha(args.run/name)==digest
    mp.mp.dps=80;records=[];maxerr=np.zeros(3);maxclip=np.zeros(3);nall=nclip=0;allpass=True
    for scan in (37,55,63):
        modelpath=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=modelpath/'point_cloud/iteration_30000/point_cloud.ply'
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json')
        assert sha(ply)==next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
        raw=PlyData.read(ply)['vertex'].data;cameras={c['img_name']:c for c in read(modelpath/'cameras.json')}
        for _ in range(8):
            i=len(records);row=read(args.run/f'packet_{i:02d}.json');point=read(ROOT/f'research/experiments/directional_field_world_v1/point_{i:02d}.json')
            assert row['scan']==scan and row['camera']==point['camera'] and row['pixel']==point['pixel']
            camera=cameras[row['camera']];checks=[]
            with np.load(args.run/f'packet_{i:02d}.npz',allow_pickle=False) as data:
                ids=data['selected_ids'];assert len(ids)==row['selected_count'] and sum(data['clipped'])==row['clipped_count']
                with np.load(ROOT/f'research/experiments/perspective_reparam_v1/packet_{i:02d}.npz',allow_pickle=False) as prior:
                    assert np.array_equal(ids,prior['selected_ids']) and np.array_equal(data['clipped'],prior['clipped_center'])
                assert np.array_equal(abs(data['candidate']-data['expected'])/np.maximum(1e-12,abs(data['expected'])),data['errors'])
                for j,identity in enumerate(ids):
                    exact=oracle(raw[int(identity)],camera['position'],point['query_world_direction'])
                    errors=np.array([float(abs(mp.mpf(float(data['candidate'][j,k]))-exact[k])/max(mp.mpf('1e-12'),abs(exact[k]))) for k in range(3)])
                    clipped=bool(data['clipped'][j]);maxerr=np.maximum(maxerr,errors)
                    if clipped:maxclip=np.maximum(maxclip,errors)
                    passed=bool(max(errors)<=1e-8);allpass=allpass and passed;nall+=1;nclip+=clipped
                    checks.append({'gaussian_id':int(identity),'clipped':clipped,'oracle':[mp.nstr(x,35) for x in exact],
                                   'scaled_errors':errors.tolist(),'pass':passed})
            records.append({'point':i,'scan':scan,'camera':row['camera'],'pixel':row['pixel'],'checks':checks})
            print(f'{len(records)}/24 raw oracle complete',flush=True)
        del raw
    summary=read(args.run/'summary.json');assert nall==summary['selected_inputs']==1992 and nclip==summary['clipped_inputs']==46
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','source_sha256':sha(Path(__file__)),
        'verified_sources_and_zip':len(manifest['source_sha256']),'verified_artifacts':len(manifest['artifacts_sha256']),
        'verified_PLYs':3,'independent_80digit_inputs':nall,'including_clipped':nclip,
        'max_scaled_errors_to_raw_oracle':maxerr.tolist(),'max_clipped_errors_to_raw_oracle':maxclip.tolist(),
        'numerical_gate_pass':allpass,'records':records,
        'limits':['Uses the original binary world direction, not a newly normalized camera calibration.',
                  'Independent high precision is numerical evidence, not interval or proof-assistant certification.',
                  'No actual CUDA/VJP/FP32, quality, timing or novelty verification.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','verified_sources_and_zip','verified_artifacts','independent_80digit_inputs','including_clipped','max_scaled_errors_to_raw_oracle','max_clipped_errors_to_raw_oracle','numerical_gate_pass')},indent=2),flush=True)
    if not allpass:raise ArithmeticError('Original raw-oracle gate failed; preserved')

if __name__=='__main__':main()
