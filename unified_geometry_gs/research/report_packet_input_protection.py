"""Verify every saved variant and independently recompute all newly failed factors."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import mpmath as mp

ROOT=Path(__file__).resolve().parents[1]
THRESHOLD=np.array([1e-5,1e-4,1e-4])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def verify(run):
    m=read(run/'manifest.json');assert m['status']=='complete'
    with zipfile.ZipFile(run/'sources.zip') as z:
        assert set(z.namelist())==set(m['source_sha256'])
        for name,digest in m['source_sha256'].items():
            assert sha(ROOT/name)==digest and hashlib.sha256(z.read(name)).hexdigest()==digest
    for name,digest in m['artifacts_sha256'].items():assert sha(run/name)==digest
    return len(m['source_sha256']),len(m['artifacts_sha256'])
def recompute(candidate,target,invalid):
    d=abs(candidate-target)
    e=np.stack((d[...,0]/np.maximum(1e-8,abs(target[...,0])),
                d[...,1]/np.maximum(1e-8,abs(target[...,1])),d[...,2]),axis=-1)
    bad=invalid|np.any(e>THRESHOLD,axis=-1)
    return e,bad
def factor80(data,i):
    def scalar(x):return mp.mpf(float(x))
    R=[[scalar(x) for x in row] for row in data['rotation'][i]];axes=list(map(scalar,data['axes'][i]))
    mean=[scalar(data['z'][i])*scalar(x) for x in data['qmean'][i]]
    ray=list(map(scalar,data['qray'][i]));norm=mp.sqrt(sum(x*x for x in ray));ray=[x/norm for x in ray]
    local_o=[-sum(R[k][j]*mean[k] for k in range(3))/axes[j] for j in range(3)]
    local_d=[sum(R[k][j]*ray[k] for k in range(3))/axes[j] for j in range(3)]
    quad=sum(x*x for x in local_d);p=-sum(x*y for x,y in zip(local_o,local_d))/quad
    energy=sum((x+p*y)**2 for x,y in zip(local_o,local_d))
    return [p,1/mp.sqrt(quad),mp.mpf('.75')*mp.exp(-energy/2)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    selected=ROOT/'research/experiments/packet_input_protection_v1';confirm=ROOT/'research/experiments/packet_input_confirmation_v1'
    a=verify(selected);b=verify(confirm);s=read(selected/'summary.json');c=read(confirm/'summary.json')
    real=np.zeros(64,dtype=int);synthetic=np.zeros(64,dtype=int);checked=0
    for path in sorted(selected.glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            e,bad=recompute(d['candidates'],d['target'][None],d['invalid'])
            assert np.array_equal(e,d['errors']) and np.array_equal(bad,d['bad'])
            count=np.sum(bad,axis=1)
            if path.stem=='synthetic':synthetic+=count
            else:real+=count
            checked+=d['candidates'].shape[0]*d['candidates'].shape[1]
    for row in s['variants']:assert row['real_bad']==real[row['mask']] and row['synthetic_bad']==synthetic[row['mask']]
    passing=np.flatnonzero((real+synthetic)==0).tolist()
    minimal=[m for m in passing if not any(n!=m and (n&m)==n for n in passing)]
    assert passing==s['passing_masks'] and minimal==s['inclusion_minimal_passing_masks']
    fail=set();variant_data={}
    with np.load(confirm/'inputs.npz',allow_pickle=False) as inputs:
        for row in c['variants']:
            with np.load(confirm/(row['name']+'.npz'),allow_pickle=False) as d:
                e,bad=recompute(d['candidate'],inputs['expected'],d['invalid'])
                assert np.array_equal(e,d['errors']) and np.array_equal(bad,d['bad'])
                assert np.flatnonzero(bad).tolist()==row['bad_ids'] and int(sum(bad))==row['bad_inputs']
                assert np.max(e,axis=0).tolist()==row['max_errors']
                fail.update(row['bad_ids']);variant_data[row['name']]={'candidate':d['candidate'].copy(),'bad':bad.copy()}
        mp.mp.dps=80;diagnoses=[]
        for i in sorted(fail):
            ref= factor80(inputs,i);rf=np.asarray([float(x) for x in ref])
            reference_error=abs(inputs['expected'][i]-rf)/np.maximum(1e-12,abs(rf))
            variants={}
            for name,d in variant_data.items():
                e,bad=recompute(d['candidate'][i:i+1],rf[None],np.array([False]))
                assert bool(bad[0])==bool(d['bad'][i]),(i,name,e)
                variants[name]={'errors_vs_factor80':e[0].tolist(),'failed':bool(bad[0])}
            diagnoses.append({'case':i,'reference80':[mp.nstr(x,40) for x in ref],
                'saved_expected_scaled_errors':reference_error.tolist(),'variants':variants})
    result={'status':'complete','checked_utc':datetime.now(timezone.utc).isoformat(),'mpmath':mp.__version__,
      'selection_source_and_artifact_counts':a,'confirmation_source_and_artifact_counts':b,
      'selection_evaluations_verified':checked,'confirmation_evaluations_verified':4096*5,
      'inclusion_minimal_selection_masks':minimal,'confirmation_candidate_bad_counts':{r['name']:r['bad_inputs'] for r in c['variants']},
      'confirmation_candidates_pass':c['selected_candidates_pass'],'independent_factor80_failed_union_cases':len(diagnoses),
      'failed_classifications_match_factor80':True,'diagnoses':diagnoses,
      'source_sha256':sha(Path(__file__)),'limits':['All error summaries replayed; not a second implementation of the decoder.',
         'Independent 80-digit factor oracle covers the union of newly failed synthetic inputs, not all4096.',
         'Saved QR rotations treated as real binary inputs; no interval or proof-assistant certification.',
         'Two rejected candidates cannot be reclassified by the original selection-set pass.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','selection_evaluations_verified','confirmation_evaluations_verified','confirmation_candidate_bad_counts','independent_factor80_failed_union_cases','failed_classifications_match_factor80')},indent=2),flush=True)

if __name__=='__main__':main()
