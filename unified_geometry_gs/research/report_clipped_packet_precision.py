"""Replay all precision artifacts and diagnose every failed synthetic packet post hoc."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def decoded_packet(row,rounded):
    def scalar(x):return mp.mpf(float(np.float32(x)) if rounded else float(x))
    def vector(key):return [scalar(x) for x in row[key]]
    plane=vector('plane');width=scalar(row['width']);conic=vector('conic')
    mean=vector('qmean');bar=vector('qbar');ray=vector('qray');fx,fy=map(scalar,(800.,850.))
    z=plane[2]/mp.sqrt(sum(x*x for x in mean));r=sum(x*x for x in bar);b=r/(width*width);scale=z*z*b
    c=[(plane[i]*(fx,fy)[i]*mp.sqrt(r)/z+bar[i])/r for i in range(2)]
    K=((conic[0]*fx*fx/scale,conic[1]*fx*fy/scale),(conic[1]*fx*fy/scale,conic[2]*fy*fy/scale))
    a=[ray[i]-bar[i] for i in range(2)];m=[mean[i]-bar[i] for i in range(2)]
    A=1+sum(c[i]*a[i] for i in range(2));B=1+sum(c[i]*m[i] for i in range(2))
    def bilinear(x,y):return sum(x[i]*K[i][j]*y[j] for i in range(2) for j in range(2))
    den=A*A+bilinear(a,a);cross=A*B+bilinear(a,m)
    v=[A*m[i]-B*a[i] for i in range(2)]
    gram=(K[0][0]*K[1][1]-K[0][1]**2)*(a[0]*m[1]-a[1]*m[0])**2+bilinear(v,v)
    norm=mp.sqrt(sum(x*x for x in ray));energy=scale*gram/den
    return [z*norm*cross/den,norm/mp.sqrt(b*den),scalar(.75)*mp.exp(-energy/2)]

def exact_factor(row):
    R=[[mp.mpf(float(x)) for x in r] for r in row['rotation']]
    axes=[mp.mpf(float(x)) for x in row['axes']]
    mean=[mp.mpf(float(row['z']))*mp.mpf(float(x)) for x in row['qmean']]
    ray=[mp.mpf(float(x)) for x in row['qray']];norm=mp.sqrt(sum(x*x for x in ray));ray=[x/norm for x in ray]
    local_o=[-sum(R[i][j]*mean[i] for i in range(3))/axes[j] for j in range(3)]
    local_d=[sum(R[i][j]*ray[i] for i in range(3))/axes[j] for j in range(3)]
    A=sum(x*x for x in local_d);p=-sum(x*y for x,y in zip(local_o,local_d))/A
    energy=sum((x+p*y)**2 for x,y in zip(local_o,local_d))
    return [p,1/mp.sqrt(A),mp.mpf('.75')*mp.exp(-energy/2)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    manifest=read(args.run/'manifest.json');assert manifest['status']=='complete'
    with zipfile.ZipFile(args.run/'sources.zip') as archive:
        assert set(archive.namelist())==set(manifest['source_sha256'])
        for name,digest in manifest['source_sha256'].items():assert sha(ROOT/name)==digest and hashlib.sha256(archive.read(name)).hexdigest()==digest
    for name,digest in manifest['artifacts_sha256'].items():assert sha(args.run/name)==digest
    total=gradbad=fp32bad=0;clipbound=0
    for i in range(24):
        row=read(args.run/f'point_{i:02d}.json')
        with np.load(args.run/f'point_{i:02d}.npz',allow_pickle=False) as data:
            assert len(data['selected_ids'])==row['inputs'];total+=row['inputs']
            e=abs(data['raw_packet_gradients']-data['raw_direct_gradients'])/np.maximum(1e-8,abs(data['raw_direct_gradients']))
            assert np.array_equal(e,data['raw_gradient_errors'])
            bad=(~np.isfinite(e))|(e>1e-4)
            assert np.array_equal(bad,data['raw_gradient_bad']);assert int(sum(bad.ravel()))==row['raw_fp64_gradient_bad_components']
            gradbad+=int(sum(bad.ravel()));fp32bad+=int(sum(data['fp32_bad']));clipbound+=int(sum(data['clip_boundary']))
            assert np.max(data['fp32_errors'],axis=0).tolist()==row['fp32_max_errors']
    summary=read(args.run/'summary.json');assert total==summary['inputs']==1992 and total*36==summary['raw_gradient_components']==71712
    assert gradbad==summary['raw_fp64_gradient_bad_components'] and fp32bad==summary['fp32_packet_bad_inputs']
    assert clipbound==summary['clip_boundary_inputs']
    synthetic=read(args.run/'synthetic_fp32.json');theory=read(ROOT/'research/results/clipped_packet_theory_v1.json')['synthetic_cases']
    failed=[r for r in synthetic if not r['pass']];assert len(failed)==summary['synthetic_fp32_bad_cases']
    diagnoses=[];mp.mp.dps=80
    for failedrow in failed:
        row=theory[failedrow['case']];reference=exact_factor(row);rounded=decoded_packet(row,True);original=decoded_packet(row,False)
        error=lambda x:[float(abs(a-b)/max(mp.mpf('1e-12'),abs(b))) for a,b in zip(x,reference)]
        diagnoses.append({'case':row['case'],'clipped':row['clipped'],'posthoc':True,
          'raw_factor_80digit':[mp.nstr(x,40) for x in reference],
          'original_double_packet_80digit_scaled_errors':error(original),
          'rounded_float32_packet_80digit':[mp.nstr(x,40) for x in rounded],
          'rounded_float32_packet_80digit_scaled_errors':error(rounded),
          'saved_float32_evaluation':failedrow['candidate'],'saved_float32_errors':failedrow['errors'],
          'interpretation':'Re-evaluating the rounded packet at high precision still exceeds the original peak gate; this is not repaired by arithmetic precision alone. No universal decoder impossibility theorem is claimed.'})
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','source_sha256':sha(Path(__file__)),
      'verified_sources_and_zip':len(manifest['source_sha256']),'verified_artifacts':len(manifest['artifacts_sha256']),
      'verified_real_inputs':total,'verified_gradient_components':total*36,'raw_gradient_bad_components':gradbad,
      'real_fp32_bad_inputs':fp32bad,'synthetic_fp32_bad_cases':len(failed),'development_gate_pass':summary['development_gate_pass'],
      'failed_cases':diagnoses,'limits':['Replays stored raw AD arithmetic, not a new independent physical gradient oracle.',
        'Posthoc failed-case analysis had already observed case2 and a float64 re-evaluation failure.',
        'MP factor reference treats saved rotation/axes as real binary inputs; no interval certificate.',
        'No actual float32 preprocessing, CUDA backward, shared-primitive accumulation or quality/timing verification.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
