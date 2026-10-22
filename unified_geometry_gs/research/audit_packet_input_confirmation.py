"""Locked candidates on independently generated same-distribution numerical cases."""
from __future__ import annotations
import argparse,hashlib,json,os,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from research.audit_packet_input_protection import evaluate

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)

def execute(out,manifest):
    start=time.perf_counter();rng=np.random.default_rng(13017)
    fields={k:[] for k in ('rotation','axes','qmean','qbar','qray','z','plane','width','conic','expected')}
    focal=np.array([800.,850.]);amplitude=.75
    for i in range(4096):
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)));axes=10**rng.uniform(-2,0,3);variance=axes**2
        qmean=np.r_[rng.uniform(-1.8,1.8,2),1.];qbar=np.r_[np.clip(qmean[:2],-.65,.65),1.]
        qray=np.r_[rng.uniform(-.8,.8,2),1.];z=rng.uniform(.5,5)
        P=(rotation/variance)@rotation.T;cov=(rotation*variance)@rotation.T
        rbar=qbar@qbar;pb=P@qbar;b=qbar@pb
        J=np.array([[focal[0]/z,0.,-focal[0]*qbar[0]/z],[0.,focal[1]/z,-focal[1]*qbar[1]/z]])
        inv=np.linalg.inv(J@cov@J.T);conic=np.array([inv[0,0],inv[0,1],inv[1,1]])
        linear=z/np.sqrt(rbar)*(rbar*pb/b-qbar)[:2]/focal;plane=np.r_[linear,z*np.linalg.norm(qmean)];width=np.sqrt(rbar/b)
        direction=qray/np.linalg.norm(qray);mean=z*qmean;local_o=-rotation.T@mean;local_d=rotation.T@direction
        quadratic=np.sum(local_d**2/variance);peak=-np.sum(local_o*local_d/variance)/quadratic
        perpendicular=local_o+peak*local_d
        expected=np.array([peak,1/np.sqrt(quadratic),amplitude*np.exp(-.5*np.sum(perpendicular**2/variance))])
        for k in fields:fields[k].append(locals()[k])
    fields={k:np.asarray(v) for k,v in fields.items()};fields.update(focal=focal,amplitude=np.full(4096,amplitude))
    fields['clipped']=np.any(fields['qbar']!=fields['qmean'],axis=1)
    np.savez_compressed(out/'inputs.npz',**fields)
    rows=[]
    for name,mask in (('fp32',None),('all_rounded_double_decode',0),('qmean_protected',8),('other_minimal_protection',21),('all_double',63)):
        candidate,errors,invalid,bad=evaluate(fields,fields['expected'],mask)
        np.savez_compressed(out/f'{name}.npz',candidate=candidate,errors=errors,invalid=invalid,bad=bad)
        rows.append({'name':name,'mask':mask,'inputs':4096,'bad_inputs':int(sum(bad)),
          'bad_ids':np.flatnonzero(bad).tolist(),'clipped_bad':int(sum(bad&fields['clipped'])),
          'invalid_inputs':int(sum(invalid)),'max_errors':np.max(errors,axis=0).tolist(),
          'error_quantiles':{str(q):np.quantile(errors,q,axis=0).tolist() for q in (.5,.95,.99)},
          'all_inputs_pass':not bool(np.any(bad))})
        save(out/'progress.json',{'utc':now(),'last_variant':name})
    summary={'status':'complete','seed':13017,'inputs':4096,'clipped_inputs':int(sum(fields['clipped'])),
       'variants':rows,'selected_candidates_pass':all(r['all_inputs_pass'] for r in rows if r['mask'] in (8,21)),
       'limits':['Same synthetic distribution and double packet generator as selection; independently drawn inputs only.',
          'No new real scenes or actual float32/CUDA preprocessing, backward, quality or timing.',
          'No re-selection among masks after observing this result; old failures remain failures.']}
    save(out/'summary.json',summary)
    manifest.update(status='complete',completed_utc=now(),wall_seconds=time.perf_counter()-start,
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',manifest)
    print(json.dumps({'status':summary['status'],'inputs':4096,'selected_candidates_pass':summary['selected_candidates_pass'],
        'variants':[{k:r[k] for k in ('name','bad_inputs','max_errors')} for r in rows]},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise ValueError('Fresh output required')
    selected=ROOT/'research/experiments/packet_input_protection_v1/summary.json'
    s=json.loads(selected.read_text(encoding='utf-8'));assert s['inclusion_minimal_passing_masks']==[8,21]
    sources=[Path(__file__),ROOT/'research/audit_packet_input_protection.py',ROOT/'research/clipped_packet.py',
             ROOT/'research/PACKET_INPUT_PROTECTION_CONFIRM_PROTOCOL.md',selected]
    out.mkdir(parents=True);manifest={'created_utc':now(),'status':'running','pid':os.getpid(),'numpy':np.__version__,
         'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources}}
    save(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.writestr(p.relative_to(ROOT).as_posix(),p.read_bytes())
    try:execute(out,manifest)
    except BaseException as exc:
        manifest.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc));save(out/'manifest.json',manifest)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
