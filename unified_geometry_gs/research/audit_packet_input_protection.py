"""Exhaust all six input-protection groups without changing the previous gates."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from research.clipped_packet import recover

ROOT=Path(__file__).resolve().parents[1]
GROUPS=(('plane',),('width',),('conic',),('qmean',),('qbar','qray','focal'),('amplitude',))
NAMES=('plane','width','conic','qmean','camera_geometry','amplitude')
KEYS=('plane','width','conic','qmean','qbar','qray','focal','amplitude')
THRESHOLD=np.array([1e-5,1e-4,1e-4])

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)

def evaluate(packet,target,mask=None):
    protected=set() if mask is None else {k for j,g in enumerate(GROUPS) if mask&(1<<j) for k in g}
    values=[]
    for key in KEYS:
        a=np.asarray(packet[key],dtype=np.float64)
        if key not in protected:a=a.astype(np.float32)
        if mask is not None:a=a.astype(np.float64)
        values.append(a)
    p,w,a,d=recover(*values)
    if mask is None:assert p.dtype==w.dtype==a.dtype==np.float32
    candidate=np.column_stack((p,w,a));difference=abs(candidate-target)
    errors=np.column_stack((difference[:,0]/np.maximum(1e-8,abs(target[:,0])),
                           difference[:,1]/np.maximum(1e-8,abs(target[:,1])),difference[:,2]))
    invalid=(~np.all(np.isfinite(candidate),axis=1))|(d['energy']<0)|(d['detK']<=0)|(d['denominator']<=0)
    bad=invalid|np.any(errors>THRESHOLD,axis=1)
    if not np.all(np.isfinite(errors)):raise ArithmeticError('Nonfinite error: preserve arrays before further development')
    return candidate,errors,invalid,bad

def execute(out,manifest,previous,precision,theory):
    start=time.perf_counter();sets=[];bitwise=compared=0;baseline_mask_same=True
    for i in range(24):
        with np.load(previous/f'packet_{i:02d}.npz',allow_pickle=False) as p:
            packet={k:p[k].copy() for k in KEYS};target=p['expected'].copy();clip=p['clipped'].copy();ids=p['selected_ids'].copy()
        with np.load(precision/f'point_{i:02d}.npz',allow_pickle=False) as old:
            base=evaluate(packet,target)
            bitwise+=int(np.sum(base[0]==old['fp32_candidate']));compared+=base[0].size
            baseline_mask_same &= np.array_equal(base[3],old['fp32_bad'])
        sets.append((f'real_{i:02d}',packet,target,clip,ids))
    rows=theory['synthetic_cases']
    packet={k:np.asarray([r[k] for r in rows]) for k in ('plane','width','conic','qmean','qbar','qray')}
    packet.update(focal=np.array([800.,850.]),amplitude=np.full(len(rows),.75))
    target=np.asarray([r['expected'] for r in rows]);clip=np.asarray([r['clipped'] for r in rows]);ids=np.arange(len(rows))
    base=evaluate(packet,target);old=read(precision/'synthetic_fp32.json')
    bitwise+=int(np.sum(base[0]==np.asarray([r['candidate'] for r in old])));compared+=base[0].size
    baseline_mask_same &= np.array_equal(base[3],np.array([not r['pass'] for r in old]))
    sets.append(('synthetic',packet,target,clip,ids))
    totals=[{'mask':m,'protected':[NAMES[j] for j in range(6) if m&(1<<j)],'real_bad':0,'synthetic_bad':0,
             'real_clipped_bad':0,'synthetic_clipped_bad':0,'invalid':0,'max_errors':[0.,0.,0.]} for m in range(64)]
    for name,packet,target,clip,ids in sets:
        candidates=[];errors=[];invalid=[];bad=[]
        base=evaluate(packet,target)
        for m in range(64):
            result=evaluate(packet,target,m)
            candidates.append(result[0]);errors.append(result[1]);invalid.append(result[2]);bad.append(result[3])
            r=totals[m];kind='synthetic' if name=='synthetic' else 'real'
            r[kind+'_bad']+=int(sum(result[3]));r[kind+'_clipped_bad']+=int(sum(result[3]&clip))
            r['invalid']+=int(sum(result[2]));r['max_errors']=np.maximum(r['max_errors'],np.max(result[1],axis=0)).tolist()
            if name=='synthetic':
                r['case2_errors']=result[1][2].tolist()
                r['synthetic_bad_ids']=ids[result[3]].tolist()
        np.savez_compressed(out/f'{name}.npz',selected_ids=ids,clipped=clip,target=target,
            candidates=np.stack(candidates),errors=np.stack(errors),invalid=np.stack(invalid),bad=np.stack(bad),
            fp32_baseline_candidate=base[0],fp32_baseline_errors=base[1],fp32_baseline_bad=base[3])
        save(out/'progress.json',{'utc':now(),'last_set':name})
    assert baseline_mask_same,'Previous FP32 failure classification changed'
    passing=[r['mask'] for r in totals if r['real_bad']==r['synthetic_bad']==0]
    minimal=[m for m in passing if not any(n!=m and (n&m)==n for n in passing)]
    summary={'status':'complete','real_inputs':1992,'synthetic_inputs':128,'combinations':64,
             'baseline_components_bitwise_equal':bitwise,'baseline_components_compared':compared,
             'baseline_failure_masks_same':baseline_mask_same,'passing_masks':passing,'inclusion_minimal_passing_masks':minimal,
             'variants':totals,'limits':['Exploratory selection on the same development inputs; no independent confirmation.',
             'Every combination uses double arithmetic after optional input rounding.',
             'No actual float32/CUDA preprocessing, backward, timing, storage-cost or image-quality conclusion.',
             'Original clipped_packet_precision_v1 overall failure remains unchanged.']}
    save(out/'summary.json',summary)
    manifest.update(status='complete',completed_utc=now(),wall_seconds=time.perf_counter()-start,
         artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',manifest)
    print(json.dumps({k:summary[k] for k in ('status','real_inputs','synthetic_inputs','combinations','baseline_failure_masks_same','passing_masks','inclusion_minimal_passing_masks')},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise ValueError('Fresh output required')
    previous=ROOT/'research/experiments/clipped_packet_v1';precision=ROOT/'research/experiments/clipped_packet_precision_v1'
    sources={Path(__file__),ROOT/'research/clipped_packet.py',ROOT/'research/PACKET_INPUT_PROTECTION_PROTOCOL.md',ROOT/'research/results/clipped_packet_theory_v1.json'}
    for run in (previous,precision):
        m=read(run/'manifest.json');assert m['status']=='complete'
        for name,digest in m['artifacts_sha256'].items():assert sha(run/name)==digest;sources.add(run/name)
        sources.add(run/'manifest.json')
    out.mkdir(parents=True)
    manifest={'created_utc':now(),'status':'running','pid':os.getpid(),'platform':platform.platform(),'numpy':np.__version__,
              'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)}}
    save(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for source in sorted(sources):z.writestr(source.relative_to(ROOT).as_posix(),source.read_bytes())
    try:execute(out,manifest,previous,precision,read(ROOT/'research/results/clipped_packet_theory_v1.json'))
    except BaseException as exc:
        manifest.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc))
        save(out/'manifest.json',manifest);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
