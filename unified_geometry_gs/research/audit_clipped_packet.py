"""All1992 saved selected inputs, with center clipping handled by the full affine packet."""
from __future__ import annotations
import argparse,hashlib,json,os,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from research.affine_rays_cpu import read_model
from research.affine_ray_ids import project_with_ids,ray_with_ids
from research.clipped_packet import recover

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def finite_list(x):return [float(v) if np.isfinite(v) else None for v in x]

def execute(out,manifest,previous):
    start=time.perf_counter();rows=[]
    for scan in (37,55,63):
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json');modelpath=ROOT/f'output_compare/ggs_pipeline/scan{scan}'
        ply=modelpath/'point_cloud/iteration_30000/point_cloud.ply'
        assert sha(ply)==next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
        model=read_model(ply);cameras={c['img_name']:c for c in read(modelpath/'cameras.json')}
        for _ in range(8):
            index=len(rows);prior=read(previous/f'packet_{index:02d}.json');assert prior['scan']==scan
            camera=cameras[prior['camera']];pixel=prior['pixel'];proj=project_with_ids(model,camera)
            ray=ray_with_ids(proj,pixel);ids=ray['memberships']['selected'];n=len(ids)
            inverse=np.full(len(model[0]),-1,dtype=np.int64);inverse[proj['_gaussian_ids']]=np.arange(len(proj['_gaussian_ids']))
            pos=inverse[ids];assert np.all(pos>=0)
            with np.load(ROOT/f'research/experiments/matched_ray_factors_v1/ray_{index:02d}.npz',allow_pickle=False) as original:
                assert np.array_equal(ids,original['selected_ids'])
                for key in ('alpha','peaks','widths'):assert np.array_equal(ray[key],original['affine_'+key])
                target=np.column_stack((original['world_peaks_selected'],original['world_widths_selected'],original['world_alpha_selected']))
            focal=np.array([proj['fx'],proj['fy']]);center=np.array([(proj['image_width']-1)/2,(proj['image_height']-1)/2])
            # The candidate only receives existing packet data; no raw mean shortcut.
            qmean=np.column_stack(((proj['uv'][pos]-center)/focal,np.ones(n)))
            limit=1.3*np.array([proj['image_width'],proj['image_height']])/(2*focal)
            qbar=np.column_stack((np.clip(qmean[:,:2],-limit,limit),np.ones(n)))
            qray=np.column_stack((np.broadcast_to((np.asarray(pixel)-center)/focal,(n,2)),np.ones(n)))
            clipped=np.any(qmean!=qbar,axis=1)
            plane=proj['plane'][pos];width=proj['widths'][pos];conic=proj['conic'][pos];amp=proj['opacity'][pos]
            p,s,a,diag=recover(plane,width,conic,qmean,qbar,qray,focal,amp);candidate=np.column_stack((p,s,a))
            errors=abs(candidate-target)/np.maximum(1e-12,abs(target))
            invalid=(~np.all(np.isfinite(candidate),axis=1))|(diag['detK']<=0)|(diag['denominator']<=0)|(diag['energy']<0)
            bad=invalid|np.any(errors>1e-8,axis=1)
            with np.load(previous/f'packet_{index:02d}.npz',allow_pickle=False) as oldpacket:
                assert np.array_equal(ids,oldpacket['selected_ids'])
                assert np.array_equal(clipped,oldpacket['clipped_center'])
                olddomain=oldpacket['domain'];v1errors=abs(candidate[olddomain]-oldpacket['candidate'])/np.maximum(1e-12,abs(oldpacket['candidate']))
            np.savez_compressed(out/f'packet_{index:02d}.npz',selected_ids=ids,clipped=clipped,
                plane=plane,width=width,conic=conic,qmean=qmean,qbar=qbar,qray=qray,focal=focal,amplitude=amp,
                expected=target,candidate=candidate,errors=errors,invalid=invalid,bad=bad,
                v1_domain=olddomain,v1_errors=v1errors,**diag)
            result={'scan':scan,'camera':prior['camera'],'pixel':pixel,'selected_count':n,
                'clipped_count':int(sum(clipped)),'invalid_count':int(sum(invalid)),
                'bad_count':int(sum(bad)),'bad_ids':ids[bad].tolist(),'old_unclipped_v1_count':int(sum(olddomain)),
                'max_scaled_errors':finite_list(np.max(errors,axis=0)),
                'max_clipped_scaled_errors':finite_list(np.max(errors[clipped],axis=0)) if np.any(clipped) else None,
                'max_v1_agreement_errors':finite_list(np.max(v1errors,axis=0)) if len(v1errors) else None,
                'numerical_gate_pass':bool(not np.any(bad) and np.all(v1errors<=1e-8))}
            rows.append(result);save(out/f'packet_{index:02d}.json',result)
            save(out/'progress.json',{'utc':now(),'completed':len(rows),'inputs':sum(r['selected_count'] for r in rows)})
            print(f'{len(rows)}/24 scan{scan} cam{prior["camera"]}: clipped={int(sum(clipped))}/{n}, bad={int(sum(bad))}',flush=True)
            del proj,inverse
        del model
    max_errors=np.max([r['max_scaled_errors'] for r in rows],axis=0)
    summary={'points':len(rows),'selected_inputs':sum(r['selected_count'] for r in rows),
        'clipped_inputs':sum(r['clipped_count'] for r in rows),'invalid_inputs':sum(r['invalid_count'] for r in rows),
        'bad_inputs':sum(r['bad_count'] for r in rows),'max_scaled_errors':finite_list(max_errors),
        'max_clipped_scaled_errors':np.max([r['max_clipped_scaled_errors'] for r in rows if r['max_clipped_scaled_errors'] is not None],axis=0).tolist(),
        'max_v1_agreement_errors':np.max([r['max_v1_agreement_errors'] for r in rows if r['max_v1_agreement_errors'] is not None],axis=0).tolist(),
        'numerical_gate_pass':all(r['numerical_gate_pass'] for r in rows),
        'limits':['Full selected1992 input scope, not all primitive/pixel combinations or nonzero screen kernels.',
                  'No CUDA, raw-parameter VJP, FP32, timing, GT, image/geometry quality or novelty claim.',
                  'Packet data may inherit conic conditioning or noninverse covariance/precision roundoff.']}
    save(out/'summary.json',summary)
    manifest.update(status='complete',completed_utc=now(),wall_seconds=time.perf_counter()-start,
        numerical_gate_pass=summary['numerical_gate_pass'],
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',manifest);print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise ValueError('Fresh run required')
    theory=ROOT/'research/results/clipped_packet_theory_v1.json';checks=read(theory);assert checks['overall_gate_pass']
    for name,digest in checks['source_sha256'].items():assert sha(ROOT/'research'/name)==digest
    previous=ROOT/'research/experiments/perspective_reparam_v1';prior=read(previous/'manifest.json')
    assert prior['status']=='complete' and prior['numerical_gate_pass']
    for name,digest in prior['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in prior['artifacts_sha256'].items():assert sha(previous/name)==digest
    sources=set(ROOT/name for name in prior['source_sha256'])
    sources.update([Path(__file__),ROOT/'research/clipped_packet.py',ROOT/'research/verify_clipped_packet.py',
      ROOT/'research/CLIPPED_PACKET_PROTOCOL.md',theory,ROOT/'research/report_perspective_reparam.py',previous/'manifest.json',previous/'summary.json'])
    for i in range(24):sources.update([previous/f'packet_{i:02d}.json',previous/f'packet_{i:02d}.npz'])
    out.mkdir(parents=True);manifest={'created_utc':now(),'status':'running','pid':os.getpid(),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}}
    save(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(sources):archive.writestr(str(source.relative_to(ROOT)),source.read_bytes())
    try:execute(out,manifest,previous)
    except BaseException as exc:
        manifest.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc))
        save(out/'manifest.json',manifest);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
