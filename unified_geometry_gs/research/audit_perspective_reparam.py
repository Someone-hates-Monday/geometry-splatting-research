"""Apply the frozen affine-packet identity to all matched selected real primitives."""
from __future__ import annotations
import argparse,hashlib,json,os,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from research.affine_rays_cpu import read_model
from research.affine_ray_ids import project_with_ids,ray_with_ids
from research.perspective_reparam import recover

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)

def execute(out,manifest,previous):
    start=time.perf_counter();rows=[]
    for scan in (37,55,63):
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json');modelpath=ROOT/f'output_compare/ggs_pipeline/scan{scan}'
        ply=modelpath/'point_cloud/iteration_30000/point_cloud.ply'
        assert sha(ply)==next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
        model=read_model(ply);cameras={c['img_name']:c for c in read(modelpath/'cameras.json')}
        for _ in range(8):
            i=len(rows);prior=read(previous/f'ray_{i:02d}.json');assert prior['scan']==scan
            camera=cameras[prior['camera']];pixel=prior['pixel'];projected=project_with_ids(model,camera)
            ray=ray_with_ids(projected,pixel);ids=ray['memberships']['selected'];n=len(ids)
            with np.load(previous/f'ray_{i:02d}.npz',allow_pickle=False) as data:
                assert np.array_equal(ids,data['selected_ids'])
                for key in ('alpha','peaks','widths'):assert np.array_equal(ray[key],data['affine_'+key])
                expected=np.column_stack((data['world_peaks_selected'],data['world_widths_selected'],data['world_alpha_selected']))
            inverse=np.full(len(model[0]),-1,dtype=np.int64);inverse[projected['_gaussian_ids']]=np.arange(len(projected['_gaussian_ids']))
            position=inverse[ids];assert np.all(position>=0)
            mean_camera=(model[0][ids]-camera['position'])@np.asarray(camera['rotation'])
            q0=mean_camera/mean_camera[:,2,None]
            diff=projected['uv'][position]-pixel;conic=projected['conic'][position]
            delta=np.column_stack((-diff[:,0]/projected['fx'],-diff[:,1]/projected['fy'],np.zeros(n)))
            power=-.5*(conic[:,0]*diff[:,0]**2+conic[:,2]*diff[:,1]**2)-conic[:,1]*diff[:,0]*diff[:,1]
            energy=-2*power;plane=projected['plane'][position];t0=plane[:,2];s0=projected['widths'][position]
            shift=(plane[:,:2]*diff).sum(axis=1);amplitude=projected['opacity'][position]
            clipped=(abs(q0[:,0])>1.3*projected['image_width']/(2*projected['fx']))|(abs(q0[:,1])>1.3*projected['image_height']/(2*projected['fy']))
            invalid=(energy<0)|(~np.isfinite(energy))|(~np.isfinite(s0))|(s0<=0)|(t0<=0)
            domain=~clipped&~invalid;indices=np.flatnonzero(domain)
            recovered=recover(t0[domain],s0[domain],shift[domain],energy[domain],q0[domain],delta[domain],amplitude[domain])
            candidate=np.column_stack(recovered[:3]);assert np.all(np.isfinite(candidate)) and np.all(recovered[3]>0)
            target=expected[domain];errors=abs(candidate-target)/np.maximum(1e-12,abs(target))
            affine=np.column_stack((ray['peaks'],ray['widths'],ray['alpha']))
            affine_errors=abs(affine[domain]-target)/np.maximum(1e-12,abs(target))
            bad=np.any(errors>1e-8,axis=1)
            np.savez_compressed(out/f'packet_{i:02d}.npz',selected_ids=ids,domain=domain,clipped_center=clipped,invalid_input=invalid,
                q0=q0,delta=delta,t0=t0,s0=s0,linear_peak_shift=shift,energy=energy,amplitude=amplitude,
                applicable_indices=indices,candidate=candidate,expected=target,errors=errors,affine_errors=affine_errors,
                recovered_denominator=recovered[3])
            result={'scan':scan,'camera':prior['camera'],'pixel':pixel,'selected_count':n,
                'applicable_count':len(indices),'excluded_clipped_center':int(sum(clipped)),
                'excluded_invalid_input':int(sum(invalid)),'excluded_ids':ids[~domain].tolist(),
                'bad_count':int(sum(bad)),'bad_ids':ids[domain][bad].tolist(),
                'max_scaled_errors':np.max(errors,axis=0).tolist() if len(indices) else None,
                'max_affine_scaled_errors':np.max(affine_errors,axis=0).tolist() if len(indices) else None,
                'numerical_gate_pass':bool(not np.any(bad))}
            rows.append(result);save(out/f'packet_{i:02d}.json',result)
            save(out/'progress.json',{'utc':now(),'completed':len(rows),'applicable':sum(r['applicable_count'] for r in rows)})
            print(f'{len(rows)}/24 scan{scan} cam{prior["camera"]}: applicable={len(indices)}/{n}, bad={int(sum(bad))}',flush=True)
            del projected,inverse
        del model
    defined=[r for r in rows if r['max_scaled_errors'] is not None]
    summary={'points':len(rows),'selected_inputs':sum(r['selected_count'] for r in rows),
        'applicable_inputs':sum(r['applicable_count'] for r in rows),'excluded_clipped_center':sum(r['excluded_clipped_center'] for r in rows),
        'excluded_invalid_input':sum(r['excluded_invalid_input'] for r in rows),'bad_inputs':sum(r['bad_count'] for r in rows),
        'max_scaled_errors':np.max([r['max_scaled_errors'] for r in defined],axis=0).tolist(),
        'max_affine_scaled_errors':np.max([r['max_affine_scaled_errors'] for r in defined],axis=0).tolist(),
        'numerical_gate_pass':all(r['numerical_gate_pass'] for r in rows),
        'limits':['Selected primitive-ray inputs only; clipped and invalid-domain inputs retained separately.',
                  'Float64 CPU calculation; no CUDA, VJP, new-view image, GT, timing or novelty claim.',
                  'Mathematical equivalence does not repair the existing selection rules or conic roundoff.']}
    save(out/'summary.json',summary)
    manifest.update(status='complete',completed_utc=now(),wall_seconds=time.perf_counter()-start,
                    numerical_gate_pass=summary['numerical_gate_pass'],
                    artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',manifest);print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise ValueError('Fresh run required')
    theory=ROOT/'research/results/perspective_reparam_theory_v1.json';theoretical=read(theory);assert theoretical['numerical_gate_pass']
    for name,digest in theoretical['source_sha256'].items():assert sha(ROOT/'research'/name)==digest
    previous=ROOT/'research/experiments/matched_ray_factors_v1';prior=read(previous/'manifest.json');assert prior['status']=='complete'
    for name,digest in prior['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in prior['artifacts_sha256'].items():assert sha(previous/name)==digest
    sources=set(ROOT/name for name in prior['source_sha256'])
    sources.update([Path(__file__),ROOT/'research/perspective_reparam.py',ROOT/'research/PERSPECTIVE_REPARAM_PROTOCOL.md',
                    ROOT/'research/verify_perspective_reparam.py',theory,previous/'manifest.json',previous/'summary.json'])
    for i in range(24):sources.update([previous/f'ray_{i:02d}.json',previous/f'ray_{i:02d}.npz'])
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
