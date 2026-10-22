"""Fixed-ray factorial shape contrasts and ordered selection-rule decomposition."""
from __future__ import annotations
import argparse,hashlib,itertools,json,os,platform,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import brentq
from research.affine_rays_cpu import read_model,project_model,ray_parameters
from research.affine_ray_ids import project_with_ids,ray_with_ids,verify_projection
from research.audit_world_affine_rays import restriction,evaluate

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def stats(x):return dict(zip(('min','median','max'),map(float,np.quantile(x,[0,.5,1])))) if len(x) else None

def root(alpha,peaks,widths,initial,camera=False):
    alpha,peaks,widths=[np.asarray(x) for x in (alpha,peaks,widths)]
    assert np.all((alpha>=0)&(alpha<1)) and np.all(widths>0)
    original_count=len(alpha);nz=alpha>0
    alpha,peaks,widths=alpha[nz],peaks[nz],widths[nz]
    at0=evaluate(alpha,peaks,widths,0.)['logT_minus'] if camera else 0.
    total=-float(np.log1p(-alpha).sum());remaining=total+at0
    result={'finite':bool(remaining>np.log(2)),'members':original_count,'nominal_nonzero_alpha':len(alpha),
            'nominal_zero_alpha_skipped':original_count-len(alpha),'optical_depth':total,
            'origin_logT_minus':at0,'remaining_optical_depth':remaining,'camera_start':camera}
    if not result['finite']:return result
    def f(t):return evaluate(alpha,peaks,widths,t)['logT_minus']-at0+np.log(2)
    span=max(1.,2*abs(initial));lo=0. if camera else -span;hi=span;expand=0
    while f(lo)<0 or f(hi)>0:
        if expand>=32:raise ArithmeticError('Finite quantile bracket not found')
        span*=2;lo=0. if camera else -span;hi=span;expand+=1
    value,info=brentq(f,lo,hi,xtol=5e-13,rtol=1e-13,full_output=True)
    residual=abs(f(value));assert info.converged and residual<=1e-8
    result.update(depth=float(value),log_residual=residual,iterations=info.iterations,bracket_expansions=expand)
    return result

def shape_allocation(arms):
    if not all(x['finite'] for x in arms):return None
    allocations=np.zeros(3)
    for order in itertools.permutations(range(3)):
        mask=0
        for i in order:
            nxt=mask|(1<<i);allocations[i]+=arms[nxt]['depth']-arms[mask]['depth'];mask=nxt
    allocations/=6
    error=abs(sum(allocations)-(arms[7]['depth']-arms[0]['depth']))
    assert error<=1e-10*max(1.,abs(arms[7]['depth']),abs(arms[0]['depth']))
    return {'peak':float(allocations[0]),'width':float(allocations[1]),'amplitude':float(allocations[2]),
            'sum_error':error,'meaning':'Mean marginal increments over all six orders; no causal uniqueness claim.'}

def measure(model,camera,record,legacy,prior,projected,reference,index,out):
    ray=ray_with_ids(projected,record['pixel']);original=ray_parameters(reference,record['pixel'])
    for key,value in zip(('alpha','peaks','widths','init'),original):
        if isinstance(value,np.ndarray):assert np.array_equal(value,ray[key]),key
        else:assert value==ray[key],key
    n=len(model[0]);members=ray['memberships'];selected=members['selected']
    assert len(selected)==legacy['contributors'] and len(np.unique(selected))==len(selected)
    sets=[set(members[name]) for name in ('selected','eligible','tile','valid','near')]
    assert all(a<=b for a,b in zip(sets,sets[1:]))
    world=restriction(model,np.asarray(camera['position']),np.asarray(record['query_world_direction']))
    wa,wp,ww=world;wacap=np.minimum(.99,wa[selected]);initial=legacy['median_ray_depth']
    arms=[]
    for mask in range(8):
        p=wp[selected] if mask&1 else ray['peaks']
        w=ww[selected] if mask&2 else ray['widths']
        a=wacap if mask&4 else ray['alpha']
        arms.append(root(a,p,w,initial))
    assert arms[0]['finite']
    baseline_error=abs(arms[0]['depth']-initial)
    assert baseline_error<=max(1e-8,2*legacy['root_absolute_error_bound'])
    stages={'selected_cap':arms[7]}
    for name in ('selected','eligible','tile','valid','near'):
        ids=members[name];stages[name]=root(wa[ids],wp[ids],ww[ids],initial)
    stages['full']=root(wa,wp,ww,initial)
    stages['full_camera']=root(wa,wp,ww,initial,camera=True)
    assert stages['full_camera']['finite']==prior['camera_median_finite']
    final_error=None
    if stages['full_camera']['finite']:
        final_error=abs(stages['full_camera']['depth']-prior['world_camera_median'])
        assert final_error<=1e-8*max(1.,abs(prior['world_camera_median']))
    names=['selected_cap','selected','eligible','tile','valid','near','full','full_camera']
    labels=['uncap_world_amplitude','remove_affine_prefix_stop','remove_affine_power_alpha_eligibility',
            'remove_pixel_tile_membership','remove_projection_valid_image_extent','remove_center_near_plane',
            'finite_camera_origin']
    increments={}
    for before,after,label in zip(names,names[1:],labels):
        a,b=stages[before],stages[after]
        increments[label]=b['depth']-a['depth'] if a['finite'] and b['finite'] else None
    allocation=shape_allocation(arms);telescope=None
    if allocation is not None and all(x is not None for x in increments.values()):
        vector=[allocation[key] for key in ('peak','width','amplitude')]+list(increments.values())
        total=stages['full_camera']['depth']-arms[0]['depth']
        error=abs(sum(vector)-total)
        assert error<=1e-9*max(1.,abs(total))
        telescope={'total_world_camera_minus_affine':total,'component_sum':sum(vector),'sum_error':error,
                   'component_L1':sum(abs(x) for x in vector),
                   'cancellation_fraction':1-abs(total)/sum(abs(x) for x in vector) if any(vector) else 0.}
    packed={}
    for name,ids in members.items():
        mask=np.zeros(n,dtype=np.uint8);mask[ids]=1;packed['membership_'+name]=np.packbits(mask,bitorder='little')
    np.savez_compressed(out/f'ray_{index:02d}.npz',model_count=np.int64(n),selected_ids=selected,
        affine_alpha=ray['alpha'],affine_peaks=ray['peaks'],affine_widths=ray['widths'],
        world_alpha_selected=wa[selected],world_peaks_selected=wp[selected],world_widths_selected=ww[selected],
        **packed)
    return {'scan':record['scan'],'camera':record['camera'],'pixel':record['pixel'],
            'legacy_affine_depth':initial,'gaussian_count':n,'projection_and_ray_arrays_bitwise_equal':True,
            'stage_membership_counts':{name:len(ids) for name,ids in members.items()},
            'baseline_root_error':baseline_error,'prior_world_camera_root_error':final_error,
            'world_full_zero_peak_alpha_count':int(sum(wa==0)),
            'selected_world_amplitude_over_cap':int(sum(wa[selected]>.99)),
            'eligible_affine_amplitude_over_cap':ray['eligible_capped_count'],
            'tile_positive_power_count':ray['tile_positive_power_count'],
            'tile_below_alpha_threshold_count':ray['tile_below_alpha_threshold_count'],
            'factor_order':['peak','width','amplitude'],'factorial_arms':arms,
            'permutation_average_allocation':allocation,'world_stages':stages,
            'ordered_increments':increments,'telescope':telescope}

def execute(out,manifest,previous):
    start=time.perf_counter();rows=[]
    for scan in (37,55,63):
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json');modelpath=ROOT/f'output_compare/ggs_pipeline/scan{scan}'
        ply=modelpath/'point_cloud/iteration_30000/point_cloud.ply'
        assert sha(ply)==next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
        model=read_model(ply);cameras={c['img_name']:c for c in read(modelpath/'cameras.json')}
        for _ in range(8):
            i=len(rows);record=read(ROOT/f'research/experiments/directional_field_world_v1/point_{i:02d}.json')
            assert record['scan']==scan;camera=cameras[record['camera']]
            reference=project_model(model,camera);projected=project_with_ids(model,camera)
            verify_projection(reference,projected)
            result=measure(model,camera,record,old['rows'][record['legacy_row']],read(previous/f'ray_{i:02d}.json'),projected,reference,i,out)
            rows.append(result);save(out/f'ray_{i:02d}.json',result)
            save(out/'progress.json',{'utc':now(),'completed':len(rows),'scan':scan,'camera':record['camera']})
            print(f'{len(rows)}/24 scan{scan} cam{record["camera"]}: factorial={result["permutation_average_allocation"] is not None}, telescope={result["telescope"] is not None}',flush=True)
            del reference,projected
        del model
    finite=[r for r in rows if r['telescope'] is not None]
    component_summaries={}
    for key in ('peak','width','amplitude'):
        values=[abs(r['permutation_average_allocation'][key])/abs(r['legacy_affine_depth']) for r in rows if r['permutation_average_allocation'] is not None]
        component_summaries['factor_'+key]={'defined':len(values),'absolute_increment_over_legacy_depth':stats(values)}
    for key in rows[0]['ordered_increments']:
        values=[abs(r['ordered_increments'][key])/abs(r['legacy_affine_depth']) for r in rows if r['ordered_increments'][key] is not None]
        component_summaries[key]={'defined':len(values),'absolute_increment_over_legacy_depth':stats(values)}
    summary={'points':len(rows),'projection_bitwise_parity':24,'ray_parameter_bitwise_parity':24,
      'complete_factorial_cases':sum(r['permutation_average_allocation'] is not None for r in rows),
      'complete_telescope_cases':len(finite),
      'stage_finite_counts':{name:sum(r['world_stages'][name]['finite'] for r in rows) for name in rows[0]['world_stages']},
      'max_baseline_root_error':max(r['baseline_root_error'] for r in rows),
      'max_prior_world_root_error':max(r['prior_world_camera_root_error'] for r in rows if r['prior_world_camera_root_error'] is not None),
      'max_telescope_error':max((r['telescope']['sum_error'] for r in finite),default=None),
      'component_summaries':component_summaries,
      'cancellation_fraction':stats([r['telescope']['cancellation_fraction'] for r in finite]),
      'limits':['Fixed selected rays and original CPU graph, not population prevalence or production CUDA parity.',
        'Factorial ray hybrids need not arise from a shared 3D Gaussian model.',
        'Permutation averages are standard Shapley-style allocations; ordered rule differences depend on the chosen path.',
        'No GT, causal identification, reconstruction/image improvement, novelty or acceleration claim.']}
    save(out/'summary.json',summary)
    manifest.update(status='complete',completed_utc=now(),wall_seconds=time.perf_counter()-start,
      artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',manifest);print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise ValueError('Fresh run directory required')
    previous=ROOT/'research/experiments/world_affine_rays_v1';prior=read(previous/'manifest.json');assert prior['status']=='complete'
    for name,digest in prior['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in prior['artifacts_sha256'].items():assert sha(previous/name)==digest
    sources=set(ROOT/name for name in prior['source_sha256'])
    sources.update([Path(__file__),ROOT/'research/affine_ray_ids.py',ROOT/'research/MATCHED_RAY_FACTOR_PROTOCOL.md',previous/'manifest.json',previous/'summary.json'])
    sources.update(previous/f'ray_{i:02d}.json' for i in range(24))
    manifest={'created_utc':now(),'status':'running','pid':os.getpid(),
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}}
    out.mkdir(parents=True);save(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(sources):archive.writestr(str(source.relative_to(ROOT)),source.read_bytes())
    try:execute(out,manifest,previous)
    except BaseException as exc:
        manifest.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc))
        save(out/'manifest.json',manifest);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
