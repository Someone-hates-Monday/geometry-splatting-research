"""Nominal full-world Gaussian camera-quantile comparison on fixed saved rays."""
from __future__ import annotations
import argparse,hashlib,json,os,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from research.affine_rays_cpu import read_model

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def stats(x):return dict(zip(('min','median','max'),map(float,np.quantile(x,[0,.5,1]))))

def restriction(model,origin,direction):
    means,variances,amplitudes,R=model;n=len(means)
    peaks=np.empty(n);widths=np.empty(n);alpha=np.empty(n)
    for lo in range(0,n,65536):
        hi=min(n,lo+65536);local_o=np.einsum('nij,ni->nj',R[lo:hi],origin-means[lo:hi])
        local_d=np.einsum('nij,j->ni',R[lo:hi].transpose(0,2,1),direction)
        A=np.sum(local_d*local_d/variances[lo:hi],axis=1)
        p=-np.sum(local_o*local_d/variances[lo:hi],axis=1)/A
        perpendicular=local_o+p[:,None]*local_d
        peak_q=np.sum(perpendicular*perpendicular/variances[lo:hi],axis=1)
        peaks[lo:hi]=p;widths[lo:hi]=1/np.sqrt(A)
        alpha[lo:hi]=amplitudes[lo:hi]*np.exp(-peak_q/2)
    assert np.all(np.isfinite(peaks)) and np.all(np.isfinite(widths)) and np.all(widths>0)
    assert np.all(np.isfinite(alpha)) and np.all((alpha>=0)&(alpha<1))
    return alpha,peaks,widths

def evaluate(alpha,peaks,widths,t,need_slope=False):
    z=(t-peaks)/widths;G=alpha*np.exp(-z*z/2)
    logvi=.5*np.log1p(-G);terminal=np.log1p(-alpha)
    logT=float(np.where(t<=peaks,logvi,terminal-logvi).sum())
    logV=float(logvi.sum())
    result={'logT_minus':logT,'logV':logV}
    if need_slope:
        result['attenuation']=float(np.sum(G/(2*(1-G))*abs(z)/widths))
    return result

def measure(model,record,camera,legacy):
    origin=np.array(camera['position']);direction=np.array(record['query_world_direction'])
    t_old=record['saved_affine_median_depth'];alpha,peaks,widths=restriction(model,origin,direction)
    assert np.allclose(origin+t_old*direction,record['world_point'],rtol=0,atol=0)
    old=evaluate(alpha,peaks,widths,t_old,True)
    expectedK=record['summed_attenuation'][5]
    logVerror=abs(old['logV']-record['log_product_vacancy'])
    slope_scaled_error=abs(old['attenuation']-expectedK)/max(1e-12,expectedK)
    assert logVerror<=1e-8*max(1.,abs(record['log_product_vacancy']))
    assert slope_scaled_error<=1e-7
    at0=evaluate(alpha,peaks,widths,0.)['logT_minus']
    terminal=float(np.log1p(-alpha).sum());remaining=at0-terminal
    current={**old,'T_camera':float(np.exp(old['logT_minus']-at0)),
             'V':float(np.exp(old['logV'])),'logV_minus_logT_minus':old['logV']-old['logT_minus'],
             'logV_minus_logT_camera':old['logV']-old['logT_minus']+at0}
    result={'scan':record['scan'],'camera':record['camera'],'pixel':record['pixel'],
            'legacy_affine_median':t_old,'legacy_affine_gap':legacy['log_vacancy_transmittance_gap'],
            'gaussians_evaluated':len(alpha),'zero_peak_alpha_underflow':int(sum(alpha==0)),
            'origin_logT_minus':at0,'remaining_optical_depth':remaining,
            'point_logV_check_error':logVerror,'point_attenuation_scaled_error':slope_scaled_error,
            'world_at_legacy_median':current,'camera_median_finite':bool(remaining>np.log(2))}
    if not result['camera_median_finite']:return result
    def f(t):return evaluate(alpha,peaks,widths,t)['logT_minus']-at0+np.log(2)
    high=max(1.,2*t_old);expansions=0
    while f(high)>0 and expansions<32:high*=2;expansions+=1
    if f(high)>0:raise ArithmeticError('Finite world camera quantile not bracketed')
    root,info=brentq(f,0.,high,xtol=5e-13,rtol=1e-13,full_output=True)
    residual=abs(f(root));assert info.converged and residual<=1e-8
    atroot=evaluate(alpha,peaks,widths,root,True)
    delta=root-t_old
    # Diagnostic scale only: all numerically nonzero world peak amplitudes, no new pruning.
    active=alpha>0;typical=float(np.median(widths[active]))
    result.update(world_camera_median=float(root),signed_depth_difference=float(delta),
                  absolute_depth_difference_over_legacy_depth=float(abs(delta)/abs(t_old)),
                  all_nonzero_peak_median_width=typical,absolute_depth_difference_over_width=float(abs(delta)/typical),
                  root_log_residual=residual,solver_iterations=info.iterations,bracket_expansions=expansions,
                  world_at_camera_median={**atroot,'T_camera':float(np.exp(atroot['logT_minus']-at0)),
                  'V':float(np.exp(atroot['logV'])),'logV_minus_logT_minus':atroot['logV']-atroot['logT_minus'],
                  'logV_minus_logT_camera':atroot['logV']-atroot['logT_minus']+at0})
    return result

def execute(out,manifest,run):
    start=time.perf_counter();rows=[]
    for scan in (37,55,63):
        modelpath=ROOT/f'output_compare/ggs_pipeline/scan{scan}';old=read(ROOT/f'research/results/median_cpu_scan{scan}.json')
        ply=modelpath/'point_cloud/iteration_30000/point_cloud.ply'
        assert sha(ply)==next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
        model=read_model(ply);cameras={c['img_name']:c for c in read(modelpath/'cameras.json')}
        for _ in range(8):
            index=len(rows);record=read(run/f'point_{index:02d}.json');assert record['scan']==scan
            result=measure(model,record,cameras[record['camera']],old['rows'][record['legacy_row']])
            rows.append(result);save(out/f'ray_{index:02d}.json',result)
            save(out/'progress.json',{'utc':now(),'completed':len(rows)})
            print(f'{len(rows)}/24 scan{scan} camera{record["camera"]}: finite={result["camera_median_finite"]}',flush=True)
        del model
    finite=[r for r in rows if r['camera_median_finite']]
    groups=[]
    for scan in (37,55,63):
        selected=[r for r in finite if r['scan']==scan]
        groups.append({'scan':scan,'finite':len(selected),'signed_depth_difference':stats([r['signed_depth_difference'] for r in selected]),
                       'abs_difference_over_legacy_depth':stats([r['absolute_depth_difference_over_legacy_depth'] for r in selected]),
                       'old_point_world_camera_T':stats([r['world_at_legacy_median']['T_camera'] for r in selected]),
                       'origin_logT_minus':stats([r['origin_logT_minus'] for r in selected])})
    summary={'points':len(rows),'camera_median_finite':len(finite),'per_scan':groups,
             'max_root_log_residual':max(r['root_log_residual'] for r in finite),
             'max_point_attenuation_scaled_error':max(r['point_attenuation_scaled_error'] for r in rows),
             'max_point_logV_check_error':max(r['point_logV_check_error'] for r in rows),
             'limits':['Nominal double world-ray calculation, not interval-certified root bounds.',
                       'Changing affine shapes, primitive selection/clamps and finite camera origin is confounded.',
                       'No GT, surface prevalence, causal decomposition or quality improvement claim.']}
    save(out/'summary.json',summary)
    manifest.update(status='complete',completed_utc=now(),wall_seconds=time.perf_counter()-start,
                    artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',manifest);print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve()
    if out.exists():raise ValueError('Fresh run directory required')
    run=ROOT/'research/experiments/directional_field_world_v1';prior=read(run/'manifest.json');assert prior['status']=='complete'
    for name,digest in prior['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in prior['artifacts_sha256'].items():assert sha(run/name)==digest
    sources=[Path(__file__),ROOT/'research/WORLD_AFFINE_RAY_PROTOCOL.md']+[ROOT/name for name in prior['source_sha256']]+[run/'manifest.json']+[run/f'point_{i:02d}.json' for i in range(24)]
    out.mkdir(parents=True);manifest={'created_utc':now(),'status':'running','pid':os.getpid(),
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    save(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for source in sources:z.writestr(str(source.relative_to(ROOT)),source.read_bytes())
    try:execute(out,manifest,run)
    except BaseException as exc:
        manifest.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc))
        save(out/'manifest.json',manifest);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
