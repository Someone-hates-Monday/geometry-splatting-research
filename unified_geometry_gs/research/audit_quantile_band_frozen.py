"""Frozen real-ray band-depth diagnostics; never trains or edits a renderer."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings
import zipfile
import numpy as np
import scipy
from scipy.integrate import quad, quad_vec, IntegrationWarning
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.affine_rays_cpu import read_model,project_model,ray_parameters
from research.median_solver import certified_bracket,log_fields

QL, QU = .45, .55


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def root(q,a,p,s):
    lo,hi=certified_bracket(a,p,s,q=q)
    return brentq(lambda t:log_fields(t,a,p,s)[0]-np.log(q),lo,hi,xtol=1e-12,rtol=1e-14)


def band(a,p,s):
    lo,hi=root(QU,a,p,s),root(QL,a,p,s)
    def integrand(t):
        g=a*np.exp(-.5*((t-p)/s)**2)
        lv=.5*np.log1p(-g)
        T=np.exp(np.where(t<=p,lv,np.log1p(-a)-lv).sum())
        position=T*abs(t-p)*g/(2*s*s*(1-g))
        opacity=T*np.where(t<=p,-g/(2*a*(1-g)),-1/(1-a)+g/(2*a*(1-g)))
        return np.r_[T-QL,position,opacity]
    points=sorted(set(float(x) for x in p if lo<x<hi))
    value,error,info=quad_vec(integrand,lo,hi,epsabs=1e-11,epsrel=1e-9,
                              points=points,limit=10000,full_output=True)
    n=len(a); value/=(QU-QL)
    if not info.success or not np.all(np.isfinite(value)):
        raise ArithmeticError('Spatial quadrature failed: '+str(info.message))
    return lo+value[0],value[1:n+1],value[n+1:],error/(QU-QL),int(info.neval),[lo,hi]


def independent(a,p,s):
    levels=[np.exp(log_fields(float(x),a,p,s)[0]) for x in p]
    pts=sorted(set([QL,QU]+[float(x) for x in levels if QL<x<QU]))
    vals=[quad(lambda q:root(q,a,p,s),lo,hi,epsabs=1e-10,epsrel=1e-10,limit=200)
          for lo,hi in zip(pts[:-1],pts[1:])]
    return sum(x[0] for x in vals)/(QU-QL),sum(x[1] for x in vals)/(QU-QL)


def quantiles(values):
    return dict(zip(('min','p50','p90','p99','max'),map(float,np.quantile(values,[0,.5,.9,.99,1])))) if values else None


def summarize(rows):
    valid=[r for r in rows if r.get('band_computed')]
    keys=('absolute_depth_shift','width_normalized_depth_shift','median_dimensionless_log_slope',
          'median_opacity_gradient_l2','band_opacity_gradient_l2','opacity_gradient_norm_ratio',
          'position_gradient_l1_difference','hypothetical_thin_mixed_mass')
    summary={key:quantiles([r[key] for r in valid if r.get(key) is not None]) for key in keys}
    n=len(valid)
    shift=sum(r['width_normalized_depth_shift']>.1 for r in valid)
    mixed=sum(r['hypothetical_thin_mixed_mass']>.1 for r in valid)
    return {'sampled_rays':len(rows),'finite_median_rays':sum(r['finite_median'] for r in rows),
            'finite_band_rays':sum(r['finite_band'] for r in rows),'computed_band_rays':n,
            'failed_check_rays':sum(bool(r.get('failed_checks')) for r in rows),
            'independent_validation_rays':sum('independent_validation' in r for r in rows),
            'summary':summary,'shift_guard_count':shift,'thin_mixed_guard_count':mixed,
            'shift_guard_fraction':shift/n if n else None,'thin_mixed_guard_fraction':mixed/n if n else None,
            'universal_replacement_development_gate_pass':bool(n and shift/n<=.1 and mixed/n<=.1)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists(): raise ValueError('New study directory required')
    out.mkdir(parents=True)
    warnings.simplefilter('error',IntegrationWarning)
    sources=[Path(__file__),ROOT/'research/affine_rays_cpu.py',ROOT/'research/median_solver.py',
             ROOT/'research/QUANTILE_BAND_FROZEN_PROTOCOL.md']
    oldfiles=[ROOT/f'research/results/median_cpu_scan{x}.json' for x in (37,55,63)]
    sources+=oldfiles
    legacy=[json.loads(p.read_text(encoding='utf-8')) for p in oldfiles]
    models=[]
    for scan,x in zip((37,55,63),legacy):
        model=ROOT/f'output_compare/ggs_pipeline/scan{scan}'
        ply=model/'point_cloud/iteration_30000/point_cloud.ply'
        if sha(ply)!=next(v for k,v in x['source_sha256'].items() if k.endswith('point_cloud.ply')):
            raise ValueError('Frozen PLY changed')
        if sha(ROOT/'research/affine_rays_cpu.py')!=next(v for k,v in x['source_sha256'].items() if k.endswith('affine_rays_cpu.py')):
            raise ValueError('Frozen affine source changed')
        sources.extend((model/'cameras.json',model/'cfg_args'))
        models.append({'scan':scan,'model':str(model),'ply_sha256':sha(ply)})
    manifest={'created_utc':now(),'status':'running','pid':__import__('os').getpid(),
              'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'models':models,'completed_scenes':[]}
    write_json(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in sources:archive.writestr(str(p.relative_to(ROOT)),p.read_bytes())
    allrows=[];start=time.perf_counter()
    for item,x in zip(models,legacy):
        scan=item['scan'];modelpath=Path(item['model'])
        model=read_model(modelpath/'point_cloud/iteration_30000/point_cloud.ply')
        cameras=json.loads((modelpath/'cameras.json').read_text(encoding='utf-8'))
        camera_by_name={c['img_name']:c for c in cameras}
        cfg=(modelpath/'cfg_args').read_text(encoding='utf-8')
        if 'resolution=2,' not in cfg or 'kernel_size=0.0,' not in cfg:raise ValueError('Unsupported projection config')
        offsets=[0];alphas=[];peaks=[];widths=[];camera_names=[];pixels=[];rows=[]
        names=list(dict.fromkeys(r['camera'] for r in x['rows']))
        for name in names:
            projected=project_model(model,camera_by_name[name]);validated=False
            for previous in (r for r in x['rows'] if r['camera']==name):
                a,p,s,_=ray_parameters(projected,previous['pixel'])
                alphas.extend(a);peaks.extend(p);widths.extend(s);offsets.append(len(alphas))
                camera_names.append(name);pixels.append(previous['pixel'])
                terminal=float(np.exp(np.log1p(-a).sum()))
                row={'scan':scan,'camera':name,'pixel':previous['pixel'],'contributors':len(a),'terminal_transmission':terminal,
                     'finite_median':terminal<.5,'finite_band':terminal<QL,'failed_checks':[]}
                if len(a)!=previous['contributors'] or row['finite_median']!=previous['finite_root']:
                    raise ValueError('Saved ray selection changed')
                if not row['finite_band']:
                    row['not_computed_reason']='terminal transmittance >= lower band endpoint'
                else:
                    try:
                        t=root(.5,a,p,s)
                        mean,gp,ga,err,neval,bounds=band(a,p,s)
                        _,_,slope=log_fields(t,a,p,s)
                        g=a*np.exp(-.5*((t-p)/s)**2)
                        fp=abs(t-p)*g/(2*s*s*(1-g))
                        fa=np.where(t<=p,g/(2*a*(1-g)),1/(1-a)-g/(2*a*(1-g)))
                        median_gp=fp/slope if slope>0 else None
                        median_ga=-fa/slope if slope>0 else None
                        order=np.argsort(p,kind='stable');C=np.r_[1.,np.cumprod(1-a[order])]
                        weights=np.maximum(0,np.minimum(QU,C[:-1])-np.maximum(QL,C[1:]))/(QU-QL)
                        scale=float(np.median(s));median_norm=float(np.linalg.norm(median_ga)) if median_ga is not None else None
                        row.update({'band_computed':True,'median_depth':t,'band_depth':float(mean),
                                    'old_median_depth_error':abs(t-previous['median_ray_depth']),
                                    'absolute_depth_shift':float(abs(mean-t)),'width_normalized_depth_shift':float(abs(mean-t)/scale),
                                    'median_width':scale,'median_dimensionless_log_slope':float(slope*scale),
                                    'median_opacity_gradient_l2':median_norm,'band_opacity_gradient_l2':float(np.linalg.norm(ga)),
                                    'opacity_gradient_norm_ratio':float(np.linalg.norm(ga)/median_norm) if median_norm else None,
                                    'median_max_position_weight':float(max(median_gp)) if median_gp is not None else None,
                                    'band_max_position_weight':float(max(gp)),'band_position_sum_error':float(abs(sum(gp)-1)),
                                    'position_gradient_l1_difference':float(sum(abs(gp-median_gp))) if median_gp is not None else None,
                                    'hypothetical_thin_mixed_mass':float(max(0.,1-max(weights))),
                                    'hypothetical_thin_band_depth':float(np.dot(p[order],weights)),
                                    'quadrature_error_estimate':float(err),'quadrature_evaluations':neval,'band_root_bounds':bounds})
                        if row['old_median_depth_error']>1e-8:row['failed_checks'].append('legacy median parity')
                        if min(gp)<-1e-10 or row['band_position_sum_error']>1e-7:row['failed_checks'].append('position convex weights')
                        if not validated:
                            validated=True
                            qmean,qerr=independent(a,p,s)
                            ia,ip=int(np.argmax(abs(ga))),int(np.argmax(abs(gp)))
                            ha=min(1e-6,a[ia]/10,(1-a[ia])/10);hp=1e-5*scale
                            ap,am=a.copy(),a.copy();ap[ia]+=ha;am[ia]-=ha
                            pp,pm=p.copy(),p.copy();pp[ip]+=hp;pm[ip]-=hp
                            fda=(band(ap,p,s)[0]-band(am,p,s)[0])/(2*ha)
                            fdp=(band(a,pp,s)[0]-band(a,pm,s)[0])/(2*hp)
                            ea=abs(fda-ga[ia])/max(abs(ga[ia]),1e-5);ep=abs(fdp-gp[ip])/max(abs(gp[ip]),1e-5)
                            row['independent_validation']={'quantile_mean':qmean,'absolute_forward_error':abs(qmean-mean),
                                'quad_reported_error':qerr,'opacity_index':ia,'position_index':ip,'opacity_step':ha,'position_step':hp,
                                'opacity_fd':fda,'position_fd':fdp,'opacity_scaled_error':float(ea),'position_scaled_error':float(ep)}
                            if abs(qmean-mean)>1e-8 or max(ea,ep)>1e-4:row['failed_checks'].append('independent forward/FD')
                    except Exception as exc:
                        row['failed_checks'].append(type(exc).__name__+': '+str(exc))
                rows.append(row)
            del projected
            write_json(out/'progress.json',{'utc':now(),'scan':scan,'camera_completed':name,'rows_completed':len(rows),'failed_check_rays':sum(bool(r['failed_checks']) for r in rows)})
            print(f'scan{scan} camera {name}: {len(rows)} / 1024 rays; failed={sum(bool(r["failed_checks"]) for r in rows)}',flush=True)
        npz=out/f'scan{scan}_rays.npz'
        np.savez_compressed(npz,offsets=np.asarray(offsets),alpha=np.asarray(alphas),peaks=np.asarray(peaks),widths=np.asarray(widths),
                            camera=np.asarray(camera_names),pixels=np.asarray(pixels))
        report={'created_utc':now(),'scan':scan,'input_npz_sha256':sha(npz),'rows':rows,**summarize(rows)}
        write_json(out/f'scan{scan}.json',report);allrows.extend(rows)
        manifest['completed_scenes'].append(scan);write_json(out/'manifest.json',manifest)
        del model
    summary=summarize(allrows)
    manifest.update({'status':'complete','completed_utc':now(),'wall_seconds':time.perf_counter()-start,
                     'artifacts_sha256':{p.name:sha(p) for p in out.iterdir() if p.name not in ('manifest.json','progress.json','summary.json')},
                     'all_declared_numerical_checks_pass':not summary['failed_check_rays']})
    summary.update({'created_utc':now(),'all_declared_numerical_checks_pass':not summary['failed_check_rays'],
                    'limits':['Frozen float64 affine reference, not production gradient, training or quality evidence.',
                              'Related rays are not independent training replicates; no significance claim.',
                              'Thin mixing measure is a hypothetical thinning proxy, not actual geometry error.']})
    write_json(out/'summary.json',summary);write_json(out/'manifest.json',manifest)
    print(json.dumps(summary,indent=2),flush=True)
    if summary['failed_check_rays']:raise SystemExit(2)


if __name__=='__main__':main()
