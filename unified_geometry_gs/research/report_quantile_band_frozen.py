"""Independently verify frozen sources/artifacts and extreme q-space VJPs."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from research.audit_quantile_band_frozen import band,summarize,QL,QU
from research.median_solver import certified_bracket

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'research/results/quantile_band_frozen_report_v2.json'
    if out.exists():raise ValueError('New report required')
    run=ROOT/'research/experiments/quantile_band_frozen_v1'
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete' and manifest['all_declared_numerical_checks_pass']
    with zipfile.ZipFile(run/'sources.zip') as z:
        for name,digest in manifest['source_sha256'].items():
            assert sha(ROOT/name)==digest
            assert hashlib.sha256(z.read(name)).hexdigest()==digest
    for name,digest in manifest['artifacts_sha256'].items():assert sha(run/name)==digest
    allrows=[];inputs={};offsetbase=0;sourcechecks=[]
    for item in manifest['models']:
        scan=item['scan'];model=Path(item['model'])
        assert sha(model/'point_cloud/iteration_30000/point_cloud.ply')==item['ply_sha256']
        old=json.loads((ROOT/f'research/results/median_cpu_scan{scan}.json').read_text(encoding='utf-8'))
        report=json.loads((run/f'scan{scan}.json').read_text(encoding='utf-8'))
        npz=np.load(run/f'scan{scan}_rays.npz',allow_pickle=False)
        offsets=npz['offsets'];n=len(report['rows']);assert n==1024 and len(offsets)==n+1
        assert sha(run/f'scan{scan}_rays.npz')==report['input_npz_sha256']
        assert offsets[0]==0 and np.all(np.diff(offsets)>=0) and offsets[-1]==len(npz['alpha'])
        for i,(row,previous) in enumerate(zip(report['rows'],old['rows'])):
            lo,hi=offsets[i:i+2];a=npz['alpha'][lo:hi];p=npz['peaks'][lo:hi]
            assert row['camera']==previous['camera']==str(npz['camera'][i])
            assert row['pixel']==previous['pixel']==npz['pixels'][i].tolist()
            assert len(a)==row['contributors']==previous['contributors']
            assert bool(np.exp(np.log1p(-a).sum())<QL)==row['finite_band']
            assert not row['failed_checks']
            if row.get('band_computed'):
                assert row['old_median_depth_error']<=1e-8 and row['band_position_sum_error']<=1e-7
                order=np.argsort(p,kind='stable');C=np.r_[1.,np.cumprod(1-a[order])]
                w=np.maximum(0,np.minimum(QU,C[:-1])-np.maximum(QL,C[1:]))/(QU-QL)
                assert abs(max(0.,1-max(w))-row['hypothetical_thin_mixed_mass'])<1e-12
            if 'independent_validation' in row:
                v=row['independent_validation']
                assert v['absolute_forward_error']<=1e-8 and max(v['opacity_scaled_error'],v['position_scaled_error'])<=1e-4
            inputs[offsetbase+i]=(npz,lo,hi,scan,i)
        allrows+=report['rows'];offsetbase+=n
        sourcechecks.append({'scan':scan,'npz_rows_checked':n,'summary_recomputed':summarize(report['rows'])=={k:report[k] for k in summarize(report['rows'])}})
        assert sourcechecks[-1]['summary_recomputed']
    recomputed=summarize(allrows)
    assert recomputed=={k:summary[k] for k in recomputed}
    finite=[i for i,r in enumerate(allrows) if r.get('band_computed')]
    selected=list(dict.fromkeys([max(finite,key=lambda i:allrows[i]['band_opacity_gradient_l2']),
                                 max(finite,key=lambda i:allrows[i]['opacity_gradient_norm_ratio'])]))
    extreme=[]
    for index in selected:
        row=allrows[index];data,lo,hi,scan,local=inputs[index]
        a,p,s=(data[k][lo:hi] for k in ('alpha','peaks','widths'))
        def scalar_forward(aa,pp):
            def T(t):
                v=np.sqrt(1-aa*np.exp(-.5*((t-pp)/s)**2))
                return float(np.prod(np.where(t<=pp,v,(1-aa)/v)))
            def root(q):
                lo,hi=certified_bracket(aa,pp,s,q=q)
                return brentq(lambda t:T(t)-q,lo,hi,xtol=1e-12,rtol=1e-14)
            lo,hi=root(QU),root(QL)
            points=sorted(set([lo,hi]+[float(x) for x in pp if lo<x<hi]))
            integrals=[quad(lambda t:T(t)-QL,x,y,epsabs=1e-13,epsrel=1e-12,limit=500)[0]
                       for x,y in zip(points[:-1],points[1:])]
            return lo+sum(integrals)/(QU-QL)
        independent_mean=scalar_forward(a,p)
        fda=[];fdp=[];opacity_steps=[];position_step=1e-5*np.median(s)
        for j in range(len(a)):
            step=min(1e-6,a[j]/10,(1-a[j])/10)
            ap,am=a.copy(),a.copy();ap[j]+=step;am[j]-=step
            pp,pm=p.copy(),p.copy();pp[j]+=position_step;pm[j]-=position_step
            fda.append((scalar_forward(ap,p)-scalar_forward(am,p))/(2*step))
            fdp.append((scalar_forward(a,pp)-scalar_forward(a,pm))/(2*position_step))
            opacity_steps.append(float(step))
        mean,gp,ga,_,_,_=band(a,p,s)
        ep=np.linalg.norm(np.asarray(fdp)-gp)/max(np.linalg.norm(gp),1e-8)
        ea=np.linalg.norm(np.asarray(fda)-ga)/max(np.linalg.norm(ga),1e-8)
        record={'global_row':index,'scan':scan,'local_row':local,'camera':row['camera'],'pixel':row['pixel'],
                'band_opacity_norm':row['band_opacity_gradient_l2'],'opacity_norm_ratio':row['opacity_gradient_norm_ratio'],
                'independent_scalar_mean':float(independent_mean),'spatial_mean':float(mean),'saved_mean':row['band_depth'],
                'position_vjp_relative_error':float(ep),'opacity_vjp_relative_error':float(ea),
                'opacity_fd':fda,'position_fd':fdp,'opacity_analytic':ga.tolist(),'position_analytic':gp.tolist(),
                'opacity_steps':opacity_steps,'position_step':float(position_step)}
        extreme.append(record)
        progress=ROOT/'research/results/quantile_band_frozen_report_v2_progress.json'
        progress.write_text(json.dumps(extreme,indent=2)+'\n',encoding='utf-8')
        print('extreme row '+str(index)+' checked: '+str((float(ep),float(ea))),flush=True)
        if max(abs(independent_mean-mean),abs(mean-row['band_depth']))>1e-8 or max(ep,ea)>1e-5:
            raise AssertionError('Exploratory extreme oracle disagreement: '+json.dumps(record))
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'report_source_sha256':sha(Path(__file__)),
            'protocol_sha256':sha(ROOT/'research/QUANTILE_BAND_OUTLIER_V2_PROTOCOL.md'),
            'run_manifest_sha256':sha(run/'manifest.json'),'source_bytes_and_archive_verified':len(manifest['source_sha256']),
            'artifact_bytes_verified':len(manifest['artifacts_sha256']),'scene_checks':sourcechecks,'summary_recomputed':recomputed,
            'exploratory_selected_outlier_checks':extreme,'source_input_summary_and_forward_fd_checks_pass':True,
            'original_q_space_full_vjp_status':'failed_nonconverged_v1; source/protocol/log preserved in quantile_band_oracle_failure_v1.zip',
            'limits':['Outliers selected after observing results; numerical audit, not confirmatory estimation.',
                      'Frozen affine diagnostic has no real-scene quality or production-gradient claim.',
                      'Development replacement gate failed and is unchanged.',
                      'v2 forward FD success does not fix or pass the original q-space full-VJP oracle.']}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'sources':len(manifest['source_sha256']),'rays':len(allrows),'extreme_checks':extreme,
                      'all_independent_checks_pass':True},indent=2))


if __name__=='__main__':main()
