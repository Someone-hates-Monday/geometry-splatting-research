"""Fixed 2^3 saved-coefficient root combinations; CPU-only posthoc attribution.

mask0=factor, mask7=packet; bit0=alpha, bit1=peak, bit2=rho.
All states are retained. Six-order Shapley means are algebraic allocations,
not causal/statistical contributions. Old oracle and all old results are read-only.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, time, traceback, zipfile
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import mpmath as mp
import numpy as np
from research.exact_outlier_root_oracle import Coefficients, ROOT, sha, verify_capture

NAMES=('alpha','peak','rho')
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def number(x):return mp.nstr(x,65)
def save(p,value):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)

def attribution(values,zero=None):
    if zero is None:zero=mp.mpf(0)
    phi=[zero,zero,zero];orders=[]
    for permutation in itertools.permutations(range(3)):
        mask=0;changes=[]
        for j in permutation:
            nextmask=mask|(1<<j);delta=values[nextmask]-values[mask]
            phi[j]+=delta;changes.append((j,mask,nextmask,delta));mask=nextmask
        orders.append((permutation,changes))
    phi=[x/6 for x in phi]
    interactions={}
    for mask in range(1,8):
        value=zero
        for submask in range(8):
            if submask&mask!=submask:continue
            value+=(-1)**(mask.bit_count()-submask.bit_count())*values[submask]
        interactions[mask]=value
    from_interactions=[sum((v/mask.bit_count() for mask,v in interactions.items() if mask&(1<<j)),zero) for j in range(3)]
    return phi,orders,interactions,from_interactions

def allocation_selftest():
    values=[]
    for mask in range(8):
        a,p,r=[Fraction(bool(mask&(1<<j))) for j in range(3)]
        values.append(2+3*a-4*p+Fraction(1,5)*r+5*a*p+Fraction(7,10)*a*r-Fraction(3,5)*p*r+2*a*p*r)
    phi,orders,interactions,phi2=attribution(values,Fraction(0))
    expected=[3+Fraction(57,20)+Fraction(2,3),-4+Fraction(11,5)+Fraction(2,3),Fraction(1,5)+Fraction(1,20)+Fraction(2,3)]
    assert phi==expected and phi==phi2 and sum(phi)==values[7]-values[0]
    assert interactions[3]==5 and interactions[5]==Fraction(7,10) and interactions[6]==-Fraction(3,5) and interactions[7]==2
    return {'pass':True,'arithmetic':'exact Fraction','known_polynomial_shapley':list(map(str,phi))}

def root_model(coeff):
    mass=coeff.mass();double=coeff.root64();root=coeff.root()
    result={'exact_mass':mass,'stable_double':double,'mp120':root}
    if root['status']=='finite':
        with mp.workdps(180):
            lo,hi=map(mp.mpf,root['numeric_bracket']);left,right=coeff.residual(lo),coeff.residual(hi)
            result['endpoint_sign_check_180']={'dps':180,'left':number(left),'right':number(right),'strict_signs_pass':bool(left<0<right)}
        if double['status']=='finite':result['stable_double_absolute_error_against_mp']=number(abs(mp.mpf(double['depth'])-mp.mpf(root['depth'])))
    return result

def compare_endpoint(model,old):
    new=model['mp120']
    if new['status']!=old['status']:return {'pass':False,'reason':'root status differs','old':old['status'],'new':new['status']}
    if new['status']!='finite':return {'pass':True,'reason':'both retain same nonfinite/unresolved status'}
    error=abs(mp.mpf(new['depth'])-mp.mpf(old['depth']))
    allowance=(mp.mpf(new['numeric_bracket_width'])+mp.mpf(old['numeric_bracket_width']))/2+mp.mpf('1e-40')
    return {'pass':bool(error<=allowance),'absolute_error':number(error),'numeric_bracket_allowance':number(allowance),
            'exact_saved_root_string_equal':new['depth']==old['depth']}

def case_analysis(i,data,oldrow,out):
    old={a['arm']:a for a in oldrow['arms']};assert set(old)=={'packet','factor'}
    ids=data['primitive_ids'][data['accepted_mask']]
    assert np.array_equal(ids,data['packet_selected_ids']) and np.array_equal(ids,data['factor_selected_ids'])
    cutoff=int(data['actual_n_contrib'][0]);assert int(data['copied_initialization'][4])==cutoff
    assert np.array_equal(data['accepted_mask'],(data['rank_one_based']<=cutoff)&(data['copied_coefficients_all'][:,2]==1))
    source_arrays={}
    for arm,columns in (('packet',(7,5,6)),('factor',(11,9,10))):
        for field,col in zip(NAMES,columns):
            array=np.asarray(data[arm+'_'+field]);assert array.dtype==np.float32
            assert array.tobytes()==data['copied_coefficients_all'][data['accepted_mask'],col].tobytes()
            source_arrays[arm+'_'+field]=array
    models=[];arrays=[]
    for mask in range(8):
        fields=[source_arrays[('packet' if mask&(1<<j) else 'factor')+'_'+field] for j,field in enumerate(NAMES)]
        arrays.append(np.column_stack(fields));result=root_model(Coefficients(*fields))
        result.update(mask=mask,packet_fields=[NAMES[j] for j in range(3) if mask&(1<<j)])
        models.append(result)
        save(out/'progress.json',{'utc':now(),'case':i,'mask':mask,'completed_models':i*8+mask+1,'total_models':72,'root_status':result['mp120']['status']})
        print(f'case {i}/8 mask {mask}/7 ({i*8+mask+1}/72): {result["mp120"]["status"]}',flush=True)
    endpoint_checks={'mask0_factor':compare_endpoint(models[0],old['factor']['minus_infinity_root']),
                     'mask7_packet':compare_endpoint(models[7],old['packet']['minus_infinity_root'])}
    result={'case':i,'accepted_inputs':len(ids),'actual_n_contrib':cutoff,'rln':float(data['copied_initialization'][5]),
            'models':models,'endpoint_oracle_checks':endpoint_checks}
    np.savez_compressed(out/f'case_{i:02d}.npz',selected_ids=ids,mask=np.arange(8,dtype=np.uint8),
        alpha_peak_rho=np.stack(arrays),rln=data['copied_initialization'][5:6])
    if not all(x['pass'] for x in endpoint_checks.values()):
        result['interpretation_status']='stopped_endpoint_oracle_mismatch';save(out/f'case_{i:02d}.json',result)
        raise ArithmeticError('Mask endpoint differs from prior independent oracle; interpretation stopped')
    if not all(m['mp120']['status']=='finite' and m['endpoint_sign_check_180']['strict_signs_pass'] for m in models):
        result['interpretation_status']='undefined_nonfinite_or_unresolved_model';save(out/f'case_{i:02d}.json',result)
        return result
    values=[mp.mpf(m['mp120']['depth']) for m in models];phi,orders,interactions,phi2=attribution(values)
    total=values[7]-values[0];sumabs=mp.fsum(abs(x) for x in phi)
    sum_error=abs(mp.fsum(phi)-total);interaction_error=max(abs(a-b) for a,b in zip(phi,phi2))
    assert sum_error<mp.mpf('1e-100') and interaction_error<mp.mpf('1e-100')
    rln=mp.mpf(float(data['copied_initialization'][5]))
    labels={mask:'+'.join(NAMES[j] for j in range(3) if mask&(1<<j)) for mask in range(1,8)}
    result.update(interpretation_status='finite_algebraic_allocation',
        deltas_from_factor_ray={str(mask):number(v-values[0]) for mask,v in enumerate(values)},
        deltas_from_factor_conditional_z={str(mask):number((v-values[0])*rln) for mask,v in enumerate(values)},
        shapley_mean_ray={name:number(value) for name,value in zip(NAMES,phi)},
        shapley_mean_conditional_z={name:number(value*rln) for name,value in zip(NAMES,phi)},
        shapley_sum_error=number(sum_error),shapley_vs_mobius_error=number(interaction_error),
        interactions_ray={labels[mask]:number(value) for mask,value in interactions.items()},
        six_orders=[{'order':[NAMES[j] for j in permutation],
            'steps':[{'field':NAMES[j],'from_mask':a,'to_mask':b,'delta_ray':number(v)} for j,a,b,v in changes]}
            for permutation,changes in orders],
        cancellation_fraction=number(1-abs(total)/sumabs) if sumabs else '0.0',
        allocation_directions={name:('same_as_total' if value*total>0 else 'opposes_total' if value*total<0 else 'zero') for name,value in zip(NAMES,phi)},
        alpha_when_geometry_factor_ray=number(values[1]-values[0]),
        alpha_when_geometry_packet_ray=number(values[7]-values[6]),
        peak_rho_joint_without_alpha_ray=number(values[6]-values[0]),
        alpha_geometry_interaction_ray=number(values[7]-values[6]-values[1]+values[0]),
        root_numeric_half_bracket_bounds=[number(mp.mpf(m['mp120']['numeric_bracket_width'])/2) for m in models],
        limits=['Finite root estimates and brackets are numerical evidence, not interval certificates.',
                'Six-order averages and Moebius interactions allocate a finite function difference, not causal/statistical effects.',
                'Conditional z units use copied probe rln and inherit its scope; no new output rounding is applied.'])
    save(out/f'case_{i:02d}.json',result)
    return result

def extrema(rows,key,subkeys=None):
    candidates=[]
    for row in rows:
        if row['interpretation_status']!='finite_algebraic_allocation':continue
        value=row[key]
        for name in subkeys or (None,):
            v=mp.mpf(value[name] if name is not None else value)
            candidates.append((abs(v),row['case'],name,v))
    if not candidates:return None
    absolute,case,name,value=max(candidates)
    return {'absolute_value':number(absolute),'signed_value':number(value),'case':case,'field':name}

def execute(args,out,manifest,files,old):
    start=time.perf_counter();rows=[]
    for i,path in enumerate(files):
        assert path.name==f'ray_{i:02d}.npz'
        assert old['rows'][i]['input_sha256']==sha(path)
        with np.load(path,allow_pickle=False) as data:row=case_analysis(i,data,old['rows'][i],out)
        identity=read(path.with_suffix('.json'));row['identity']={k:identity[k] for k in ('scan','camera','pixel')}
        save(out/f'case_{i:02d}.json',row);rows.append(row)
    assert sum(r['accepted_inputs'] for r in rows)==772
    status_counts=Counter(m['mp120']['status'] for r in rows for m in r['models'])
    errors=[(mp.mpf(m['stable_double_absolute_error_against_mp']),r['case'],m['mask']) for r in rows for m in r['models'] if 'stable_double_absolute_error_against_mp' in m]
    maxerr=max(errors) if errors else None
    report={'created_utc':now(),'status':'complete','posthoc':True,'rays':9,'models':72,'accepted_inputs':772,
      'gaussian_entries_across_mixed_models':772*8,'mask_definition':{'baseline':'factor mask0','endpoint':'packet mask7','bits':dict(enumerate(NAMES))},
      'mp_dps':120,'endpoint_dps':180,'root_status_counts':dict(status_counts),
      'stable_double_status_counts':dict(Counter(m['stable_double']['status'] for r in rows for m in r['models'])),
      'endpoint_oracle_matches':sum(x['pass'] for r in rows for x in r['endpoint_oracle_checks'].values()),
      'endpoint_sign_checks_pass':sum(m.get('endpoint_sign_check_180',{}).get('strict_signs_pass',False) for r in rows for m in r['models']),
      'interpretable_rays':sum(r['interpretation_status']=='finite_algebraic_allocation' for r in rows),
      'stable_double_max_absolute_error_against_mp':None if maxerr is None else {'error':number(maxerr[0]),'case':maxerr[1],'mask':maxerr[2]},
      'largest_total_change_ray':extrema(rows,'deltas_from_factor_ray',('7',)),
      'largest_alpha_only_change_ray':extrema(rows,'alpha_when_geometry_factor_ray'),
      'largest_peak_rho_joint_without_alpha_ray':extrema(rows,'peak_rho_joint_without_alpha_ray'),
      'largest_alpha_geometry_interaction_ray':extrema(rows,'alpha_geometry_interaction_ray'),
      'largest_shapley_component_ray':extrema(rows,'shapley_mean_ray',NAMES),
      'largest_cancellation_fraction':extrema(rows,'cancellation_fraction'),
      'rows':rows,'wall_seconds':time.perf_counter()-start,
      'limits':['All nine inputs were selected posthoc from prior frozen-image discrepancies; no generalization/prevalence claim.',
        'This evaluates 72 fixed saved-FP32 coefficient models, not raw-world/GT surfaces or a new rendering implementation.',
        'Changing saved alpha while keeping saved p/rho defines a mathematical hybrid, not necessarily a realizable common 3D Gaussian.',
        'Six-order Shapley averages and interactions are finite algebraic allocations, not causal/statistical contribution rates.',
        'MP roots and 180-digit endpoint signs are numerical evidence, not outward interval/formal certificates.',
        'No CUDA compilation, GPU work, training, backward, timing benefit, quality gain or novelty claim.']}
    save(out/'report.json',report);save(args.report,report)
    save(out/'summary.json',{k:v for k,v in report.items() if k!='rows'})
    manifest.update(status='complete',completed_utc=now(),wall_seconds=report['wall_seconds'],
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','progress.json')},
        external_report_path=str(args.report.resolve()),external_report_sha256=sha(args.report))
    save(out/'manifest.json',manifest)
    print(json.dumps({k:report[k] for k in ('status','rays','models','root_status_counts','endpoint_oracle_matches','endpoint_sign_checks_pass','interpretable_rays','stable_double_max_absolute_error_against_mp')},indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--oracle',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=ROOT/'research/results/exact_outlier_mix_report_v1.json');args=parser.parse_args()
    args.capture=args.capture.resolve();out=args.out.resolve()
    if out.exists() or args.report.exists():raise ValueError('Fresh run and external report required')
    mp.mp.dps=120;selftest=allocation_selftest();files,capture_checks=verify_capture(args.capture)
    old=read(args.oracle);assert old['status']=='complete' and old['inputs']==9 and old['analyzed_exact_arms']==18
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest
    assert old['capture_checks']['manifest_sha256']==sha(args.capture/'manifest.json')
    sources={Path(__file__),ROOT/'research/EXACT_OUTLIER_COEFFICIENT_PROTOCOL.md',args.oracle.resolve(),args.capture/'manifest.json'}
    sources.update(ROOT/name for name in old['source_sha256'])
    for path in files:sources.update((path,path.with_suffix('.json')))
    out.mkdir(parents=True);manifest={'created_utc':now(),'status':'running','pid':os.getpid(),'CPU_only':True,
      'mpmath':mp.__version__,'numpy':np.__version__,'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
      'allocation_exact_fraction_selftest':selftest,'capture_checks':capture_checks,
      'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)}}
    save(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):archive.writestr(path.relative_to(ROOT).as_posix(),path.read_bytes())
    try:execute(args,out,manifest,files,old)
    except BaseException as exc:
        manifest.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc));save(out/'manifest.json',manifest)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
