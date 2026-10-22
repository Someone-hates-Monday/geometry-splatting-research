"""Trusted interval witness: legal F32 worlds alias after correct F32 ray rounding."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
import argparse,hashlib,json,zipfile
import numpy as np
import mpmath as mp

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def frac(v):return F(float(v))


def iv_fraction(v):return mp.iv.mpf(v.numerator)/mp.iv.mpf(v.denominator)


def endpoints(x):return [mp.mpf(v) for v in x._mpi_]


def interval_record(x):
    return {'display_bounds':[mp.nstr(v,90) for v in endpoints(x)],
            'exact_binary_mpf_endpoints':[[int(n) for n in v] for v in x._mpi_]}


def rounding_cell(value):
    x=np.float32(value);prev=np.nextafter(x,np.float32(-np.inf));nxt=np.nextafter(x,np.float32(np.inf))
    return (frac(prev)+frac(x))/2,(frac(x)+frac(nxt))/2


def certificate(epsilon,ys,coefficients,dps):
    mp.mp.dps=dps+30;mp.iv.dps=dps
    eps=iv_fraction(epsilon);half=iv_fraction(F(1,2));low_cell,hi_cell=rounding_cell(.5)
    a=F(64,125);b=F(186,125);roots=[];alphas=[]
    for name,y,t in (('A',ys[0],a),('B',ys[1],b)):
        alpha=iv_fraction(F(3,4))*mp.iv.exp(-(iv_fraction(y)/eps)**2/2)
        l,h=endpoints(alpha)
        assert l>mp.mpf(low_cell.numerator)/low_cell.denominator
        assert h<mp.mpf(hi_cell.numerator)/hi_cell.denominator
        if name=='A':assert l>mp.mpf('.5')
        else:assert h<mp.mpf('.5')
        alpha2=half;g1=alpha*mp.iv.exp(-((iv_fraction(t)-half)/eps)**2/2)
        g2=alpha2*mp.iv.exp(-((iv_fraction(t)-iv_fraction(F(3,2)))/eps)**2/2)
        # t lies strictly between both peaks.
        T=(1-alpha)*mp.iv.sqrt((1-g2)/(1-g1))
        tl,th=endpoints(T)
        if name=='A':assert th<mp.mpf('.5')
        else:assert tl>mp.mpf('.5')
        alphas.append(interval_record(alpha));roots.append({'world':name,'evaluation_t':str(t),'T_interval':interval_record(T),
                    'inferred_root_side':'root_A < 64/125' if name=='A' else 'root_B > 186/125'})
    # Widths/directions are identical; p is exactly the z position, rho=1/eps.
    rho=F(1)/epsilon;rho_value=coefficients[0,1];lc,hc=rounding_cell(rho_value)
    assert lc<rho<hc
    assert np.array_equal(coefficients[:,0],np.array([.5,1.5],np.float32))
    assert coefficients[0,1]==coefficients[1,1]
    assert np.array_equal(coefficients[:,2],np.array([.5,.5],np.float32))
    return {'iv_dps':dps,'correct_round_alpha_to_point5_for_both_worlds':True,'alpha_intervals':alphas,
            'correct_round_rho_exact_rational':str(rho),'rho_rounding_cell':[str(lc),str(hc)],
            'root_localization_checks':roots,'root_separation_strict_lower_bound':str(b-a),
            'any_single_output_uniform_abs_error_strict_lower_bound':str((b-a)/2)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    run=ROOT/'research/experiments/coherent_factor_v2';manifest=read(run/'manifest.json');summary=read(run/'summary.json')
    assert manifest['status']==summary['status']=='complete'
    counts=dict(sources=0,archive_sources=0,artifacts=0,coefficient_error_vectors=0,root_error_checks=0)
    with zipfile.ZipFile(run/'sources.zip') as z:
        for name,h in manifest['source_sha256'].items():
            assert sha(ROOT/name)==h and hashlib.sha256(z.read(name)).hexdigest()==h
            counts['sources']+=1;counts['archive_sources']+=1
    for name,h in manifest['artifacts_sha256'].items():assert sha(run/name)==h;counts['artifacts']+=1
    for group in summary['groups']:
        with np.load(run/(group['group']+'.npz')) as d:
            target=d['reference']
            for record in group['variants']:
                v=record['variant'];c=d[v+'_F32_peak_rho_alpha'];actual=np.column_stack((c[:,0],1/c[:,1].astype(float),c[:,2]))
                errors=np.column_stack((abs(actual[:,0]-target[:,0])/np.maximum(1e-8,abs(target[:,0])),abs(actual[:,1]-target[:,1])/np.maximum(1e-8,abs(target[:,1])),abs(actual[:,2]-target[:,2])))
                assert np.array_equal(errors,d[v+'_errors'])
                bad=np.any(errors>np.array([1e-5,1e-4,1e-4]),axis=1)|np.any(~np.isfinite(errors),axis=1)
                assert np.array_equal(bad,d[v+'_bad']) and int(sum(bad))==record['bad']
                counts['coefficient_error_vectors']+=len(c)
    mp.mp.dps=180
    for row in summary['root_rows']:
        ref=mp.mpf(row['reference_root']['camera_ray_depth'])
        for v,record in row['variants'].items():
            r=record['F32_model_root'];assert r['status']=='finite'
            error=abs(mp.mpf(r['depth'])-ref)
            assert abs(error-mp.mpf(record['F32_model_abs_error']))<mp.mpf('1e-70')
            assert bool(error<=mp.mpf('1e-3'))==record['F32_model_root_gate_pass'];counts['root_error_checks']+=1
    with np.load(run/'critical18.npz') as d:
        first=d['qr64_F32_peak_rho_alpha'][:2];second=d['qr64_F32_peak_rho_alpha'][2:4]
        assert np.array_equal(first.view(np.uint32),second.view(np.uint32))
        yA=d['qr64_mean'][0,1];yB=d['qr64_mean'][2,1]
        assert float(np.float32(yA))==yA and float(np.float32(yB))==yB and yA!=yB
        eps=frac(np.float32(.002));ys=[frac(yA),frac(yB)]
        checks=[certificate(eps,ys,first,precision) for precision in (60,120)]
        bits=first.view(np.uint32).tolist()
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','posthoc_existential_witness':True,'audit_counts':counts,
            'primary_development_gate_pass':summary['primary_development_gate_pass'],'root_failures':summary['root_failures'],
            'worlds':{'epsilon_exact_binary_rational':str(eps),'y_A_exact_binary_rational':str(ys[0]),'y_B_exact_binary_rational':str(ys[1]),
                      'centers_z_exact':[str(F(1,2)),str(F(3,2))],'amplitudes_exact':[str(F(3,4)),str(F(1,2))],
                      'rotation_camera_identity':True,'isotropic_positive_scales':True},
            'identical_F32_peak_rho_alpha_bits':bits,'trusted_interval_certificates':checks,
            'rounded_model_exact_root':'1: identical widths and alpha=.5 on peaks .5,1.5 imply exact symmetric GGGS median.',
            'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),'run_manifest':sha(run/'manifest.json'),'run_summary':sha(run/'summary.json')},
            'limits':['Bound applies to algorithms reading only correctly rounded F32 ray coefficient arrays; original world parameters retain additional information.',
                      'Worlds were selected after observing the fixed critical fixtures; this existential witness is not prevalence evidence.',
                      'Trusted mpmath.iv outward interval implementation is assumed correct; no proof-assistant validation.',
                      'Ray transmittance models only, not complete renderer eligibility, initial bracket, CUDA instruction identity, backward or GT quality.',
                      'General quantile instability and exact QR are known concepts; novelty remains unverified.']}
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'counts':counts,'uniform_abs_root_error_strict_lower_bound':'61/125 = .488','primary_pass':summary['primary_development_gate_pass']}))


if __name__=='__main__':main()
