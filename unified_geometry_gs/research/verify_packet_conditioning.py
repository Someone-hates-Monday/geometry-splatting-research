"""Posthoc exact-formula perturbation checks; never changes an earlier gate.

Independent mpmath decoding is used for saved binary packet inputs. A separate
interval calculation certifies a two-packet quantization witness. Finite-error
bound screens are high-precision numerical checks, not interval certificates.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, platform
from datetime import datetime, timezone
from pathlib import Path
import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('plane', 'width', 'conic', 'qmean', 'qbar', 'qray', 'focal', 'amplitude')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def scalar(x, ctx): return ctx.mpf(float(x))
def norm(x, ctx): return ctx.sqrt(sum(v*v for v in x))
def dot(a, b): return sum(x*y for x, y in zip(a, b))
def matvec(k, x): return [k[0]*x[0]+k[1]*x[1], k[1]*x[0]+k[2]*x[1]]
def matrix_norm(k, ctx): return ctx.sqrt(k[0]**2 + 2*k[1]**2 + k[2]**2)
def difference(a, b): return [x-y for x, y in zip(a,b)]

def decode(packet, ctx=mp.mp):
    # Independent expression graph: no research.clipped_packet import.
    v = {k: [scalar(t,ctx) for t in np.asarray(packet[k]).ravel()] for k in KEYS}
    plane, width, conic = v['plane'], v['width'][0], v['conic']
    q, bar, ray, focal, amplitude = v['qmean'],v['qbar'],v['qray'],v['focal'],v['amplitude'][0]
    z=plane[2]/norm(q,ctx); rb=dot(bar,bar); b=rb/width**2
    c=[(plane[j]*focal[j]*ctx.sqrt(rb)/z+bar[j])/rb for j in range(2)]
    k=[conic[0]*focal[0]**2/(z*z*b),conic[1]*focal[0]*focal[1]/(z*z*b),conic[2]*focal[1]**2/(z*z*b)]
    a=difference(ray[:2],bar[:2]); m=difference(q[:2],bar[:2])
    A=1+dot(c,a); B=1+dot(c,m); d=A*A+dot(a,matvec(k,a))
    C=A*B+dot(a,matvec(k,m)); M=B*B+dot(m,matvec(k,m))
    vgram=[A*m[j]-B*a[j] for j in range(2)]
    determinant=k[0]*k[2]-k[1]**2
    gram=determinant*(a[0]*m[1]-a[1]*m[0])**2+dot(vgram,matvec(k,vgram))
    E=z*z*b*gram/d; r=norm(ray,ctx); p=z*r*C/d; sigma=r/ctx.sqrt(b*d)
    alpha=amplitude*ctx.exp(-E/2)
    return dict(z=z,b=b,c=c,k=k,a=a,m=m,A=A,B=B,d=d,C=C,M=M,r=r,
                E=E,p=p,sigma=sigma,alpha=alpha,amplitude=amplitude,determinant=determinant)

def perturbation_bounds(x,y):
    # Euclidean/Frobenius upper bounds, all finite perturbations retained.
    ec=norm(difference(x['c'],y['c']),mp); ea=norm(difference(x['a'],y['a']),mp)
    em=norm(difference(x['m'],y['m']),mp); ek=matrix_norm(difference(x['k'],y['k']),mp)
    cn,an,mn,kn=norm(x['c'],mp),norm(x['a'],mp),norm(x['m'],mp),matrix_norm(x['k'],mp)
    eA=ec*an+cn*ea+ec*ea; eB=ec*mn+cn*em+ec*em
    eaa=ek*(an+ea)**2+kn*(2*an*ea+ea*ea)
    emm=ek*(mn+em)**2+kn*(2*mn*em+em*em)
    eam=ek*(an+ea)*(mn+em)+kn*(ea*mn+an*em+ea*em)
    ed=2*abs(x['A'])*eA+eA*eA+eaa
    eC=abs(x['A'])*eB+abs(x['B'])*eA+eA*eB+eam
    eM=2*abs(x['B'])*eB+eB*eB+emm
    lower=x['d']-ed
    if lower<=0: return {'conclusive':False,'reason':'den lower bound nonpositive'}
    ratio=x['C']/x['d']; eq=(eC+abs(ratio)*ed)/lower
    ez=abs(y['z']-x['z']); er=abs(y['r']-x['r']); eb=abs(y['b']-x['b'])
    es=abs(x['z'])*er+abs(x['r'])*ez+ez*er
    ep=abs(x['z']*x['r'])*eq+es*(abs(ratio)+eq)
    U=x['b']*x['d']; eU=abs(x['b'])*ed+x['d']*eb+eb*ed
    if U<=eU or x['r']<=er: return {'conclusive':False,'reason':'width interval not positive'}
    wlo=(x['r']-er)/mp.sqrt(U+eU); whi=(x['r']+er)/mp.sqrt(U-eU)
    ew=max(abs(wlo-x['sigma']),abs(whi-x['sigma']))
    elogw=max(abs(mp.log(wlo/x['sigma'])),abs(mp.log(whi/x['sigma'])))
    ebet=eM+(2*abs(x['C'])*eC+eC*eC+x['C']**2*ed/x['d'])/lower
    beta=x['E']/(x['z']**2*x['b'])
    scale=x['z']**2*x['b']; escale=(abs(x['b'])+eb)*(2*abs(x['z'])*ez+ez**2)+x['z']**2*eb
    eE=abs(scale)*ebet+escale*(abs(beta)+ebet)
    elo=max(mp.mpf('0'),x['E']-eE); ehi=x['E']+eE
    eamp=abs(x['amplitude']-y['amplitude'])
    alo=max(mp.mpf('0'),x['amplitude']-eamp)*mp.exp(-ehi/2)
    ahi=(x['amplitude']+eamp)*mp.exp(-elo/2)
    ealpha=max(abs(alo-x['alpha']),abs(ahi-x['alpha']))
    if ahi>=1: return {'conclusive':False,'reason':'opacity upper bound not below one'}
    hbound=ealpha/(1-ahi)+ahi*ep/(2*mp.sqrt(mp.e)*(1-ahi)*wlo)+ahi*elogw/(mp.e*(1-ahi))
    return dict(conclusive=True,peak=ep,width=ew,energy=eE,alpha=ealpha,
                log_transmittance=hbound,alpha_upper=ahi,den_relative=ed/x['d'],
                cross_relative=eC/abs(x['C']) if x['C'] else mp.inf)

def H(t,x):
    G=x['alpha']*mp.exp(-((t-x['p'])/x['sigma'])**2/2)
    near=-mp.log1p(-G)/2
    return near if t<=x['p'] else -mp.log1p(-x['alpha'])-near

def check_pair(packet):
    original={k:np.asarray(packet[k],dtype=np.float64) for k in KEYS}
    rounded={k:np.asarray(packet[k],dtype=np.float32).astype(np.float64) for k in KEYS}
    x,y=decode(original),decode(rounded); bound=perturbation_bounds(x,y)
    if not bound['conclusive']: return {'conclusive':False,'reason':bound['reason']}
    errors={k:abs(y[k]-x[k]) for k in ('p','sigma','E','alpha')}
    expected={'p':'peak','sigma':'width','E':'energy','alpha':'alpha'}
    checked={k:errors[k]<=bound[expected[k]]*(1+mp.mpf('1e-60'))+mp.mpf('1e-70')*max(abs(x[k]),mp.mpf('1e-6000')) for k in errors}
    # Several points spanning both moving peaks; the analytic theorem covers all t.
    probes=[x['p']+j*x['sigma'] for j in (-5,-1,0,1,5)]+[y['p']]
    herrors=[abs(H(t,y)-H(t,x)) for t in probes]
    hpass=all(e<=bound['log_transmittance']*(1+mp.mpf('1e-60'))+mp.mpf('1e-70')*max(x['alpha'],mp.mpf('1e-6000')) for e in herrors)
    return {'conclusive':True,'all_bound_checks_pass':all(checked.values()) and hpass,
            'component_checks':checked,'sampled_logT_checks_pass':hpass,
            'peak_error':mp.nstr(errors['p'],25),'peak_relative_error':mp.nstr(errors['p']/max(abs(x['p']),mp.mpf('1e-8')),25),
            'peak_bound':mp.nstr(bound['peak'],25),'peak_relative_bound':mp.nstr(bound['peak']/max(abs(x['p']),mp.mpf('1e-8')),25),
            'cross_relative_bound':mp.nstr(bound['cross_relative'],25),'den_relative_bound':mp.nstr(bound['den_relative'],25),
            'alpha_upper':mp.nstr(bound['alpha_upper'],25),'uniform_logT_bound':mp.nstr(bound['log_transmittance'],25)}

def symbolic():
    import sympy as s
    d,dd,C,dC,z,dz,r,dr=s.symbols('d dd C dC z dz r dr',nonzero=True,real=True)
    t=s.symbols('t',real=True)
    checks={'quotient_finite_delta':s.cancel((C+dC)/(d+dd)-C/d-(d*dC-C*dd)/(d*(d+dd)))==0}
    checks['peak_differential']=s.cancel(s.diff((z+t*dz)*(r+t*dr)*(C+t*dC)/(d+t*dd),t).subs(t,0)-z*r/d*(dC+C*(dz/z+dr/r-dd/d)))==0
    u,v,du,dv=s.symbols('u v du dv',real=True)
    t0,cx,cy,kx,kxy,ky=s.symbols('t0 cx cy kx kxy ky',real=True,nonzero=True)
    zz=t0/s.sqrt(1+u*u+v*v); gamma=-(u*du+v*dv)/(1+u*u+v*v)
    checks['qmean_dz_relative']=s.simplify((du*s.diff(zz,u)+dv*s.diff(zz,v))/zz-gamma)==0
    cc=cx+cy/zz; KK=kx/(zz*zz)
    checks['qmean_dc']=s.simplify(du*s.diff(cc,u)+dv*s.diff(cc,v)+(cc-cx)*gamma)==0
    checks['qmean_dK']=s.simplify(du*s.diff(KK,u)+dv*s.diff(KK,v)+2*KK*gamma)==0
    return checks

def interval_endpoints(x): return mp.mpf(x._mpi_[0]),mp.mpf(x._mpi_[1])
def interval_text(x): return [mp.nstr(v,75) for v in interval_endpoints(x)]

def quantization_witness(row):
    """Construct two legal center-clipped SPD packets in one FP32 input cell."""
    packet={k:np.asarray(row[k],np.float32).astype(float) for k in ('plane','width','conic','qmean','qbar','qray')}
    packet.update(focal=np.array([800.,850.]),amplitude=np.array(.75))
    # Use the original exact binary clip limit .65, so both means have the same
    # legally clipped reference center; its FP32 value is unchanged.
    packet['qbar']=np.asarray(row['qbar'],float)
    qcenter=packet['qmean']; choices=[]
    for j in range(2):
        center=np.float32(qcenter[j]); neighbors=[np.nextafter(center,np.float32(-np.inf)),np.nextafter(center,np.float32(np.inf))]
        choices.append([float(center)+(float(n)-float(center))/2*(1-2**-12) for n in neighbors])
    candidates=[]
    for xy in itertools.product(*choices):
        item=dict(packet,qmean=np.r_[xy,1.]); candidates.append((decode(item)['p'],item))
    candidates.sort(key=lambda p:p[0]); pair=[candidates[0][1],candidates[-1][1]]
    identical=all(np.asarray(pair[0][k],np.float32).tobytes()==np.asarray(pair[1][k],np.float32).tobytes() for k in KEYS)
    clipping=all(np.array_equal(np.r_[np.clip(p['qmean'][:2],-.65,.65),1.],p['qbar']) for p in pair)
    mp.iv.dps=100; values=[decode(p,mp.iv) for p in pair]
    valid=all(interval_endpoints(v['z'])[0]>0 and interval_endpoints(v['b'])[0]>0 and interval_endpoints(v['k'][0])[0]>0 and interval_endpoints(v['determinant'])[0]>0 for v in values)
    lo0,hi0=interval_endpoints(values[0]['p']); lo1,hi1=interval_endpoints(values[1]['p'])
    # All acceptance endpoint arithmetic is interval-enclosed, including the
    # exact rational tolerance. Both peaks are certified below -1e-8, so the
    # relative branch of the original max(floor,abs(p)) rule applies.
    tau=mp.iv.mpf(1)/100000
    negative_relative_branch=hi0<-mp.mpf('1e-8') and hi1<-mp.mpf('1e-8')
    right0=values[0]['p']*(1-tau); left1=values[1]['p']*(1+tau)
    acceptance0_upper=interval_endpoints(right0)[1]
    acceptance1_lower=interval_endpoints(left1)[0]
    separation=interval_endpoints(left1-right0)[0]
    return {'case_source':row['case'],'posthoc':True,'rounding_mode':'numpy IEEE binary32 round-to-nearest ties-to-even',
       'same_fp32_input_bytes':identical,'qbar_is_clip_of_both_qmeans':clipping,'interval_spd_and_positive_scale':valid,
       'peak_intervals':[interval_text(v['p']) for v in values],
       'peak_interval_exact_binary_endpoints':[[[int(k) for k in t] for t in v['p']._mpi_] for v in values],
       'acceptance0_upper':mp.nstr(acceptance0_upper,75),'acceptance1_lower':mp.nstr(acceptance1_lower,75),
       'positive_separation_lower':mp.nstr(separation,75),
       'positive_separation_lower_exact_binary':[int(k) for k in separation._mpf_],
       'negative_relative_tolerance_branch_certified':bool(negative_relative_branch),
       'disjoint_original_tolerance_intervals':bool(acceptance0_upper<acceptance1_lower),
       'certified_two_packet_witness':bool(identical and clipping and valid and negative_relative_branch and separation>0 and acceptance0_upper<acceptance1_lower),
       'packets':[{k:np.asarray(v[k]).tolist() for k in KEYS} for v in pair],
       'limits':['Interval results use mpmath.iv 1.3 experimental arithmetic; implementation trust is explicit.',
         'The pair defines two legal SPD Gaussian packets via exact Schur inverse, not two independently observed real scene primitives.',
         'Only the finite peak tolerance for the full all-FP32 packet representation is ruled out uniformly on this domain.',
         'This does not rule out added metadata, relevance-aware criteria, better preprocessing, or useful downstream accuracy.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--include-real',action='store_true');a=p.parse_args()
    if a.out.exists(): raise ValueError('Fresh output required')
    mp.mp.dps=100
    theory=ROOT/'research/results/clipped_packet_theory_v1.json'; rows=read(theory)['synthetic_cases']; sources=[Path(__file__),ROOT/'research/PACKET_CONDITIONING_THEORY.md',theory]
    synth=[]
    for row in rows:
        packet={k:np.asarray(row[k]) for k in ('plane','width','conic','qmean','qbar','qray')}
        packet.update(focal=np.array([800.,850.]),amplitude=np.array(.75))
        synth.append(dict(case=row['case'],**check_pair(packet)))
    real=[]
    if a.include_real:
        folder=ROOT/'research/experiments/clipped_packet_v1'
        for i in range(24):
            source=folder/f'packet_{i:02d}.npz';sources.append(source)
            with np.load(source,allow_pickle=False) as data:
                for j in range(len(data['selected_ids'])):
                    packet={k:(data[k] if k=='focal' else data[k][j]) for k in KEYS}
                    real.append(dict(point=i,id=int(data['selected_ids'][j]),**check_pair(packet)))
            print(f'point {i+1}/24: {len(real)} real input finite bound checks',flush=True)
    checks=symbolic(); witness=quantization_witness(rows[2])
    allrows=synth+real
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','posthoc':True,'platform':platform.platform(),
      'mpmath':mp.__version__,'numpy':np.__version__,'mp_decimal_precision':100,'symbolic_checks':checks,
      'synthetic_input_count':len(synth),'real_input_count':len(real),'conclusive_bounds':sum(r['conclusive'] for r in allrows),
      'inconclusive_bound_inputs':sum(not r['conclusive'] for r in allrows),
      'bound_check_failures':sum(r['conclusive'] and not r['all_bound_checks_pass'] for r in allrows),
      'two_packet_quantization_witness':witness,'synthetic_rows':synth,'real_rows':real,
      'source_sha256':{str(s.relative_to(ROOT)):sha(s) for s in sources},
      'limits':['Finite perturbation bound screens are 100-digit numerical checks, not interval certificates.',
        'Only the explicitly labeled two-packet witness has interval-enclosed peak and SPD calculations.',
        'No actual production FP32 preprocessing, CUDA, full-scene median, image quality, performance or novelty claim.',
        'All original failing experiments and thresholds remain unchanged.']}
    result['verification_pass']=all(checks.values()) and result['bound_check_failures']==0 and witness['certified_two_packet_witness']
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('synthetic_input_count','real_input_count','conclusive_bounds','inconclusive_bound_inputs','bound_check_failures','symbolic_checks','verification_pass')},indent=2),flush=True)
    if not result['verification_pass']:raise ArithmeticError('Preserved verification failure')

if __name__=='__main__':main()
