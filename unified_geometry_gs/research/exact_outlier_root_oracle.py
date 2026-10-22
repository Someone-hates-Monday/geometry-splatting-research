"""Independent saved-FP32 GGGS root oracle and explicitly labelled CPU replay.

Canonical NPZ input: alpha/peak/rho float32 vectors; optionally initialization[9]
and depth_z/slope scalars. Multi-arm input uses affine_*, packet_*, factor_*.
Capture input: coefficients[N,13] plus accepted_mask[N], initialization[9],
and optional packet_depth_z/factor_depth_z etc. The mask must be supplied from
the actual prefix, not reconstructed by this oracle. No renderer modifications.
"""
from __future__ import annotations
import argparse,bisect,hashlib,json,math,zipfile
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HALF=Fraction(1,2)
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def textnumber(v):
    # Preserve binary32/double output values; np.float32 repr has too few digits.
    if isinstance(v,(float,np.floating)):v=mp.mpf(float(v))
    return mp.nstr(v,45)
def fraction_record(q):return {'numerator':str(q.numerator),'denominator':str(q.denominator)}
def finite_float(v):
    v=float(v)
    return v if math.isfinite(v) else str(v)

class Coefficients:
    def __init__(self,alpha,peak,rho):
        self.arrays=[np.asarray(v) for v in (alpha,peak,rho)]
        if any(v.dtype!=np.float32 or v.ndim!=1 for v in self.arrays):
            raise ValueError('Require original 1D float32 alpha/peak/rho, without silent casting')
        if not(len(alpha)==len(peak)==len(rho)) or not len(alpha):raise ValueError('Nonempty equal vectors required')
        if not all(np.isfinite(v).all() for v in self.arrays):raise ValueError('Nonfinite coefficient')
        if np.any((alpha<0)|(alpha>=1)) or np.any(rho<=0):raise ValueError('Require 0<=alpha<1 and rho>0')
        self.alpha,self.peak,self.rho=[[mp.mpf(float(x)) for x in v] for v in self.arrays]
        self.fractions=[Fraction(float(x)) for x in alpha]
        order=sorted(range(len(alpha)),key=lambda i:float(peak[i]))
        self.ordered_peaks=[float(peak[i]) for i in order];self.order=order
        self.prefix=[Fraction(1)]
        for i in order:self.prefix.append(self.prefix[-1]*(1-self.fractions[i]))
        self.terminal=self.prefix[-1]

    def residual(self,t):
        # H(t)-log2. Its plateau constant is formed by exact rational mass,
        # avoiding subtraction of two rounded O(1) log values.
        k=bisect.bisect_left(self.ordered_peaks,t)
        beta=self.prefix[k];delta=(HALF-beta)/beta
        constant=mp.log1p(mp.mpf(delta.numerator)/delta.denominator)
        tails=[]
        for a,p,r in zip(self.alpha,self.peak,self.rho):
            if not a:continue
            g=a*mp.exp(-((t-p)*r)**2/2)
            near=-mp.log1p(-g)/2
            tails.append(-near if t>p else near)
        return constant+mp.fsum(tails)

    def H(self,t):
        terms=[]
        for a,p,r in zip(self.alpha,self.peak,self.rho):
            g=a*mp.exp(-((t-p)*r)**2/2)
            near=-mp.log1p(-g)/2
            terms.append(near if t<=p else -mp.log1p(-a)-near)
        return mp.fsum(terms)

    def slopeH(self,t):
        return mp.fsum(a*mp.exp(-((t-p)*r)**2/2)*abs((t-p)*r)*r/
            (2*(1-a*mp.exp(-((t-p)*r)**2/2))) for a,p,r in zip(self.alpha,self.peak,self.rho))

    def mass(self):
        relation='less' if self.terminal<HALF else 'equal' if self.terminal==HALF else 'greater'
        return {'terminal_exact':fraction_record(self.terminal),'terminal_minus_half_exact':fraction_record(self.terminal-HALF),
          'terminal_float64':float(self.terminal),'relation_to_half':relation,
          'full_minus_infinity_finite_root':relation=='less',
          'classification':'finite' if relation=='less' else 'critical_asymptotic_no_finite_root' if relation=='equal' else 'insufficient_mass_no_finite_root',
          'nonzero_alpha_terms':sum(a>0 for a in self.alpha)}

    def root(self,camera_origin=False,tolerance='1e-32',max_iterations=480):
        mass=self.mass();H0=self.H(mp.mpf(0))
        if not camera_origin and not mass['full_minus_infinity_finite_root']:
            return {'status':mass['classification']}
        if camera_origin:
            remaining=-mp.log(mp.mpf(self.terminal.numerator)/self.terminal.denominator)-H0-mp.log(2)
            if abs(remaining)<mp.mpf(10)**(-mp.mp.dps//2):
                return {'status':'camera_origin_mass_nearcritical_unresolved','remaining_logmass_margin':textnumber(remaining)}
            if remaining<0:return {'status':'camera_origin_insufficient_mass','remaining_logmass_margin':textnumber(remaining)}
        f=(lambda t:self.residual(t)-H0) if camera_origin else self.residual
        reach=max(mp.mpf(1),10/min(self.rho))
        lo=mp.mpf(0) if camera_origin else min(self.peak)-reach
        hi=max(mp.mpf(1),max(self.peak)+reach)
        expansion=[]
        for _ in range(80):
            left_value=f(lo);expansion.append({'side':'left','t':textnumber(lo),'residual':textnumber(left_value)})
            if left_value<0:break
            if camera_origin:return {'status':'camera_origin_unexpected_left_sign'}
            lo-=reach;reach*=2
        else:return {'status':'left_bracket_unresolved'}
        reach=max(mp.mpf(1),hi-lo)
        for _ in range(80):
            right_value=f(hi);expansion.append({'side':'right','t':textnumber(hi),'residual':textnumber(right_value)})
            if right_value>0:break
            hi+=reach;reach*=2
        else:return {'status':'right_bracket_unresolved'}
        target=mp.mpf(tolerance);iterations=0
        while hi-lo>target and iterations<max_iterations:
            mid=(lo+hi)/2;value=f(mid);iterations+=1
            if value<0:lo=mid
            elif value>0:hi=mid
            else:
                # A rounded equality is not an exact root certificate.
                left=(lo+mid)/2;right=(mid+hi)/2
                if f(left)<0<f(right):lo,hi=left,right
                else:return {'status':'precision_plateau_unresolved','bracket':[textnumber(lo),textnumber(hi)],'iterations':iterations,'bracket_expansion':expansion}
        if hi-lo>target:return {'status':'iteration_limit','bracket':[textnumber(lo),textnumber(hi)],'bracket_expansion':expansion}
        value=(lo+hi)/2;R=f(value);slope=self.slopeH(value)
        return {'status':'finite','depth':textnumber(value),'numeric_bracket':[textnumber(lo),textnumber(hi)],
            'numeric_bracket_width':textnumber(hi-lo),'iterations':iterations,'root_log_residual':textnumber(R),
            'root_T_minus_half':textnumber(mp.expm1(-R)/2),'H_slope':textnumber(slope),'minus_T_slope':textnumber(mp.exp(-R)*slope/2),
            'regular_positive_slope':bool(slope>0),'camera_origin_conditioned':camera_origin,'bracket_expansion':expansion}

    def root64(self):
        if not self.mass()['full_minus_infinity_finite_root']:return {'status':self.mass()['classification']}
        p,r=np.asarray(self.arrays[1],float),np.asarray(self.arrays[2],float)
        reach=max(1.,10/float(min(r)));lo=float(min(p))-reach;hi=float(max(p))+reach;expansion=[]
        for side in ('left','right'):
            for _ in range(80):
                t=lo if side=='left' else hi;v=self.residual64(t)
                expansion.append({'side':side,'t':t,'residual':finite_float(v)})
                if (v<0 if side=='left' else v>0):break
                if side=='left':lo-=reach
                else:hi+=reach
                reach*=2
            else:return {'status':'double_bracket_unresolved','bracket_expansion':expansion}
        for iteration in range(240):
            if hi-lo<=1e-12:
                return {'status':'finite','depth':(lo+hi)/2,'numeric_bracket':[lo,hi],
                    'numeric_bracket_width':hi-lo,'iterations':iteration,'bracket_expansion':expansion}
            mid=(lo+hi)/2;v=self.residual64(mid)
            if v<0:lo=mid
            elif v>0:hi=mid
            else:
                left=(lo+mid)/2;right=(mid+hi)/2
                if self.residual64(left)<0<self.residual64(right):lo,hi=left,right
                else:return {'status':'double_precision_plateau_unresolved','numeric_bracket':[lo,hi],'bracket_expansion':expansion}
        return {'status':'double_iteration_limit','numeric_bracket':[lo,hi],'bracket_expansion':expansion}

    def residual64(self,t):
        k=bisect.bisect_left(self.ordered_peaks,float(t));beta=self.prefix[k];delta=(HALF-beta)/beta
        try:constant=math.log1p(float(delta))
        except OverflowError:constant=math.log((HALF/beta).numerator)-math.log((HALF/beta).denominator)
        terms=[]
        for a,p,r in zip(*self.arrays):
            g=float(a)*math.exp(-.5*((float(t)-float(p))*float(r))**2)
            near=-math.log1p(-g)/2
            terms.append(-near if t>p else near)
        return math.fsum([constant,*terms])

    def product(self,t,dtype):
        t=dtype(t);T=dtype(1)
        with np.errstate(under='ignore',invalid='ignore',divide='ignore'):
            for alpha,peak,rho in zip(*self.arrays):
                a,p,r=dtype(alpha),dtype(peak),dtype(rho)
                delta=dtype(dtype(t-p)*r);g=dtype(np.exp(dtype(-.5)*dtype(delta*delta)))
                vacancy=dtype(dtype(1)-dtype(a*g));rv=dtype(dtype(1)/np.sqrt(vacancy))
                numerator=dtype(dtype(1)-a) if t>p else vacancy
                T=dtype(T*dtype(numerator*rv))
        return T

def finite_search(coeff,initial,mode,split=8,rounds=5):
    """Same subdivision/last->=.5 choice; CPU FP32 is not CUDA instruction replay."""
    if mode=='mp':dtype=mp.mpf;evaluate=lambda t:coeff.residual(t)
    elif mode=='float64_log':dtype=np.float64;evaluate=lambda t:coeff.residual64(t)
    elif mode=='float64_product':dtype=np.float64;evaluate=lambda t:np.float64(.5)-coeff.product(t,np.float64)
    elif mode=='float32_product':dtype=np.float32;evaluate=lambda t:np.float32(.5)-coeff.product(t,np.float32)
    else:raise ValueError(mode)
    lo,hi=[dtype(float(v)) for v in initial[1:3]];gate=bool(initial[8])
    history=[];values=None
    for iteration in range(rounds):
        step=dtype(dtype(hi-lo)/dtype(split))
        if iteration==0:values=[evaluate(dtype(lo+dtype(step*dtype(j)))) for j in range(split+1)]
        else:
            for j in range(1,split):values[j]=evaluate(dtype(lo+dtype(step*dtype(j))))
        if iteration==0:gate=gate and values[0]<=0 and values[-1]>=0
        start=0
        for j in range(1,split):
            if values[j]<=0:start=j
        history.append({'iteration':iteration,'bracket_before':[finite_float(lo),finite_float(hi)],
          'left_residual':textnumber(values[0]),'right_residual':textnumber(values[-1]),'chosen_subinterval':start,
          'residual_convention':'half_minus_T' if mode.endswith('_product') else 'H_minus_log2',
          'zero_residual_gridpoints':int(sum(bool(v==0) for v in values)),'initial_endpoint_bracket_gate':bool(gate)})
        newhi=dtype(lo+dtype(dtype(start+1)*step));newlo=dtype(lo+dtype(dtype(start)*step))
        lo,hi=newlo,newhi;values[0],values[-1]=values[start],values[start+1]
    if not gate:return {'status':'invalid_initial_bracket','history':history}
    if mode.endswith('_product'):
        # values=q-T, so T_lo-q=-values[0] and T_lo-T_hi=values[-1]-values[0].
        numerator=dtype(-values[0]);denominator=dtype(values[-1]-values[0])
    else:
        expm1=mp.expm1 if mode=='mp' else math.expm1
        numerator=expm1(-values[0]);denominator=numerator-expm1(-values[-1])
    if denominator==0:
        return {'status':'rounded_flat_endpoint_values','final_bracket':[textnumber(lo),textnumber(hi)],
          'output_if_cuda_saturate_nan_returns_zero':textnumber(lo),'history':history}
    weight=dtype(numerator/denominator);weight=min(dtype(1),max(dtype(0),weight))
    depth=dtype(dtype(weight*hi)+dtype(dtype(1)-weight)*lo)
    return {'status':'finite','depth':textnumber(depth),'final_bracket':[textnumber(lo),textnumber(hi)],
            'final_left_residual':textnumber(values[0]),'final_right_residual':textnumber(values[-1]),'history':history}

def scalar(data,key,default=None):
    return float(np.asarray(data[key]).reshape(())) if key in data else default

def load(path,include_affine=False):
    cases=[]
    with np.load(path,allow_pickle=False) as data:
        initkey='copied_initialization' if 'copied_initialization' in data else 'initialization'
        initialization=np.asarray(data[initkey]) if initkey in data else None
        if initialization is not None and (initialization.shape!=(9,) or initialization.dtype!=np.float32):raise ValueError('Require captured FP32 initialization[9]')
        matrixkey='copied_coefficients_all' if 'copied_coefficients_all' in data else 'coefficients'
        if matrixkey in data:
            matrix=np.asarray(data[matrixkey]);mask=np.asarray(data['accepted_mask'],bool)
            if matrix.ndim!=2 or matrix.shape[1]!=13 or mask.shape!=(len(matrix),) or matrix.dtype!=np.float32:raise ValueError('Bad capture schema')
            if 'actual_n_contrib' in data:
                cutoff=int(np.asarray(data['actual_n_contrib']).reshape(()))
                rank=np.asarray(data['rank_one_based'])
                assert np.array_equal(rank,np.arange(1,len(matrix)+1))
                assert np.array_equal(mask,(rank<=cutoff)&(matrix[:,2]==1))
                assert int(initialization[4])==cutoff
            for name,indices in (('affine',(1,3,4)),('packet',(7,5,6)),('factor',(11,9,10))):
                if name=='affine' and not include_affine:continue
                if name!='affine':
                    validcol=8 if name=='packet' else 12
                    if not np.all(matrix[mask,validcol]==1):raise ValueError('Invalid exact coefficient in accepted set')
                depthkey='exact_'+name+'_output_z_depth' if 'exact_'+name+'_output_z_depth' in data else name+'_depth_z'
                slopekey='exact_'+name+'_output_slope' if 'exact_'+name+'_output_slope' in data else name+'_slope'
                cases.append((name,Coefficients(*[matrix[mask,k] for k in indices]),initialization,
                    scalar(data,depthkey),scalar(data,slopekey)))
                for field,col in zip(('alpha','peak','rho'),indices):
                    if name+'_'+field in data:
                        assert data[name+'_'+field].tobytes()==matrix[mask,col].tobytes()
                if name+'_selected_ids' in data:assert np.array_equal(data[name+'_selected_ids'],data['primitive_ids'][mask])
        elif 'alpha' in data:
            cases=[('single',Coefficients(*[data[k] for k in ('alpha','peak','rho')]),initialization,scalar(data,'depth_z'),scalar(data,'slope'))]
        else:
            for name in ('affine','packet','factor'):
                if name=='affine' and not include_affine:continue
                if name+'_alpha' not in data:continue
                cases.append((name,Coefficients(*[data[name+'_'+k] for k in ('alpha','peak','rho')]),initialization,
                    scalar(data,name+'_depth_z'),scalar(data,name+'_slope')))
        if not cases:raise ValueError('No canonical coefficient schema found')
    return cases

def verify_capture(folder):
    manifest_path=folder/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['status']=='complete' and manifest['captured_rays']==9
    checks={'sources':0,'archive_sources':0,'artifacts':0,'models':0,'generated_backend_sources':0,'rebuilt_binaries':0,'probe_sources':0,'probe_binaries':0,'whole_image_views':0}
    with zipfile.ZipFile(folder/'sources.zip') as archive:
        for name,digest in manifest['source_sha256'].items():
            assert sha(ROOT/name)==digest;checks['sources']+=1
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest;checks['archive_sources']+=1
    for name,digest in manifest['artifacts_sha256'].items():assert sha(folder/name)==digest;checks['artifacts']+=1
    for name,digest in manifest['model_sha256'].items():assert sha(ROOT/name)==digest;checks['models']+=1
    for arm,record in manifest['pinned_rebuilt_binaries'].items():
        assert sha(ROOT/record['binary_relative_path'])==record['rebuilt_binary_sha256'];checks['rebuilt_binaries']+=1
        assert manifest['backends'][arm]['binary_sha256']==record['rebuilt_binary_sha256']
        for name,digest in record['generated_sha256'].items():assert sha(ROOT/name)==digest;checks['generated_backend_sources']+=1
    for name,digest in manifest['probe']['generated_sha256'].items():assert sha(folder/'generated'/name)==digest;checks['probe_sources']+=1
    binaries=list((folder/'generated').glob('*.so'));assert len(binaries)==1
    assert sha(binaries[0])==manifest['probe']['binary_sha256'];checks['probe_binaries']=1
    for name in manifest['artifacts_sha256']:
        if not name.startswith('layout_'):continue
        layout=json.loads((folder/name).read_text(encoding='utf-8'))
        assert all(all(flags.values()) for flags in layout['whole_image_replay'].values());checks['whole_image_views']+=1
    assert checks['whole_image_views']==8
    files=[folder/f'ray_{int(row["case"]):02d}.npz' for row in manifest['selected_cases']]
    assert len(files)==9 and len(set(files))==9 and all(p.exists() for p in files)
    return files,{'manifest_sha256':sha(manifest_path),'verified_counts':checks,'selected_cases':manifest['selected_cases'],
        'scope':manifest['scope'],'binary_scope':'New pinned rebuilt binary with observed whole-image equality; not historical binary recovery or original hidden-buffer identity',
        'pinned_binary_sha256':{arm:{k:v for k,v in row.items() if k in ('frozen_old_binary_sha256','rebuilt_binary_sha256')} for arm,row in manifest['pinned_rebuilt_binaries'].items()}}

def analyze(name,coeff,initialization,depth_z,reported_slope,dps):
    result={'arm':name,'terms':len(coeff.alpha),'exact_mass':coeff.mass(),'mp_dps':dps,
      'minus_infinity_root':coeff.root(),'stable_double_minus_infinity_root':coeff.root64(),'camera_origin_root':coeff.root(camera_origin=True),
      'H_at_camera_origin':textnumber(coeff.H(mp.mpf(0))),'T_at_camera_origin':textnumber(mp.exp(-coeff.H(mp.mpf(0))))}
    for key in ('minus_infinity_root','camera_origin_root'):
        item=result[key]
        if item['status']=='finite':
            with mp.workdps(dps+60):
                lo,hi=[mp.mpf(x) for x in item['numeric_bracket']]
                offset=coeff.H(mp.mpf(0)) if key=='camera_origin_root' else mp.mpf(0)
                left,right=coeff.residual(lo)-offset,coeff.residual(hi)-offset
                item['higher_precision_endpoint_sign_check']={'dps':dps+60,'left':textnumber(left),'right':textnumber(right),'strict_signs_pass':bool(left<0<right)}
    if initialization is not None:
        result['captured_initialization']=[float(v) for v in initialization]
        result['initialization_scope']='Copied-expression CUDA probe, not original internal trace'
        result['finite_search_replays']={mode:finite_search(coeff,initialization,mode) for mode in ('mp','float64_log','float64_product','float32_product')}
        result['true_initial_endpoint_residuals']=[textnumber(coeff.residual(mp.mpf(float(t)))) for t in initialization[1:3]]
    if depth_z is not None:
        if initialization is None:raise ValueError('Output z-depth requires captured initialization/rln')
        rln=mp.mpf(float(initialization[5]));t=mp.mpf(depth_z)/rln
        result['observed_z_depth']=depth_z;result['observed_ray_depth_from_probe_rln']=textnumber(t)
        R=coeff.residual(t);hs=coeff.slopeH(t)
        result['at_observed_output']={'log_residual':textnumber(R),'T_minus_half':textnumber(mp.expm1(-R)/2),
          'H_slope':textnumber(hs),'minus_T_slope':textnumber(mp.exp(-R)*hs/2),
          'reported_cuda_slope_secant':reported_slope}
        result['positive_cuda_output_without_finite_represented_root']=bool(depth_z>0 and not coeff.mass()['full_minus_infinity_finite_root'])
        if result['minus_infinity_root']['status']=='finite':
            root=mp.mpf(result['minus_infinity_root']['depth'])
            result['observed_minus_represented_root_ray']=textnumber(t-root)
        zv=np.float32(depth_z);lower=(float(np.nextafter(zv,np.float32(-np.inf)))+float(zv))/2
        upper=(float(np.nextafter(zv,np.float32(np.inf)))+float(zv))/2
        result['conditional_z_multiplication_only_inferred_ray_interval']=[textnumber(mp.mpf(v)/rln) for v in (lower,upper)]
        result['ray_conversion_limit']='Interval assumes probe rln equals original rln; initialization probe is not original-kernel trace'
    return result

def selftests():
    rows=[]
    for alpha,expected in ((.5,'critical_asymptotic_no_finite_root'),(.25,'insufficient_mass_no_finite_root')):
        c=Coefficients(np.array([alpha],np.float32),np.array([2],np.float32),np.array([3],np.float32))
        status=c.root()['status'];rows.append({'case':'exact_mass_'+str(alpha),'pass':status==expected,'status':status})
    c=Coefficients(np.array([.8],np.float32),np.array([2],np.float32),np.array([3],np.float32))
    expected=mp.mpf(2)-mp.sqrt(-2*mp.log(mp.mpf('.75')/c.alpha[0]))/3
    actual=c.root();error=abs(mp.mpf(actual['depth'])-expected)
    rows.append({'case':'one_layer_closed_form','pass':error<mp.mpf('1e-30'),'error':textnumber(error)})
    initialization=np.array([2.,1.6,2.4,.2,1.,1.,0.,0.,1.],np.float32)
    replays={mode:finite_search(c,initialization,mode) for mode in ('mp','float64_log','float64_product','float32_product')}
    rows.append({'case':'one_layer_finite_search_modes','pass':all(r['status']=='finite' and abs(mp.mpf(r['depth'])-expected)<mp.mpf('1e-5') for r in replays.values()),
                 'replay_statuses':{k:v['status'] for k,v in replays.items()},'replay_depths':{k:v.get('depth') for k,v in replays.items()}})
    c=Coefficients(np.array([.5,.5],np.float32),np.array([-1,1],np.float32),np.array([1000,1000],np.float32))
    actual=c.root();error=abs(mp.mpf(actual['depth']))
    rows.append({'case':'exact_plateau_thin_tails','pass':error<mp.mpf('1e-30'),'error':textnumber(error)})
    double=c.root64()
    rows.append({'case':'double_underflow_plateau_not_false_root','pass':double['status']=='double_precision_plateau_unresolved','status':double['status']})
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,nargs='*',default=[]);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dps',type=int,default=120);parser.add_argument('--capture',type=Path);parser.add_argument('--include-affine',action='store_true');args=parser.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    if args.dps<80:raise ValueError('Use >=80 decimal digits')
    mp.mp.dps=args.dps;tests=selftests();rows=[];capture_checks=None
    if args.capture:
        if args.input:raise ValueError('Use capture or explicit inputs, not both')
        args.input,capture_checks=verify_capture(args.capture.resolve())
    for path in args.input:
        arms=[analyze(name,c,init,depth,slope,args.dps) for name,c,init,depth,slope in load(path,args.include_affine)]
        row={'input':str(path.resolve()),'input_sha256':sha(path),'arms':arms}
        byname={a['arm']:a for a in arms}
        if 'packet' in byname and 'factor' in byname:
            b,c=byname['packet'],byname['factor']
            if b['minus_infinity_root']['status']=='finite' and c['minus_infinity_root']['status']=='finite':
                delta=mp.mpf(b['minus_infinity_root']['depth'])-mp.mpf(c['minus_infinity_root']['depth'])
                row['packet_minus_factor_represented_root_delta_ray']=textnumber(delta)
                if 'observed_ray_depth_from_probe_rln' in b and 'observed_ray_depth_from_probe_rln' in c:
                    observed=mp.mpf(b['observed_ray_depth_from_probe_rln'])-mp.mpf(c['observed_ray_depth_from_probe_rln'])
                    row['delta_decomposition_ray']={'observed':textnumber(observed),'represented_roots':textnumber(delta),
                        'remaining_numerical_and_conversion_path':textnumber(observed-delta),
                        'limit':'Remainder is not attributed to product arithmetic without replay evidence; probe-rln uncertainty remains'}
            else:row['delta_decomposition_status']='undefined: at least one represented full-ray finite root does not exist or is unresolved'
        rows.append(row);print(f'{path.name}: '+','.join(a['arm']+'='+a['minus_infinity_root']['status'] for a in arms),flush=True)
    sources=[Path(__file__),ROOT/'research/EXACT_OUTLIER_ROOT_THEORY.md',ROOT/'research/EXACT_OUTLIER_ANALYSIS_PROTOCOL.md',ROOT/'submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu',ROOT/'research/THIN_QUANTILE_LIMIT.md']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','posthoc':True,'mpmath':mp.__version__,'numpy':np.__version__,
      'selftests':tests,'selftests_pass':all(r['pass'] for r in tests),'inputs':len(rows),'rows':rows,
      'analyzed_exact_arms':sum(a['arm'] in ('packet','factor') for r in rows for a in r['arms']),
      'capture_checks':capture_checks,
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
      'limits':['Only saved FP32 coefficients are treated as exact binary rationals; no raw-world equivalence claim.',
       'MP root brackets are high-precision numerical evidence, not outward interval or formal certificates.',
       'CPU float32 replay does not reproduce CUDA fast exp/rsqrt/FMA instruction identity.',
       'All posthoc selected outliers are retained; their frequency cannot be extrapolated beyond the enumerated frozen images.',
       'No training, GT improvement, speed, full-scene support or novelty claim.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if not result['selftests_pass']:raise ArithmeticError('Preserved oracle selftest failure')

if __name__=='__main__':main()
