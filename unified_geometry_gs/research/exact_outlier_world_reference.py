"""CPU MP80/140 covariance references on all nine captured outlier rays."""
from __future__ import annotations
import argparse,bisect,hashlib,json,os,time,traceback,zipfile
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import mpmath as mp
import numpy as np
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','opacity','filter_3D')
MODELS=('saved_activated','raw_physical')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def number(x,digits=None):return mp.nstr(x,digits or mp.mp.dps)
def exact(x):return mp.mpf(float(x))
def vector(x):return mp.matrix([exact(v) for v in x])
def matrix(x):return mp.matrix([[exact(v) for v in row] for row in x])
def dot(x,y):return (x.T*y)[0]
def rotation(q):
    w,x,y,z=q
    return mp.matrix([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def round32_positive(value):
    candidate=np.float32(float(value))
    nearby=[candidate,np.nextafter(candidate,np.float32(-np.inf)),np.nextafter(candidate,np.float32(np.inf))]
    nearby=[v for v in nearby if np.isfinite(v) and v>=0]
    return min(nearby,key=lambda v:(abs(exact(v)-value),int(v.view(np.uint32))&1))

def references(data,raw,model,dps):
    """80: explicit covariance inverse; 140: inverse factor and residual energy."""
    with mp.workdps(dps):
        view=data['world_view_transform'];A=matrix(view[:3,:3]).T;t=vector(view[3,:3]);Ai=A**-1
        origin=-Ai*t;q=mp.matrix([exact(data['query_xy'][0]),exact(data['query_xy'][1]),mp.mpf(1)])
        qnorm=mp.sqrt(dot(q,q));direction=Ai*q/qnorm;cap=exact(np.float32(.99));mask=data['accepted_mask']
        scales=data['filtered_scales'][mask];quaternions=data['normalized_quaternions'][mask];opacity=data['filtered_opacity'][mask].reshape(-1)
        values=[]
        for j,row in enumerate(raw):
            mean=vector(row[:3]);difference=mean-origin
            if model=='saved_activated':
                R=rotation([exact(v) for v in quaternions[j]])
                variance=[exact(v)**2 for v in scales[j]];amplitude=exact(opacity[j])
            else:
                variances=[mp.exp(2*exact(v)) for v in row[3:6]];radius=exact(row[11])
                variance=[v+radius*radius for v in variances]
                quat=[exact(v) for v in row[6:10]];qn=mp.sqrt(mp.fsum(v*v for v in quat));quat=[v/qn for v in quat]
                R=rotation(quat)
                amplitude=1/(1+mp.exp(-exact(row[10])))*mp.sqrt(mp.fprod(variances)/mp.fprod(variance))
            if dps==80:
                covariance=R*mp.diag(variance)*R.T;precision=covariance**-1
                a=dot(direction,precision*direction);b=dot(direction,precision*difference);c=dot(difference,precision*difference)
                peak=b/a;energy=c-b*b/a
            else:
                # R need not be orthogonal: use its genuine inverse.
                inverse_factor=mp.diag([1/mp.sqrt(v) for v in variance])*(R**-1)
                u=inverse_factor*direction;v=inverse_factor*difference
                a=dot(u,u);b=dot(u,v);c=dot(v,v);peak=b/a
                residual=v-peak*u;energy=dot(residual,residual)
            rho=mp.sqrt(a);uncapped=amplitude*mp.exp(-energy/2);alpha=min(cap,uncapped)
            defect=R.T*R-mp.eye(3);orthogonality=max(abs(v) for v in defect)
            valid=bool(a>0 and energy>=0 and 0<=alpha<1 and all(mp.isfinite(v) for v in (a,b,c,peak,rho,energy,uncapped)))
            alpha32=round32_positive(alpha)
            values.append({'a':number(a),'b':number(b),'c':number(c),'peak':number(peak),'rho':number(rho),'energy':number(energy),
                'amplitude':number(amplitude),'alpha_uncapped':number(uncapped),'alpha_capped':number(alpha),'cap_active':bool(uncapped>cap),
                'alpha_capped_F32_diagnostic_exact_value':number(exact(alpha32)),'alpha_capped_F32_diagnostic_bits_hex':f'{int(alpha32.view(np.uint32)):08x}',
                'rotation_orthogonality_max_absolute':number(orthogonality),'valid':valid})
        info={'camera_column_matrix_A':[[number(A[i,j]) for j in range(3)] for i in range(3)],
            'origin_world':list(map(number,origin)),'direction_world_per_camera_unit_t':list(map(number,direction)),
            'direction_world_norm':number(mp.sqrt(dot(direction,direction))),'query_norm_exact_MP':number(qnorm),
            'saved_rln_F32_exact_value':number(exact(data['rln'][0])),
            'origin_difference_from_saved_F32_camera_center_max_absolute':number(max(abs(origin[i]-exact(data['camera_center'][i])) for i in range(3))),
            'cap_value_exact_F32_point99':number(cap),'precision_dps':dps,
            'construction':'explicit_world_covariance_inverse' if dps==80 else 'genuine_inverse_rotation_factor_and_perpendicular_residual',
            'transmittance_domain':'GGGS product normalized at minus infinity; no additional camera-origin T(0) conditioning',
            'parameter_units':'unit camera-ray parameter; world direction is not renormalized'}
        return values,info

class ContinuousModel:
    def __init__(self,alpha,peak,rho,binary_alpha=None):
        self.alpha=list(alpha);self.peak=list(peak);self.rho=list(rho)
        self.order=sorted(range(len(alpha)),key=lambda j:self.peak[j]);self.ordered=[self.peak[j] for j in self.order]
        self.logs=[-mp.log1p(-a) for a in alpha];self.prefix=[mp.mpf(0)]
        for k in range(1,len(self.order)+1):self.prefix.append(mp.fsum(self.logs[j] for j in self.order[:k]))
        self.fraction_prefix=None
        if binary_alpha is not None:
            self.fraction_prefix=[Fraction(1)]
            for j in self.order:self.fraction_prefix.append(self.fraction_prefix[-1]*(1-Fraction(float(binary_alpha[j]))))

    def residual(self,t):
        k=bisect.bisect_left(self.ordered,t)
        if self.fraction_prefix is not None:
            beta=self.fraction_prefix[k];delta=(Fraction(1,2)-beta)/beta
            constant=mp.log1p(mp.mpf(delta.numerator)/delta.denominator)
        else:constant=self.prefix[k]-mp.log(2)
        tails=[]
        for alpha,peak,rho in zip(self.alpha,self.peak,self.rho):
            if alpha==0:continue
            near=-mp.log1p(-alpha*mp.exp(-((t-peak)*rho)**2/2))/2
            tails.append(-near if t>peak else near)
        return mp.fsum([constant,*tails])

    def slope(self,t):
        return mp.fsum(alpha*mp.exp(-((t-peak)*rho)**2/2)*abs((t-peak)*rho)*rho/
            (2*(1-alpha*mp.exp(-((t-peak)*rho)**2/2))) for alpha,peak,rho in zip(self.alpha,self.peak,self.rho))

    def root(self):
        margin=mp.fsum(self.logs)-mp.log(2);threshold=mp.power(10,-mp.mp.dps+20)
        if self.fraction_prefix is not None:
            terminal=self.fraction_prefix[-1]
            kind='finite' if terminal<Fraction(1,2) else 'critical_no_finite_root' if terminal==Fraction(1,2) else 'insufficient_mass_no_finite_root'
            mass={'exact_terminal_numerator':str(terminal.numerator),'exact_terminal_denominator':str(terminal.denominator)}
        else:
            kind='finite' if margin>threshold else 'insufficient_mass_no_finite_root' if margin<-threshold else 'mass_precision_unresolved'
            mass={'precision_guard':number(threshold)}
        mass.update(classification=kind,total_logmass_minus_log2=number(margin))
        if kind!='finite':return {'status':kind,'mass':mass}
        reach=max(mp.mpf(1),10/min(self.rho));lo=min(self.peak)-reach;hi=max(self.peak)+reach
        for _ in range(48):
            if self.residual(lo)<0:break
            lo-=reach;reach*=2
        else:return {'status':'left_bracket_unresolved','mass':mass}
        for _ in range(48):
            if self.residual(hi)>0:break
            hi+=reach;reach*=2
        else:return {'status':'right_bracket_unresolved','mass':mass}
        tolerance=mp.mpf('1e-30' if mp.mp.dps==80 else '1e-42')
        for iterations in range(400):
            if hi-lo<=tolerance:break
            mid=(lo+hi)/2;value=self.residual(mid)
            if value<0:lo=mid
            elif value>0:hi=mid
            else:return {'status':'residual_precision_unresolved','mass':mass,'bracket':[number(lo),number(hi)]}
        else:return {'status':'iteration_limit','mass':mass}
        depth=(lo+hi)/2;R=self.residual(depth);Ht=self.slope(depth)
        return {'status':'finite','mass':mass,'camera_ray_depth':number(depth),'bracket':[number(lo),number(hi)],'bracket_width':number(hi-lo),
            'residual_H_minus_log2':number(R),'H_slope_at_root':number(Ht),'minus_T_slope_at_root':number(mp.exp(-R)*Ht/2),'iterations':iterations}

def model_from_reference(rows):
    return ContinuousModel([mp.mpf(r['alpha_capped']) for r in rows],[mp.mpf(r['peak']) for r in rows],[mp.mpf(r['rho']) for r in rows])
def model_from_capture(data,arm):
    arrays=[data[arm+'_'+k] for k in ('alpha','peak','rho')]
    return ContinuousModel(*[[exact(v) for v in x] for x in arrays],binary_alpha=arrays[0])

def summarize_errors(data,reference):
    result={}
    for arm in ('packet','factor'):
        error={}
        for key,column in (('alpha','alpha_capped'),('peak','peak'),('rho','rho')):
            target=[mp.mpf(r[column]) for r in reference];actual=[exact(v) for v in data[arm+'_'+key]]
            absolute=[abs(a-b) for a,b in zip(actual,target)]
            scaled=[e/max(mp.mpf('1e-30'),abs(v)) for e,v in zip(absolute,target)]
            error[key]={'max_absolute':number(max(absolute)),'l1_absolute':number(mp.fsum(absolute)),
                'max_relative_floor1e_minus30':number(max(scaled))}
        result[arm]=error
    return result

def process_case(out,case,data,raw,ids):
    refs={};roots={};info={}
    for dps in (80,140):
        refs[dps]={};roots[dps]={};info[dps]={}
        with mp.workdps(dps):
            for model in MODELS:
                rows,camera=references(data,raw,model,dps);refs[dps][model]=rows;info[dps][model]=camera
                roots[dps][model]=model_from_reference(rows).root() if all(r['valid'] for r in rows) else {'status':'invalid_reference_coefficients'}
            for arm in ('packet','factor'):roots[dps][arm]=model_from_capture(data,arm).root()
    checks={};comparisons={}
    with mp.workdps(140):
        for model in MODELS:
            differences={}
            for key in ('a','b','c','peak','rho','energy','alpha_uncapped','alpha_capped'):
                errors=[abs(mp.mpf(a[key])-mp.mpf(b[key]))/max(mp.mpf('1e-60'),abs(mp.mpf(b[key]))) for a,b in zip(refs[80][model],refs[140][model])]
                differences[key]=number(max(errors))
            checks[model]={'max_scaled_coefficient_differences_floor1e_minus60':differences,'root80_status':roots[80][model]['status'],'root140_status':roots[140][model]['status']}
        for model in MODELS+('packet','factor'):
            low=roots[80][model];high=roots[140][model]
            if model not in checks:checks[model]={}
            checks[model].update(root80_status=low['status'],root140_status=high['status'])
            if low['status']==high['status']=='finite':
                candidate=model_from_reference(refs[140][model]) if model in MODELS else model_from_capture(data,model)
                lo,hi=map(mp.mpf,low['bracket']);fl=candidate.residual(lo);fh=candidate.residual(hi)
                checks[model].update(root80_bracket_signs_at140=[number(fl),number(fh)],root80_bracket_straddles_at140=bool(fl<0<fh),
                    root_depth_difference_80_140=number(abs(mp.mpf(low['camera_ray_depth'])-mp.mpf(high['camera_ray_depth']))))
        for model in MODELS:
            reference=roots[140][model];comparison={'coefficient_errors':summarize_errors(data,refs[140][model]),'reference_root_status':reference['status']}
            if reference['status']=='finite':
                r=mp.mpf(reference['camera_ray_depth']);rn=mp.mpf(info[140][model]['query_norm_exact_MP']);rln=exact(data['rln'][0]);errors={};cuda_errors={}
                for arm in ('packet','factor'):
                    if roots[140][arm]['status']=='finite':
                        value=mp.mpf(roots[140][arm]['camera_ray_depth']);errors[arm]=abs(value-r)
                        comparison[arm+'_mathematical_root_difference_camera_ray_signed']=number(value-r)
                        comparison[arm+'_mathematical_root_difference_render_z_signed']=number((value-r)*rln)
                    value=exact(data['exact_'+arm+'_output_z_depth'][0]);comparison[arm+'_CUDA_z_output_minus_reference_render_z']=number(value-r*rln);cuda_errors[arm]=abs(value-r*rln)
                comparison['reference_depth_render_z_using_saved_rln']=number(r*rln);comparison['reference_depth_ideal_camera_z']=number(r/rn)
                if len(errors)==2:
                    comparison['closer_captured_coefficient_model_under_this_reference']='packet' if errors['packet']<errors['factor'] else 'factor' if errors['factor']<errors['packet'] else 'equal'
                comparison['closer_CUDA_z_output_under_this_reference']='packet' if cuda_errors['packet']<cuda_errors['factor'] else 'factor' if cuda_errors['factor']<cuda_errors['packet'] else 'equal'
            comparisons[model]=comparison
    np.savez_compressed(out/f'ray_{case:02d}_inputs.npz',accepted_primitive_ids=ids,primitive_ids=data['primitive_ids'],raw_parameters_original_PLY_accepted=raw,**{k:data[k] for k in
        ('accepted_mask','primitive_ids','filtered_scales','normalized_quaternions','filtered_opacity','world_view_transform','full_proj_transform','camera_center','query_xy','rln','focal','pixel') if k!='primitive_ids'})
    result={'case':case,'accepted_inputs':len(raw),'primitive_ids':ids.tolist(),'references':{str(k):v for k,v in refs.items()},
        'camera_conventions':{str(k):v for k,v in info.items()},'roots':{str(k):v for k,v in roots.items()},'precision_checks':checks,'comparisons':comparisons,
        'limits':['Same 9 selected outliers and actual accepted prefix IDs; not independent scene or GT evidence.',
            'Reference alpha remains high precision after applying the exact F32 .99 cap value; no F32 rounding before reference roots.',
            'Additionally stored F32-rounded cap diagnostics are separate objects and are not used by the reference roots.',
            'MP precision agreement and endpoint signs are numerical cross-checks, not interval-certified proofs.',
            'A root-local slope does not bound a cross-layer gap; no such inference is made.']}
    save(out/f'ray_{case:02d}.json',result)
    return {'case':case,'accepted_inputs':len(raw),'root_statuses':{k:roots[140][k]['status'] for k in MODELS+('packet','factor')},
        'precision_checks':checks,'comparisons':comparisons}

def execute(out,state,capture):
    start=time.perf_counter();rows=[];total=0
    for scan in (37,55,63):
        ply=ROOT/f'output_compare/ggs_pipeline/scan{scan}/point_cloud/iteration_30000/point_cloud.ply';expected=state['model_sha256'][ply.relative_to(ROOT).as_posix()]
        assert sha(ply)==expected;vertices=PlyData.read(ply)['vertex'].data
        for case in range(9):
            meta=read(capture/f'ray_{case:02d}.json')
            if meta['scan']!=scan:continue
            with np.load(capture/f'ray_{case:02d}.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
            mask=data['accepted_mask'];ids=data['primitive_ids'][mask];source=vertices[ids]
            assert all(source[k].dtype==np.float32 for k in FIELDS), 'Original PLY raw fields must be binary32'
            raw=np.column_stack([source[k] for k in FIELDS]).astype(np.float32,copy=False)
            assert data['raw_parameters'].dtype==raw.dtype==np.float32
            assert np.array_equal(data['raw_parameters'][mask].view(np.uint32),raw.view(np.uint32))
            assert np.array_equal(data['packet_selected_ids'],ids) and np.array_equal(data['factor_selected_ids'],ids)
            assert data['world_view_transform'].dtype==np.float32 and np.array_equal(data['world_view_transform'][:,3],np.array([0,0,0,1],np.float32))
            co=data['conic_opacity'][:,3];op=data['filtered_opacity'].reshape(-1);diff=co.view(np.uint32)!=op.view(np.uint32)
            row=process_case(out,case,data,raw,ids);row.update({'scan':scan,'camera':meta['camera'],'pixel':meta['pixel'],
                'raw_PLY_byte_checks':len(ids),'conic_opacity_vs_filtered_opacity_all_candidates':{'inputs':len(co),'bit_different':int(sum(diff)),'different_ids':data['primitive_ids'][diff].tolist()},
                'conic_opacity_vs_filtered_opacity_accepted_bit_different':int(sum(diff[mask]))})
            rows.append(row);total+=len(ids);save(out/'progress.json',{'utc':now(),'completed_rays':len(rows),'accepted_inputs':total,'last_case':case})
            save(out/f'ray_{case:02d}_summary.json',row);print(f'completed {len(rows)}/9 case{case}: {row["root_statuses"]}',flush=True)
        assert sha(ply)==expected;del vertices
    assert len(rows)==9 and total==772
    summary={'status':'complete','rays':9,'accepted_input_rows':total,'PLY_raw_bytes_verified':total,'precision_dps':[80,140],
        'reference_model_names':list(MODELS),'roots_use_unrounded_high_precision_reference_coefficients':True,
        'conic_opacity_vs_filtered_opacity_all_candidate_differences':sum(r['conic_opacity_vs_filtered_opacity_all_candidates']['bit_different'] for r in rows),
        'conic_opacity_vs_filtered_opacity_accepted_differences':sum(r['conic_opacity_vs_filtered_opacity_accepted_bit_different'] for r in rows),
        'rows':sorted(rows,key=lambda r:r['case']),'limits':['Posthoc outlier mechanisms, no GT/quality or generic precision ranking.',
            'Saved F32 view uses its exact affine inverse; t is unit camera-ray parameter, not assumed unit world arclength.',
            'All nonfinite/insufficient/unresolved root states are retained, never forced to finite.']}
    save(out/'summary.json',summary)
    for name,digest in state['source_sha256'].items():assert sha(ROOT/name)==digest,name
    state.update(status='complete',completed_utc=now(),wall_seconds=time.perf_counter()-start,
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',state);print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if out.exists():raise ValueError('Fresh output required; preserve terminal failures.')
    capture=ROOT/'research/experiments/exact_outlier_capture_v3';prior=read(capture/'manifest.json');assert prior['status']=='complete' and prior['captured_rays']==9
    for name,digest in prior['artifacts_sha256'].items():assert sha(capture/name)==digest,name
    sources={Path(__file__),ROOT/'research/EXACT_OUTLIER_COEFFICIENT_PROTOCOL.md',capture/'manifest.json',capture/'summary.json'}
    for case in range(9):sources.update([capture/f'ray_{case:02d}.npz',capture/f'ray_{case:02d}.json'])
    out.mkdir(parents=True);state={'created_utc':now(),'status':'running','pid':os.getpid(),'mpmath':mp.__version__,'numpy':np.__version__,
        'precision_dps':[80,140],'raw_parameter_fields':list(FIELDS),'model_sha256':prior['model_sha256'],
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)},
        'scope':'CPU only, saved-activated true covariance inverse and raw-physical world reference; no CUDA/Torch import or kernel.'}
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for source in sorted(sources):z.write(source,source.relative_to(ROOT).as_posix())
    try:execute(out,state,capture)
    except BaseException as exc:
        state.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc));save(out/'manifest.json',state)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
