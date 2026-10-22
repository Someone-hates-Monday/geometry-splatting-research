"""Independent counters, exact cache/sample envelopes, actual K distributions."""
import hashlib,json,traceback
from fractions import Fraction as F
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_real_work_v7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(x):return F.from_float(float(x))
def domain_c(v):return v==0 or 2.**-60<=abs(v)<=2.**20
def main():
    target=OUT/'independent_audit.json';assert not target.exists()
    plan=json.loads((OUT/'input_plan.json').read_text());result=json.loads((OUT/'results.json').read_text());assert result['input_plan_sha256']==sha(OUT/'input_plan.json')
    repair=OUT/'sentinel_v7_1';correction=json.loads((repair/'results.json').read_text());assert correction['all_gates'] and not result['all_gates']
    repairplan=json.loads((repair/'plan.json').read_text());assert repairplan['initial_results_sha256']==sha(OUT/'results.json');assert repairplan['addendum_sha256']==sha(ROOT/'TGS_REAL_WORK_ADDENDUM_V7_1.md')
    assert repairplan['script_sha256']==sha(ROOT/'repair_tgs_real_work_sentinel.py')
    oldtext=(ROOT/'tgs_real_work_probe.cu').read_text();assert (repair/'tgs_real_work_probe_v7_1.cu').read_text()==oldtext.replace('mid=0xffffffff;','mid=__float2uint_rz(-1.0f);')
    assert sha(ROOT/'TGS_REAL_WORK_PROTOCOL_V7.md')==plan['protocol_sha256'];assert sha(ROOT/'tgs_real_work_probe.cu')==plan['source_sha256']
    build=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(build['module_file']))==build['module_sha256']
    execplan=json.loads((OUT/'execution_plan.json').read_text());assert execplan['script_sha256']==sha(ROOT/'profile_tgs_real_work.py')
    v5=ROOT/'experiments/tgs_phase_cost_v5';v6=ROOT/'experiments/tgs_compensated_edge_v6';assert sha(v5/'evidence_manifest.json')==plan['v5_manifest_sha256'];assert sha(v6/'evidence_manifest.json')==plan['v6_manifest_sha256']
    src={e['path']:e['sha256'] for e in json.loads((v5/'evidence_manifest.json').read_text())['files']};helper=v5/'generated/native/cuda_rasterizer/stable_window.cuh';assert sha(helper)==build['helper_sha256']==src[helper.relative_to(ROOT).as_posix()]
    w,h=plan['resolution'];rows=[]
    columns=['visited','coarse_outside','compensated_outside','inside_unique','inside_edge_uncertain','sign_uncertain','domain_fallback','native_outside','native_below_cutoff','included','native_stop','coarse_wrong','comp_outside_wrong','inside_wrong','unique_wrong','coord_bound_bad','alpha_budget_pass','alpha_budget_fail','alpha_error_exceeds_allowance','cutoff_disagreement_budget_pass','cap_disagreement_budget_pass','sample_written','conditional_window_accept','conditional_margin_reject']
    for ci,row in enumerate(result['rows']):
        p=OUT/f'camera_{ci}_raw.npz'
        with np.load(p) as z:d={k:z[k] for k in z.files}
        with np.load(repair/f'camera_{ci}_path.npz') as z:
            expected=d['path'].copy();expected[1][d['counts'][:,9].reshape(h,w)==0]=0;assert np.array_equal(z['path'],expected);assert np.array_equal(z['path'],z['native_path']);assert np.array_equal(z['native_path'],d['native_path']);d['path']=z['path'].copy()
        ct=d['counts'].astype(np.int64);assert ct.shape==(w*h,24) and (ct>=0).all();totals=ct.sum(0);assert totals.tolist()==row['totals'];assert np.all(ct[:,0]==ct[:,1:7].sum(1));assert np.all(ct[:,0]==ct[:,7]+ct[:,8]+ct[:,9]+ct[:,10]);assert np.all(ct[:,10]<=1)
        assert np.all(ct[:,16]+ct[:,17]==ct[:,3]);assert np.all(ct[:,22]+ct[:,23]==ct[:,16]);assert np.array_equal(d['rgb'],d['native_rgb']);assert np.array_equal(d['aux'],d['native_aux']);assert np.array_equal(d['path'],d['native_path']);assert np.array_equal(d['maxblend'],d['native_maxblend'])
        K=ct[:,9];hist=np.bincount(K);assert np.array_equal(hist,np.load(OUT/f'camera_{ci}_K_histogram.npy'));assert int(hist.sum())==w*h;assert sum(int(k)*int(v) for k,v in enumerate(hist))==int(totals[9]);assert np.all(d['path'][0].ravel()<=ct[:,0])
        overflow={str(k):int(hist[k+1:].sum()) if len(hist)>k+1 else 0 for k in [16,32,64,128,256]};assert overflow==row['K_overflow']
        c,an,mn,mx,vis=d['coeff'],d['anchors'],d['rmin'],d['rmax'],d['radii']>0;vi=np.flatnonzero(vis)
        B=np.maximum(np.abs(16*mn[vis]-an[vis]),np.abs(np.minimum(16*mx[vis],[w,h])-1-an[vis]));assert np.array_equal(B,d['B'][vis])
        assert int(np.prod(mx[vis]-mn[vis],axis=1).astype(np.int64).sum())==row['native_tiles']
        # Independent all-visible domain classification; no FTZ equal-to-zero shortcut.
        cv=c[vis];hi=cv[:,:9].astype(np.float32);lo=(cv[:,:9]-hi.astype(float)).astype(np.float32)
        dom=np.all((cv[:,:9]==0)|((np.abs(cv[:,:9])>=2.**-60)&(np.abs(cv[:,:9])<=2.**20)),axis=1)
        for v in [hi,lo]:dom &= np.all((v==0)|((np.abs(v)>=2.**-60)&(np.abs(v)<=2.**20)),axis=1)
        dom &= np.isfinite(cv[:,:9]).all(1)&np.isfinite(an[vis]).all(1)&(np.abs(an[vis])<2.**23).all(1)&(an[vis]==np.floor(an[vis])).all(1)&(B<=8192).all(1)&(mx[vis]>mn[vis]).all(1)
        assert np.array_equal(dom,d['domain'][vis].astype(bool))
        # Verify the PREDECLARED hash selection plus footprint extrema (not outcome-selected).
        cam=plan['cameras'][ci];ranks=np.array([int.from_bytes(hashlib.sha256(('tgs_real_work_v7/cache/'+cam['img_name']+'/'+str(i)).encode()).digest()[:8],'big') for i in vi],dtype=np.uint64)
        selected=np.unique(np.r_[vi[np.argsort(ranks,kind='stable')[:512]],vi[np.argsort(-(B[:,0]*B[:,1]),kind='stable')[:32]]]);assert np.array_equal(selected,d['audit_triangles'])
        cachechecks=0
        for i,A,C in zip(d['audit_triangles'],d['audit_split'],d['audit_cache']):
            bounds=list(map(q,[*d['B'][i],1]));cc=list(map(q,c[i]));assert all(domain_c(v) for v in c[i,:9]) if d['domain'][i] else True
            for k in range(3):
                ids=[2*k,2*k+1,6+k];aa=list(map(q,A[6*k:6*k+6]));M=Delta=M64=Dhi=F(0)
                for t,j in enumerate(ids):
                    hh=np.float32(c[i,j]);ll=np.float32(c[i,j]-float(hh));assert A[6*k+t]==hh and A[6*k+t+3]==ll
                    M+=(abs(aa[t])+abs(aa[t+3]))*bounds[t];Delta+=abs(cc[j]-aa[t]-aa[t+3])*bounds[t];M64+=abs(cc[j])*bounds[t];Dhi+=abs(cc[j]-aa[t])*bounds[t]
                assert q(C[4*k])>=M;assert q(C[4*k+1])>=Delta;assert q(C[4*k+2])>=F(1,2**51)*M64
                assert q(C[4*k+3])>=Dhi+F(1,2**22)*M+F(1,2**51)*M64;cachechecks+=1
            assert C[12]==max(C[3],C[7],C[11]);assert C[13]==np.float32(c[i,9]);assert q(C[14])>=abs(cc[9]-q(C[13]));assert q(C[15])>=cc[9]
        samples=d['samples'];assert int((samples[:,0]>0).sum())==row['sample_pairs']==int(totals[21]);samplechecks=0;sample_stage={}
        for si,s in enumerate(samples):
            if s[0]==0:continue
            i,X,Y,ordinal,stage=map(int,s[1:6]);assert Y*w+X==plan['sample_pixels'][ci][si];assert ordinal<=ct[Y*w+X,0]
            x,y=q(X)-q(an[i,0]),q(Y)-q(an[i,1]);assert abs(x)<=q(d['B'][i,0]) and abs(y)<=q(d['B'][i,1]);Q=[x,y,F(1)];planes=[];nativeplanes=[]
            for k in range(3):
                vv=[c[i,2*k],c[i,2*k+1],c[i,6+k]];planes.append(sum(q(v)*z for v,z in zip(vv,Q)));nativeplanes.append((float(vv[0])*float(x)+float(vv[1])*float(y))+float(vv[2]))
            phi64=max(nativeplanes);outside=phi64>0;active=nativeplanes.index(phi64);assert bool(s[16])==outside and int(s[17])==active
            if stage!=6:assert abs(q(s[6])-q(phi64))<=q(s[7]);assert abs(q(s[6])-max(planes))<=q(s[7])
            if stage not in [1,6]:
                for k in range(3):assert abs(q(s[10+2*k])-q(nativeplanes[k]))<=q(s[11+2*k])
                assert max(s[10],s[12],s[14])==s[8];assert max(s[11],s[13],s[15])==s[9]
            if stage in [1,2]:assert outside
            if stage in [3,4]:assert not outside
            if stage==3:assert int(s[22])==active
            sample_stage[str(stage)]=sample_stage.get(str(stage),0)+1;samplechecks+=1
        if ci<4:
            old=v5/f'real_{ci}_oracle.npz';assert sha(old)==src[old.relative_to(ROOT).as_posix()]
            with np.load(old) as z:
                for key,oldkey in [('native_rgb','rgb'),('native_aux','aux'),('native_maxblend','maxblend'),('radii','radii'),('coeff','double_coeff')]:assert np.array_equal(d[key],z[oldkey])
                assert np.array_equal(d['path'].view(np.uint32),z['contributors'])
        stats=dict(camera=ci,name=row['name'],new_holdout=row['new_holdout'],totals=dict(zip(columns,map(int,totals))),stage_fraction_visited={columns[j]:float(totals[j]/totals[0]) for j in range(1,7)},native_included_fraction=float(totals[9]/totals[0]),conditional_accept_fraction_all=float(totals[22]/totals[0]),conditional_accept_fraction_nonoutside=float(totals[22]/(totals[0]-totals[1]-totals[2])),K_quantiles=np.quantile(K,[0,.5,.9,.95,.99,1]).tolist(),K_zero=int(hist[0]),K_overflow=overflow,K_overflow_fraction={k:v/(w*h) for k,v in overflow.items()},exact_edge_cache_checks=cachechecks,exact_reached_sample_checks=samplechecks,reached_sample_stages=sample_stage,gate=bool(totals[11:16].sum()==0),raw_sha256=sha(p))
        rows.append(stats);print(json.dumps(stats),flush=True)
    output=dict(rows=rows,all_gates=all(r['gate'] for r in rows),initial_v7_endpoint_gate=False,corrected_v7_1_endpoint_gate=True,repair_results_sha256=sha(repair/'results.json'),script_sha256=sha(Path(__file__)),result_sha256=sha(OUT/'results.json'),input_plan_sha256=sha(OUT/'input_plan.json'),columns=columns,training=False,timing=False,candidate_rendering=False,power_interval_proved=False,performance_promotion=False,production_promotion=False)
    target.write_text(json.dumps(output,indent=2)+'\n');assert output['all_gates']
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'audit_failure.txt').write_text(traceback.format_exc());raise
