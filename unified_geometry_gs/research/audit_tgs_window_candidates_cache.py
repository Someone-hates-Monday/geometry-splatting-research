"""Exact coefficient/footprint envelope checks, independent oracle forward recapture."""
import hashlib,json,traceback
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
from tgs_window_candidates_common import ROOT,OUT,sha,align,modules,workload,decode
from audit_tgs_compensated_baseline_diagnostic import cache_audit as standard_cache_audit,anchor_view,rects
from probe_tgs_stable_window import SafeNative,SafeAuxNative
def q(x):return F.from_float(float(x))
def coarse_cache_audit(C,E,anchor,mn,mx,radii,w,h,selected=None):
    visible=radii>0;vi=np.flatnonzero(visible);B=np.maximum(np.abs(16*mn[visible].astype(float)-anchor[visible]),np.abs(np.minimum(16*mx[visible].astype(float),[w,h])-1-anchor[visible]));assert np.array_equal(B,E[visible,10:12]);cv=C[visible,:9];hh=cv.astype(np.float32);dom=np.isfinite(cv).all(1)&np.isfinite(anchor[visible]).all(1)&(np.abs(anchor[visible])<2.**23).all(1)&(anchor[visible]==np.floor(anchor[visible])).all(1)&(B<=8192).all(1)&(mx[visible]>mn[visible]).all(1)&(w<=8192)&(h<=8192)
    for v in [cv,hh]:dom&=((v==0)|((np.abs(v)>=2.**-60)&(np.abs(v)<=2.**20))).all(1)
    assert np.array_equal(dom,E[visible,12]==1);chosen=vi if selected is None else selected;checked=skipped=0
    for i in chosen:
        if not E[i,12]:skipped+=1;continue
        P=E[i];cc=list(map(q,C[i]));bounds=list(map(q,[*P[10:12],1]));assert P[13]==0
        for k in range(3):
            M=D=M64=F(0)
            for t,j in enumerate([2*k,2*k+1,6+k]):
                hi=np.float32(C[i,j]);assert P[3*k+t]==hi;M+=abs(q(hi))*bounds[t];D+=abs(cc[j]-q(hi))*bounds[t];M64+=abs(cc[j])*bounds[t]
            assert q(P[9])>=D+F(1,2**22)*M+F(1,2**51)*M64;checked+=1
    return dict(visible=len(vi),domain=int(dom.sum()),checked_edges=checked,skipped_domain_triangles=skipped,selected_triangles=np.asarray(chosen).tolist())
def main():
    target=OUT/'cache_audit.json';assert not target.exists();plan=json.loads((OUT/'input_plan.json').read_text());mods=modules();case=[];directory=OUT/'cache_audit';directory.mkdir();(directory/'plan.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),input_plan_sha256=sha(OUT/'input_plan.json'),selection='all visible small cases;512 minimum SHA256(v9cache/<camera>/<id>)+32 largest footprint products',training=False,timing=False),indent=2)+'\n')
    for en in plan['cases']:
        with np.load(ROOT/en['input']) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        f['pixel_grad']=f.get('patterns',np.array([f['pixel_grad']]))[0];rr={}
        for v in ['coarse','standard']:
            native=(SafeAuxNative if f['kind']=='aux' else SafeNative)(mods[v],f);run=native.forward(f['theta']);r=run[1];ct,e,c=decode(r,f['n'],v);an=anchor_view(r,f['n']);mn,mx=rects(r,f['n']);rad=r[3].cpu().numpy();fn=coarse_cache_audit if v=='coarse' else standard_cache_audit;rr[v]=fn(c,e,an,mn,mx,rad,f['w'],f['h'])
            with np.load(OUT/(v+'_'+f['name']+'.npz')) as z:assert np.array_equal(e,z['cache_errors']) and np.array_equal(c,z['double_coeff']) and np.array_equal(r[1].cpu().numpy(),z['native_rgb'])
            del r,run,native
        case.append(dict(name=f['name'],candidates=rr))
    print(json.dumps(dict(small_cases=len(case))),flush=True);work=workload();real=[]
    for ci,cam in enumerate(plan['cameras']):
        r=work.forward(mods['oracle'],ci);torch.cuda.synchronize();_,old_errors,C=decode(r,work.n,'oracle');an=anchor_view(r,work.n);mn,mx=rects(r,work.n);rad=r[3].cpu().numpy();vi=np.flatnonzero(rad>0);B=np.maximum(np.abs(16*mn[vi].astype(float)-an[vi]),np.abs(np.minimum(16*mx[vi].astype(float),plan['resolution'])-1-an[vi]));rank=np.array([int.from_bytes(hashlib.sha256(('v9cache/'+cam['img_name']+'/'+str(i)).encode()).digest()[:8],'big') for i in vi],dtype=np.uint64);selected=np.unique(np.r_[vi[np.argsort(rank,kind='stable')[:512]],vi[np.argsort(-(B[:,0]*B[:,1]),kind='stable')[:32]]]);rr={}
        with np.load(OUT/f'real_{ci}_oracle.npz') as z:assert np.array_equal(C,z['double_coeff']) and np.array_equal(r[1].cpu().numpy(),z['rgb']) and np.array_equal(r[2].cpu().numpy(),z['aux']) and np.array_equal(rad,z['radii'])
        for v in ['coarse','standard']:
            with np.load(OUT/f'real_{ci}_{v}.npz') as z:
                E=z['cache_errors'];assert np.array_equal(C,z['double_coeff']);rr[v]=(coarse_cache_audit if v=='coarse' else standard_cache_audit)(C,E,an,mn,mx,rad,*plan['resolution'],selected=selected)
                if v=='standard':assert np.array_equal(E[:,:10],old_errors)
        np.savez_compressed(directory/f'real_{ci}_inputs.npz',selected_triangles=selected,coeff=C[selected],anchors=an[selected],rect_min=mn[selected],rect_max=mx[selected],radii=rad[selected]);real.append(dict(camera=ci,name=cam['img_name'],candidates=rr,old_oracle_recapture_bitwise=True));print(json.dumps(dict(camera=ci,name=cam['img_name'],checked_edges={v:x['checked_edges'] for v,x in rr.items()})),flush=True);del r
    target.write_text(json.dumps(dict(small=case,real=real,candidate_gates={'coarse':True,'standard':True},script_sha256=sha(Path(__file__)),plan_sha256=sha(directory/'plan.json'),training=False,timing=False),indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'cache_audit_failure.txt').write_text(traceback.format_exc());raise
