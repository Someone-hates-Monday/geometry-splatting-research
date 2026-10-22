"""Independent old-native baseline/VJP and exact new-cache audit, no optimizer."""
import hashlib,json,traceback
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
from tgs_compensated_common import ROOT,OUT,sha,align,modules,v5_oracle,workload,decode,state
from probe_tgs_adaptive_window import decode as decode_v5
from probe_tgs_stable_window import SafeNative,SafeAuxNative,signature
from probe_tgs_radius_contraction import check_group
def q(v):return F.from_float(float(v))
def anchor_view(r,n):
    pos=0
    for size in [4,3,4]:pos=align(pos)+size*n
    pos=align(pos);return r[4].narrow(0,pos,8*n).view(torch.float32).reshape(n,2).cpu().numpy()
def rects(r,n):
    raw=r[4].cpu().numpy();mx=len(raw)-128-8*n;mn=mx-align(8*n);return raw[mn:mn+8*n].view(np.uint32).reshape(n,2),raw[mx:mx+8*n].view(np.uint32).reshape(n,2)
def cache_audit(C,E,anchor,mn,mx,radii,w,h,selected=None):
    visible=radii>0;ids=np.flatnonzero(visible);B=np.maximum(np.abs(16*mn[visible].astype(float)-anchor[visible]),np.abs(np.minimum(16*mx[visible].astype(float),[w,h])-1-anchor[visible]));assert np.array_equal(B,E[visible,44:46])
    v=C[visible,:9];hi=v.astype(np.float32);lo=(v-hi.astype(float)).astype(np.float32);dom=np.isfinite(v).all(1)&np.isfinite(anchor[visible]).all(1)&(np.abs(anchor[visible])<2.**23).all(1)&(anchor[visible]==np.floor(anchor[visible])).all(1)&(B<=8192).all(1)&(mx[visible]>mn[visible]).all(1)&(w<=8192)&(h<=8192)&(C[visible,9]>=2.**-20)&(C[visible,9]<=2.**20)&np.isfinite(C[visible,9])
    for values in [v,hi,lo]:dom&=((values==0)|((np.abs(values)>=2.**-60)&(np.abs(values)<=2.**20))).all(1)
    assert np.array_equal(dom,E[visible,46]==1)
    if selected is None:selected=ids
    checked=0;skipped=0
    for i in selected:
        if not E[i,46]:skipped+=1;continue
        p=E[i,10:];a=p[:18];ca=p[18:34];bs=list(map(q,[*p[34:36],1]));cc=list(map(q,C[i]));ef=[]
        for k in range(3):
            indices=[2*k,2*k+1,6+k];M=Delta=M64=Dhi=F(0)
            for t,j in enumerate(indices):
                hh=np.float32(C[i,j]);ll=np.float32(C[i,j]-float(hh));assert a[6*k+t]==hh and a[6*k+3+t]==ll
                H,L=q(hh),q(ll);M+=(abs(H)+abs(L))*bs[t];Delta+=abs(cc[j]-H-L)*bs[t];M64+=abs(cc[j])*bs[t];Dhi+=abs(cc[j]-H)*bs[t]
            assert q(ca[4*k])>=M and q(ca[4*k+1])>=Delta and q(ca[4*k+2])>=F(1,2**51)*M64 and q(ca[4*k+3])>=Dhi+F(1,2**22)*M+F(1,2**51)*M64;ef.append(ca[4*k+3]);checked+=1
        assert ca[12]==max(ef) and ca[13]==np.float32(C[i,9]);assert q(ca[14])>=abs(cc[9]-q(ca[13])) and q(ca[15])>=cc[9] and p[37]==0
    return dict(visible=len(ids),domain=int(dom.sum()),checked_edges=checked,skipped_domain_triangles=skipped,selected_triangles=np.asarray(selected).tolist())
def main():
    target=OUT/'baseline_cache_audit.json';assert not target.exists();assert json.loads((OUT/'case_results.json').read_text())['all_gates'];assert json.loads((OUT/'real_results.json').read_text())['all_gates']
    plan=json.loads((OUT/'input_plan.json').read_text());mods=modules();old=v5_oracle();small=[];directory=OUT/'baseline';directory.mkdir();frozen=dict(script_sha256=sha(Path(__file__)),input_plan_sha256=sha(OUT/'input_plan.json'),old_module_sha256=sha(Path(old.__file__)),cache_selection='512 SHA256(v8cache/<camera>/<id>) ranks +32 footprint products; all small cases',training=False,timing=False);(directory/'plan.json').write_text(json.dumps(frozen,indent=2)+'\n')
    for e in plan['cases']:
        with np.load(ROOT/e['input']) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        f['pixel_grad']=f.get('patterns',np.array([f['pixel_grad']]))[0];native=(SafeAuxNative if f['kind']=='aux' else SafeNative)(mods['oracle'],f);forward=native.forward(f['theta']);r=forward[1];counts,errors,coeff=decode(r,f['n']);an=anchor_view(r,f['n']);mn,mx=rects(r,f['n']);rad=r[3].cpu().numpy();a=cache_audit(coeff,errors,an,mn,mx,rad,f['w'],f['h'])
        with np.load(ROOT/('experiments/tgs_phase_cost_v5/oracle_'+f['name']+'.npz')) as z:
            same={k:bool(np.array_equal(value,z[key])) for k,key,value in [('rgb','native_rgb',r[1].cpu().numpy()),('aux','native_aux',r[2].cpu().numpy()),('radii','native_radii',rad),('coeff','double_coeff',coeff),('old_errors','cache_errors',errors[:,:10])]};assert all(same.values())
        small.append(dict(name=f['name'],cache=a,v5_oracle_bitwise=same));del r,forward,native
    print(json.dumps(dict(small_cases=len(small),checked_edges=sum(r['cache']['checked_edges'] for r in small))),flush=True)
    work=workload();rgb_grad=work.grad.clone();dep=torch.ones_like(work.grad_others[0])/(rgb_grad.shape[1]*rgb_grad.shape[2]);real=[]
    for ci,cam in enumerate(plan['cameras']):
        r=work.forward(old,ci);torch.cuda.synchronize();co,er,C=decode_v5(r,work.n);anchor=anchor_view(r,work.n);mn,mx=rects(r,work.n);rad=r[3].cpu().numpy();visible=rad>0;vi=np.flatnonzero(visible);B=np.maximum(np.abs(16*mn[visible].astype(float)-anchor[visible]),np.abs(np.minimum(16*mx[visible].astype(float),plan['resolution'])-1-anchor[visible]));rank=np.array([int.from_bytes(hashlib.sha256(('v8cache/'+cam['img_name']+'/'+str(i)).encode()).digest()[:8],'big') for i in vi],dtype=np.uint64);selected=np.unique(np.r_[vi[np.argsort(rank,kind='stable')[:512]],vi[np.argsort(-(B[:,0]*B[:,1]),kind='stable')[:32]]])
        oldflags,oldpath,oldpref=__import__('probe_tgs_compositing_guard').state(r,dict(n=work.n,w=work.cameras[ci]['w'],h=work.cameras[ci]['h']));data=dict(rgb=r[1].cpu().numpy(),aux=r[2].cpu().numpy(),radii=rad,maxblend=r[9].cpu().numpy(),contributors=oldpath,audit_triangles=selected,anchors=anchor[selected],rect_min=mn[selected],rect_max=mx[selected]);groups={}
        with np.load(OUT/f'real_{ci}_oracle.npz') as z:
            cache=cache_audit(C,z['cache_errors'],anchor,mn,mx,rad,*plan['resolution'],selected=selected);assert np.array_equal(C,z['double_coeff']) and np.array_equal(er,z['cache_errors'][:,:10]);assert all(np.array_equal(data[k],z[k]) for k in ['rgb','aux','radii','maxblend','contributors'])
            for loss in ['rgb','depth']:
                work.grad.copy_(rgb_grad if loss=='rgb' else torch.zeros_like(rgb_grad));work.grad_others.zero_()
                if loss=='depth':work.grad_others[0].copy_(dep)
                g=work.backward(old,r,ci);torch.cuda.synchronize();gg=[]
                for j,t in enumerate(g):
                    value=t.cpu().numpy();assert np.isfinite(value).all();data[f'{loss}_gradient_{j}']=value;cmp=check_group(value.astype(float).ravel(),z[f'{loss}_gradient_{j}'].astype(float).ravel());assert cmp['relative']<=1e-5 if cmp['reference_l2']>1e-10 else cmp['absolute_max']<=1e-6;gg.append(cmp)
                groups[loss]=gg;del g
        np.savez_compressed(directory/f'real_{ci}_v5.npz',**data);row=dict(camera=ci,name=cam['img_name'],cache=cache,old_v5_forward_bitwise=True,oracle_VJP_groups=groups,gate=True);real.append(row);print(json.dumps(dict(camera=ci,name=cam['img_name'],checked_edges=cache['checked_edges'],max_relative=max(v['relative'] for vs in groups.values() for v in vs))),flush=True);del r,data
    output=dict(small_cases=small,real_rows=real,all_gates=True,script_sha256=sha(Path(__file__)),plan_sha256=sha(directory/'plan.json'),input_plan_sha256=sha(OUT/'input_plan.json'),training=False,timing=False);target.write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'baseline_cache_failure.txt').write_text(traceback.format_exc());raise
