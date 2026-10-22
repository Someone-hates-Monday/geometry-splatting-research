"""Execute all preserved precision gates and native tile membership checks."""
import hashlib
import importlib.util
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from tgs_mixed_reference import ROOT,Native,unpack,reference as rgb_reference
from tgs_aux_reference import AuxNative,reference as aux_reference
from tgs_mixed_numpy import evaluate as rgb_numpy,complex_gradient as rgb_complex
from tgs_aux_numpy import evaluate as aux_numpy,complex_gradient as aux_complex
from probe_tgs_radius_contraction import check_group

OUT=ROOT/'experiments/tgs_stable_window_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class SafeNative(Native):
    def forward(self,theta):
        f,t=self.f,self.t
        p,sigma,opacity,sh=unpack(theta,f)
        p,sigma,opacity=t(p),t(sigma),t(opacity);sh=t(sh) if sh is not None else self.empty
        scaling=t(np.zeros(f['n']));density=t(np.zeros(f['n']))
        res=self.mod.rasterize_triangles(self.bg,p,sigma,self.counts,self.offsets,self.colors,opacity,
            scaling,density,self.view,self.proj,f['n'],float(f['tx']),float(f['ty']),f['h'],f['w'],sh,
            max(f['degree'],0),self.campos,False,True)
        torch.cuda.synchronize()
        samples=np.array([res[1][:,y,x].cpu().numpy() for x,y in f['pixels']])
        loss=float((samples.astype(float)*f['pixel_grad'].astype(float)).sum())
        return loss,res,p,sigma,sh,samples


class SafeAuxNative(AuxNative,SafeNative):pass


def rectangles(res,n,w,h):
    """Decode trailing uint2 fields from pinned GeometryState allocation layout.

    Torch allocation is 128-byte aligned; required<GeometryState> adds 128
    trailing bytes. rect_min and rect_max are the final two aligned fields.
    Assert tile instance sum as an independent structural calibration per call.
    """
    b=res[4];assert b.data_ptr()%128==0
    raw=b.cpu().numpy();end=len(raw)-128
    maxoff=end-8*n;minoff=maxoff-((8*n+127)//128)*128
    assert maxoff%128==minoff%128==0
    mx=raw[maxoff:maxoff+8*n].view(np.uint32).reshape(n,2).copy()
    mn=raw[minoff:minoff+8*n].view(np.uint32).reshape(n,2).copy()
    visible=res[3].cpu().numpy()>0;mn[~visible]=0;mx[~visible]=0
    grid=np.array([(w+15)//16,(h+15)//16])
    assert (mx>=mn).all() and (mx<=grid).all()
    count=int(np.prod(mx.astype(np.int64)-mn,axis=1).sum())
    assert count==int(res[0]),(count,int(res[0]),mn,mx)
    return mn,mx


def signature(res,f):
    mn,mx=rectangles(res,f['n'],f['w'],f['h']);tiles=f['pixels']//16
    members=((tiles[None]>=mn[:,None])&(tiles[None]<mx[:,None])).all(-1)
    alpha=np.array([float(res[2][1,y,x]) for x,y in f['pixels']])
    return dict(rect_min=mn.tolist(),rect_max=mx.tolist(),selected_tile_members=members.tolist(),alpha_support=(alpha>0).tolist())


def main():
    assert not (OUT/'results.json').exists() and not (OUT/'progress.json').exists()
    plan=json.loads((OUT/'input_plan.json').read_text());assert plan['protocol_sha256']==sha(ROOT/'TGS_STABLE_WINDOW_PROTOCOL_V2.md')
    mods={};receipts={}
    for variant in ['direct','contraction']:
        receipt=json.loads((OUT/variant/'build_receipt.json').read_text());path=Path(receipt['module_file'])
        assert sha(path)==receipt['module_sha256'] and receipt['protocol_sha256']==plan['protocol_sha256']
        spec=importlib.util.spec_from_file_location(path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        mods[variant]=m;receipts[variant]=receipt
    (OUT/'execution_plan.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),input_plan_sha256=sha(OUT/'input_plan.json'),modules={k:r['module_sha256'] for k,r in receipts.items()},training=False),indent=2)+'\n')
    cases=[(key,e) for key in ['old_cases','stress_cases','rational_cases','holdout_cases'] for e in plan[key]]
    records=[];started=time.time()
    for ci,(family,entry) in enumerate(cases):
        path=ROOT/entry['input'];assert sha(path)==entry['sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        ref=aux_reference if f['kind']=='aux' else rgb_reference
        np_ref=aux_numpy if f['kind']=='aux' else rgb_numpy
        np_grad=aux_complex if f['kind']=='aux' else rgb_complex
        cls=SafeAuxNative if f['kind']=='aux' else SafeNative
        patterns=f.get('patterns',np.array([f['pixel_grad']]))
        refs=[]
        for pattern in patterns:
            f['pixel_grad']=pattern;x=torch.tensor(f['theta'].astype(float),requires_grad=True)
            loss,samples,meta=ref(x,f);g=torch.autograd.grad(loss,x)[0].numpy();cs=np_grad(f['theta'],f)
            nl,ns,fp=np_ref(f['theta'].astype(float),f)
            assert fp==meta['fingerprint'] and abs(float(loss.detach())-nl)<1e-9
            assert np.linalg.norm(g-cs)<=1e-8*max(np.linalg.norm(g),1)
            refs.append((samples.detach().numpy(),g,meta,cs))
        images={};auxs={};radii={}
        for variant,mod in mods.items():
            f['pixel_grad']=patterns[0];native=cls(mod,f);data=native.forward(f['theta']);sig=signature(data[1],f)
            rows=[];gradients=[];fds=[]
            for pi,pattern in enumerate(patterns):
                f['pixel_grad']=pattern;native.grad.zero_()
                for (x,y),weight in zip(f['pixels'],pattern):native.grad[:,y,x]=native.t(weight)
                samples,exact,meta,cs=refs[pi];g,_=native.backward(data);gradients.append(g);n=f['n']
                groups=dict(geometry=check_group(g[:9*n],exact[:9*n]),sigma_opacity=check_group(g[9*n:11*n],exact[9*n:11*n]))
                if f['degree']>=0:groups['SH']=check_group(g[11*n:],exact[11*n:])
                special=float(g@f['special_direction']) if f['check']=='homothety_nullspace' else None
                vg=abs(special)<=1e-6 if special is not None else all(v['gate'] for v in groups.values())
                error=float(np.max(np.abs(data[-1]-samples)))
                rows.append(dict(pattern=pi,groups=groups,direction=special,forward_max_abs=error,native_gradient=g.tolist(),reference_gradient=exact.tolist(),complex_gradient=cs.tolist(),gate=bool(vg and error<=3e-5 and (data[1][3]>0).all())))
                if 'fd_steps' in f:
                    for hh in f['fd_steps']:
                        for di,d in enumerate(f['fd_directions'][pi]):
                            plus=(f['theta'].astype(float)+hh*d).astype(np.float32);minus=(f['theta'].astype(float)-hh*d).astype(np.float32)
                            lp,rp,_,_,_,sp=native.forward(plus);lm,rm,_,_,_,sm=native.forward(minus)
                            cp,_,fpp=np_ref(plus.astype(float),f);cm,_,fpm=np_ref(minus.astype(float),f)
                            ps,ms=signature(rp,f),signature(rm,f)
                            kernel_branch=fpp==fpm==meta['fingerprint'] and bool((rp[3]>0).all()) and bool((rm[3]>0).all())
                            tile=ps['selected_tile_members']==ms['selected_tile_members']==sig['selected_tile_members']
                            alpha=ps['alpha_support']==ms['alpha_support']==sig['alpha_support']
                            actual=(plus.astype(float)-minus.astype(float))/(2*hh)
                            fds.append(dict(pattern=pi,step=float(hh),direction=di,actual_direction=actual.tolist(),expected=float(exact@actual),native_fd=(lp-lm)/(2*hh),cpu_fd=float(cp-cm)/(2*hh),plus_samples=sp.tolist(),minus_samples=sm.tolist(),plus_signature=ps,minus_signature=ms,kernel_branch=bool(kernel_branch),tile_membership_stable=tile,alpha_support_stable=alpha,smooth=bool(kernel_branch and tile and alpha)))
            image=data[1][1].cpu().numpy();aux=data[1][2].cpu().numpy();rad=data[1][3].cpu().numpy()
            images[variant]=image;auxs[variant]=aux;radii[variant]=rad
            historical=None
            if family=='old_cases':
                old=ROOT/entry['old_output'];assert sha(old)==entry['old_output_sha256']
                with np.load(old) as z:historical=float(np.abs(image-z['native_rgb']).max())
            coverage=None
            if family=='holdout_cases':
                selection=ROOT/entry['selection'];assert sha(selection)==entry['selection_sha256']
                with np.load(selection) as z:pix=z['pixels'];valid=z['ideal_valid']
                mn,mx=rectangles(data[1],1,f['w'],f['h']);tilepix=pix//16
                binned=((tilepix>=mn[0])&(tilepix<mx[0])).all(-1)
                native_alpha=aux[1,pix[:,1],pix[:,0]]
                coverage=dict(ideal_valid_samples=int(valid.sum()),tile_false_negatives=int((valid&~binned).sum()),native_support_false_negatives=int((valid&(native_alpha<=0)).sum()))
            row=dict(name=f['name'],family=family,variant=variant,patterns=rows,base_signature=sig,finite_differences=fds,historical_rgb_max_delta=historical,coverage=coverage,gate=bool(all(r['gate'] for r in rows) and all(d['smooth'] for d in fds) and (coverage is None or (coverage['tile_false_negatives']==0 and coverage['native_support_false_negatives']==0))))
            records.append(row)
            np.savez_compressed(OUT/(variant+'_'+f['name']+'.npz'),native_rgb=image,native_aux=aux,native_radii=rad,native_samples=data[-1],native_gradients=np.array(gradients),reference_gradients=np.array([r[1] for r in refs]),rect_min=np.array(sig['rect_min']),rect_max=np.array(sig['rect_max']))
        same=np.array_equal(images['direct'],images['contraction']) and np.array_equal(auxs['direct'],auxs['contraction']) and np.array_equal(radii['direct'],radii['contraction'])
        for row in records[-2:]:row['whole_rgb_aux_radii_identical']=same;row['gate']=row['gate'] and same
        (OUT/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(dict(case=ci+1,total=len(cases),name=f['name'],direct_gate=records[-2]['gate'],contraction_gate=records[-1]['gate'],max_forward=max(r['forward_max_abs'] for r in records[-1]['patterns']),max_geometry=max(r['groups']['geometry']['relative'] for r in records[-1]['patterns']))),flush=True)
    final=dict(records=records,input_plan_sha256=sha(OUT/'input_plan.json'),execution_plan_sha256=sha(OUT/'execution_plan.json'),cases=len(cases),vjp_count=sum(len(r['patterns']) for r in records),fd_count=sum(len(r['finite_differences']) for r in records),failed={k:[r['name'] for r in records if r['variant']==k and not r['gate']] for k in mods},all_precision_support_gates=all(r['gate'] for r in records),seconds=time.time()-started,performance_measurement_allowed=False,training=False)
    (OUT/'results.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps({k:v for k,v in final.items() if k!='records'}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'execution_failure.txt').write_text(traceback.format_exc());raise
