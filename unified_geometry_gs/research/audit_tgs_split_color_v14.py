"""CPU reconstruction of every saved full output, native controls, midpoint and SH intervention."""
import hashlib,heapq,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.special import sph_harm_y
from tgs_split_color_common_v14 import ROOT,OUT,sha,ahash,write_new,exact

def basis(d):
    d=np.asarray(d,dtype=float);d=d/np.linalg.norm(d,axis=-1,keepdims=True)
    theta=np.arccos(np.clip(d[:,2],-1,1));phi=np.arctan2(d[:,1],d[:,0]);cols=[]
    for l in range(4):
        for m in range(-l,l+1):
            v=sph_harm_y(l,abs(m),theta,phi)
            cols.append(v.real if m==0 else math.sqrt(2)*(v.imag if m<0 else v.real))
    return np.stack(cols,axis=-1)

def checkhash(array,record):
    assert list(array.shape)==record['shape'] and str(array.dtype)==record['dtype']
    assert ahash(array)==record['sha256']

def independent_metrics(g,a,d,h,w):
    denominator=3*h*w;out={}
    for name,delta in [('noncolor',g),('appearance',a),('total',d)]:
        squared=float(np.einsum('ijk,ijk->',delta,delta));absolute=np.abs(delta)
        out[name]=dict(RMS=math.sqrt(squared/denominator),MAE=float(absolute.sum()/denominator),
            peak=float(absolute.max(initial=0)),pixels_above_3e_minus5=int(np.count_nonzero(np.any(absolute>3e-5,axis=0))),
            pixels_nonzero=int(np.count_nonzero(np.any(delta!=0,axis=0))))
    cross=2*float(np.einsum('ijk,ijk->',g,a))/denominator
    out['signed_energy_cross_term']=cross
    out['energy_identity_residual']=out['total']['RMS']**2-(out['noncolor']['RMS']**2+out['appearance']['RMS']**2+cross)
    out['A_over_G_RMS']=out['appearance']['RMS']/out['noncolor']['RMS'] if out['noncolor']['RMS']>1e-14 else None
    out['appearance_material']=out['appearance']['RMS']>out['noncolor']['RMS'] and out['appearance']['peak']>3e-5
    return out

def close(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:close(a[k],b[k])
    elif a is None:assert b is None
    elif isinstance(a,(int,bool)):assert a==b
    else:assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12+1e-10*abs(a)

def main():
    target=OUT/'independent_audit.json';assert not target.exists()
    result=json.loads((OUT/'results.json').read_text());plan=json.loads((OUT/'plan.json').read_text())
    assert result['all_isolation_gates'] and result['model_inputs_unchanged'] and result['forwards']==76
    assert result['plan_sha256']==sha(OUT/'plan.json') and len(result['cases'])==32
    for name,h in plan['scripts'].items():assert sha(ROOT/name)==h
    assert result['script_sha256']==sha(ROOT/'render_tgs_split_color_v14.py')
    verified=0;groups=[];energymax=0.;colormax=0.
    for model in plan['models']:
        scene=model['scene'];base=OUT/scene;assert sha(ROOT/model['path'])==model['sha256']
        state=torch.load(ROOT/model['path'],map_location='cpu',weights_only=True)
        opacity=torch.sigmoid(state['opacity'].detach()).numpy().reshape(-1)
        selection=json.loads((base/'selection.json').read_text());eligible=np.load(base/'eligible_ids.npy')
        baselines=[{key:np.load(base/f'baseline_{ci}'/(key+'.npy'),mmap_mode='r') for key in ['rgb','aux','radii','maxblend','contributors']} for ci in range(2)]
        expected=np.flatnonzero((np.maximum(baselines[0]['radii'],baselines[1]['radii'])>24)&(opacity>=.01))
        assert exact(eligible,expected) and ahash(eligible)==selection['eligible_sha256']
        key=lambda i:hashlib.sha256(f'tgs_split_color_v14/{scene}/{int(i)}'.encode()).hexdigest()
        assert selection['chosen_ids']==[int(i) for i in heapq.nsmallest(8,eligible,key=key)]
        for ci,b in enumerate(baselines):
            receipt=json.loads((base/f'baseline_{ci}/receipt.json').read_text())
            assert all(receipt['repeat_literal'].values()) and all(receipt['native_SH_only_noncolor_literal'].values())
            assert receipt['native_SH_only_RGB_maxabs']<=2e-4
            for name,value in b.items():checkhash(value,receipt['arrays'][name])
            assert ahash(np.load(base/f'baseline_{ci}/colors.npy',mmap_mode='r'))==receipt['color_sha256']
        for ei,index in enumerate(selection['chosen_ids']):
            inp=np.load(base/f'event_{ei}/input.npz');point=state['triangles_points'][index].detach().numpy()
            sh=torch.cat([state['features_dc'][index],state['features_rest'][index]],0).detach().numpy()
            assert exact(point,inp['parent_points']) and exact(sh,inp['SH'])
            x,y,z=point;xy=(x+y)/2;xz=(x+z)/2;yz=(y+z)/2
            child=np.array([[x,xy,xz],[y,xy,yz],[z,xz,yz],[xy,xz,yz]])
            assert exact(child,inp['child_points']) and index==int(inp['parent_id'])
            assert abs(float(inp['sigma'])-float((.01+torch.exp(state['sigma'][index].detach())).item()))<1e-6
            assert abs(float(inp['opacity'])-float(opacity[index]))<1e-7
            for ci,b in enumerate(baselines):
                folder=base/f'event_{ei}/camera_{ci}';receipt=json.loads((folder/'receipt.json').read_text())
                row=next(r for r in result['cases'] if r['scene']==scene and r['event']==ei and r['camera']==ci)
                assert receipt==row and all(row['split_noncolor_literal'].values())
                raw=np.load(folder/'changes.npz');x0,y0,x1,y1=raw['bbox'];h,w=raw['shape']
                assert b['rgb'].shape==(3,h,w) and 0<=x0<=x1<=w and 0<=y0<=y1<=h
                for name in ['rgb','aux']:assert exact(raw['before_'+name],b[name][:,y0:y1,x0:x1])
                nc=np.load(folder/'noncolor.npz');arrays={}
                for v in ['hold','moved']:
                    arrays[v]={}
                    for name in ['rgb','aux']:
                        value=b[name].copy();value[:,y0:y1,x0:x1]=raw[v+'_'+name];arrays[v][name]=value
                    for name in ['radii','maxblend','contributors']:arrays[v][name]=nc[v+'_'+name]
                    for name,value in arrays[v].items():checkhash(value,row[v+'_hashes'][name])
                for name in ['aux','radii','maxblend','contributors']:assert exact(arrays['hold'][name],arrays['moved'][name])
                assert row['hold_tiles']==row['moved_tiles']
                # Reconstruct FULL images, not just a small ROI denominator.
                G=arrays['hold']['rgb'].astype(float)-b['rgb'].astype(float)
                A=arrays['moved']['rgb'].astype(float)-arrays['hold']['rgb'].astype(float)
                D=arrays['moved']['rgb'].astype(float)-b['rgb'].astype(float)
                assert np.max(np.abs(D-G-A),initial=0)<=1e-13
                m=independent_metrics(G,A,D,int(h),int(w));close(m,row['metrics']);energymax=max(energymax,abs(m['energy_identity_residual']))
                col=np.load(folder/'colors.npz');cam=np.load(base/f'baseline_{ci}/camera.npz')
                center=cam['center'].astype(float);directions=np.vstack([point.astype(float).mean(0),child.astype(float).mean(1)])-center
                reference=np.maximum(basis(directions)@sh.astype(float)+.5,0)
                errors=[float(np.max(np.abs(col['parent']-reference[0]))),float(np.max(np.abs(col['moved']-reference[1:])))];colormax=max(colormax,*errors)
                assert max(errors)<=2e-5+2e-6*float(np.max(np.abs(reference)))
                assert exact(col['hold'],np.repeat(col['parent'][None],4,axis=0))
                allcolors=np.load(base/f'baseline_{ci}/colors.npy',mmap_mode='r');assert exact(col['parent'],allcolors[index])
                assert int(b['radii'][index])==row['parent_radii'] and nc['hold_radii'][-4:].tolist()==row['child_radii']
                groups.append(dict(scene=scene,event=ei,camera=ci,parent_id=index,metrics=m,
                    visible_parent=row['parent_radii']>0,visible_children=sum(r>0 for r in row['child_radii'])))
                verified+=1
        del state
        print(json.dumps(dict(scene=scene,cases_verified=verified)),flush=True)
    assert verified==32
    summary=[]
    for scene in ['flowers','room']:
        rs=[r for r in groups if r['scene']==scene];ratios=[r['metrics']['A_over_G_RMS'] for r in rs if r['metrics']['A_over_G_RMS'] is not None]
        summary.append(dict(scene=scene,cases=len(rs),appearance_material_count=sum(r['metrics']['appearance_material'] for r in rs),
            G_zero_or_tiny_count=sum(r['metrics']['noncolor']['RMS']<=1e-14 for r in rs),
            A_above_3e5_case_count=sum(r['metrics']['appearance']['peak']>3e-5 for r in rs),
            G_above_3e5_case_count=sum(r['metrics']['noncolor']['peak']>3e-5 for r in rs),
            holding_parent_color_larger_total_RMS_count=sum(r['metrics']['noncolor']['RMS']>r['metrics']['total']['RMS']+1e-14 for r in rs),
            median_A_over_G_RMS=float(np.median(ratios)) if ratios else None,
            maximum_full_image_MAE={name:max(r['metrics'][name]['MAE'] for r in rs) for name in ['appearance','noncolor','total']},
            maximum_peak={name:max(r['metrics'][name]['peak'] for r in rs) for name in ['appearance','noncolor','total']}))
    write_new(target,dict(all_verified=True,cases=verified,groups=groups,summary=summary,
        maximum_independent_child_color_disagreement=colormax,maximum_energy_identity_residual=energymax,
        results_sha256=sha(OUT/'results.json'),plan_sha256=sha(OUT/'plan.json'),script_sha256=sha(Path(__file__)),
        full_arrays_reconstructed_and_hashed=True,independent_GPU_replay=False,training=False,real_quality_gain=False))
    print(json.dumps(dict(all_verified=True,cases=verified,summary=summary)),flush=True)

if __name__=='__main__':main()
