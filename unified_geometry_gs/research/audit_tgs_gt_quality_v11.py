"""Independent PNG conversion, exact integer primary MSE, reversed-filter SSIM and patches."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import correlate1d
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_gt_quality_v11';SOURCE=ROOT/'sources/tgs_gt_identity_v11'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def independent_ssim(x,y):
    g=[math.exp(-.5*(i/1.5)**2) for i in range(-5,6)];g=np.array([v/math.fsum(g) for v in g],dtype=np.float64)
    def f(a):return correlate1d(correlate1d(a,g,axis=1,mode='constant',cval=0.),g,axis=0,mode='constant',cval=0.)
    mx,my=f(x),f(y);vx,vy=f(x*x)-mx**2,f(y*y)-my**2;cov=f(x*y)-mx*my;return ((2*mx*my+1e-4)*(2*cov+9e-4))/((mx**2+my**2+1e-4)*(vx+vy+9e-4)),g
def patch(x,y,cx,cy,c,g):
    h,w=x.shape[:2];terms=[]
    for j in range(-5,6):
        for i in range(-5,6):
            u,v=cx+i,cy+j;xx=float(x[v,u,c]) if 0<=u<w and 0<=v<h else 0.;yy=float(y[v,u,c]) if 0<=u<w and 0<=v<h else 0.;terms.append((float(g[i+5])*float(g[j+5]),xx,yy))
    mx=math.fsum(a*b for a,b,_ in terms);my=math.fsum(a*c for a,_,c in terms);vx=math.fsum(a*b*b for a,b,_ in terms)-mx*mx;vy=math.fsum(a*c*c for a,_,c in terms)-my*my;cov=math.fsum(a*b*c for a,b,c in terms)-mx*my;return ((2*mx*my+1e-4)*(2*cov+9e-4))/((mx*mx+my*my+1e-4)*(vx+vy+9e-4))
def main():
    target=OUT/'independent_quality_audit.json';assert not target.exists();results=json.loads((OUT/'quality_results.json').read_text());plan=json.loads((OUT/'quality_plan.json').read_text());identity=json.loads((OUT/'identity_audit.json').read_text());render=json.loads((OUT/'render_results.json').read_text());assert identity['all_gates'] and render['model_inputs_unchanged'] and results['plan_sha256']==sha(OUT/'quality_plan.json')
    for name,h in plan['scripts'].items():assert sha(ROOT/name)==h
    assert results['render_results_sha256']==sha(OUT/'render_results.json');verified=[];patches=0
    for row in results['rows']:
        scene,v=row['scene'],row['variant'];folder=OUT/scene/v;raw=np.load(folder/'rgb.npy');gt=np.load(OUT/scene/'gt_rgb.npy');photo=np.array(Image.open(SOURCE/scene/'original.JPG').convert('RGB'));published=np.array(Image.open(SOURCE/scene/'published_gt.png').convert('RGB'));assert np.array_equal(photo,gt) and np.array_equal(gt,published);assert np.isfinite(raw).all()
        converted=torch.from_numpy(raw.copy()).mul(255).add_(.5).clamp_(0,255).permute(1,2,0).to(torch.uint8).numpy();saved=np.load(folder/'png_rgb.npy');assert np.array_equal(converted,saved);d=converted.astype(np.int64)-gt.astype(np.int64);ssq=int(np.sum(d*d,dtype=np.int64));exact_primary=ssq/(gt.size*255**2);truth=gt.astype(np.float64)/255;values={}
        for kind,x in [('primary_PNG8',converted.astype(np.float64)/255),('secondary_clipped_float',np.minimum(np.maximum(raw.transpose(1,2,0),0),1).astype(np.float64))]:
            q=row[kind];chunks=[];flat=(x-truth).ravel()
            for start in range(0,len(flat),262139):
                a=flat[start:start+262139];chunks.append(float(np.einsum('i,i->',a,a)))
            mse=exact_primary if kind=='primary_PNG8' else math.fsum(chunks)/len(flat);psnr=-10*math.log10(mse) if mse else float('inf');assert math.isclose(mse,q['MSE'],rel_tol=1e-12,abs_tol=1e-16) and abs(psnr-q['PSNR_dB'])<=1e-10;ss,g=independent_ssim(x,truth);mean=float(np.mean(ss));assert abs(mean-q['SSIM'])<=1e-10;assert np.allclose(ss.mean((0,1)),q['SSIM_by_channel'],rtol=0,atol=1e-10);maximum=0.
            for s in q['SSIM_samples']:
                u,z=s['pixel']
                for c in range(3):
                    exact=patch(x,truth,u,z,c,g);maximum=max(maximum,abs(exact-s['values'][c]));patches+=1
            assert maximum<=1e-10;values[kind]=dict(MSE=mse,PSNR_dB=psnr,SSIM=mean,max_direct_patch_disagreement=maximum)
        rr=next(r for r in render['rows'] if r['scene']==scene and r['variant']==v)
        for key,h in rr['files'].items():assert sha(folder/(key+'.npy'))==h
        verified.append(dict(scene=scene,variant=v,exact_integer_PNG_squared_error=ssq,metrics=values));print(json.dumps(dict(scene=scene,variant=v,verified=True)),flush=True)
    for scene in ['flowers','room']:
        x=np.load(OUT/scene/'oracle/rgb.npy');y=np.load(OUT/scene/'coarse/rgb.npy');assert np.array_equal(x.view(np.uint8),y.view(np.uint8));a=next(r for r in results['rows'] if r['scene']==scene and r['variant']=='oracle');b=next(r for r in results['rows'] if r['scene']==scene and r['variant']=='coarse');assert a['primary_PNG8']==b['primary_PNG8'] and a['secondary_clipped_float']==b['secondary_clipped_float']
    output=dict(all_verified=True,rows=verified,direct_SSIM_patch_channel_checks=patches,full_SSIM_independent_filter_order_verified=True,PNG_conversion_bitwise_Torch=True,primary_MSE_exact_integer_verified=True,identity_redecoded_exactly=True,all_variants_retained=True,script_sha256=sha(Path(__file__)),results_sha256=sha(OUT/'quality_results.json'),plan_sha256=sha(OUT/'quality_plan.json'),training=False,scientific_goal_complete=False);target.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
