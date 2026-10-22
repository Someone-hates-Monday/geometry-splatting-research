"""CPU derivative: independent metrics/order checks after all fixed CUDA gates."""
import json,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
from tgs_split_color_common_v14 import ROOT,sha,write_new
from tgs_image_metrics_v11 import png8,score,gaussian
OUT=ROOT/'experiments/tgs_ray_order_v16a'

def independent(pred,gt,primary):
    residual=(pred-gt).ravel();mse=float(np.dot(residual,residual)/len(residual))
    assert abs(mse-primary['MSE'])<2e-15
    assert abs(-10*math.log10(mse)-primary['PSNR_dB'])<1e-10
    g=gaussian()
    def b(x):return convolve1d(convolve1d(x,g,axis=1,mode='constant'),g,axis=0,mode='constant')
    x,y=b(pred),b(gt);vx=b(pred**2)-x*x;vy=b(gt**2)-y*y;cov=b(pred*gt)-x*y
    ss=((2*x*y+.0001)*(2*cov+.0009))/((x*x+y*y+.0001)*(vx+vy+.0009))
    assert abs(float(ss.mean())-primary['SSIM'])<1e-11
    h,w=gt.shape[:2];kernel=np.outer(g,g)
    # Direct nonseparable convolution at corners, center, and interior validates padding.
    for yy,xx in [(0,0),(h-1,w-1),(h//2,w//2),(5,5)]:
        for ch in range(3):
            pp=np.pad(pred[:,:,ch],5);gg=np.pad(gt[:,:,ch],5)
            a=pp[yy:yy+11,xx:xx+11];c=gg[yy:yy+11,xx:xx+11]
            mx=float(np.sum(a*kernel));my=float(np.sum(c*kernel))
            vx=float(np.sum(a*a*kernel)-mx*mx);vy=float(np.sum(c*c*kernel)-my*my);cv=float(np.sum(a*c*kernel)-mx*my)
            value=((2*mx*my+.0001)*(2*cv+.0009))/((mx*mx+my*my+.0001)*(vx+vy+.0009))
            assert abs(value-ss[yy,xx,ch])<1e-10
    return dict(MSE=True,PSNR=True,SSIM=True,direct_patch_checks=12)

def main():
    result=json.loads((OUT/'results.json').read_text());assert result['all_correctness_gates'];packets=[];metrics=[]
    for scene in ['flowers','room']:
        folder=OUT/scene
        for p in sorted(folder.glob('packet_*.npz')):
            a=np.load(p);z=a['z'];o=a['ordinal'];rank=a['rank'];original=a['original'];n=len(z)
            assert np.array_equal(np.sort(rank),np.arange(n))
            assert np.array_equal(o,original[rank])
            assert np.array_equal(np.lexsort((o,z)),np.arange(n))
            assert np.isfinite(z).all() and bool(np.all(z>0))
            products=a['planes'][:,:3]*a['ray'];lhs=products.sum(1)*z;rhs=a['planes'][:,3]
            scale=(np.abs(products).sum(1)*np.abs(z)+np.abs(rhs))
            residual=float(np.max(np.abs(lhs-rhs)/np.maximum(scale,1e-300),initial=0))
            assert residual<64*np.finfo(float).eps
            packets.append(dict(file=str(p.relative_to(OUT)),hits=n,depth_residual=residual,passed=True))
        gt=np.asarray(Image.open(ROOT/'sources/tgs_gt_identity_v11'/scene/'published_gt.png').convert('RGB'),dtype=np.float64)/255
        for appearance in ['physical','legacy']:
            for order in ['native','ray']:
                path=folder/f'{appearance}_{order}.npy';raw=np.load(path);assert np.isfinite(raw).all()
                for mode,pred in [('PNG8',png8(raw).astype(np.float64)/255),('clipped_float',np.clip(raw.transpose(1,2,0).astype(float),0,1))]:
                    measured=score(pred,gt);audit=independent(pred,gt,measured)
                    metrics.append(dict(scene=scene,appearance=appearance,order=order,mode=mode,score=measured,audit=audit,source_sha256=sha(path)))
                Image.fromarray(png8(raw)).save(folder/f'{appearance}_{order}.png')
    deltas=[]
    for appearance in ['physical','legacy']:
        for scene in ['flowers','room']:
            a={r['order']:r['score'] for r in metrics if r['appearance']==appearance and r['scene']==scene and r['mode']=='PNG8'}
            deltas.append(dict(scene=scene,appearance=appearance,PSNR_delta_dB=a['ray']['PSNR_dB']-a['native']['PSNR_dB'],SSIM_delta=a['ray']['SSIM']-a['native']['SSIM']))
    leads={a:all(r['PSNR_delta_dB']>=.1 and r['SSIM_delta']>=0 for r in deltas if r['appearance']==a) for a in ['physical','legacy']}
    write_new(OUT/'quality_audit.json',dict(all_passed=True,packets=packets,metrics=metrics,deltas=deltas,descriptive_engineering_lead=leads,training=False,independent_confirmation=False))
    print(json.dumps(dict(deltas=deltas,leads=leads)),flush=True)

if __name__=='__main__':main()
