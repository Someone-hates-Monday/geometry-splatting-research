"""Full-frame diagnostic figure. Common displayed difference scale; no crop or fit."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tgs_split_color_common_v14 import ROOT
OUT=ROOT/'experiments/tgs_ray_order_v16a'
fig,axes=plt.subplots(2,4,figsize=(15,6.3),layout='constrained')
for row,scene in enumerate(['flowers','room']):
    gt=np.asarray(Image.open(ROOT/'sources/tgs_gt_identity_v11'/scene/'published_gt.png').convert('RGB'))
    a=np.load(OUT/scene/'legacy_native.npy').transpose(1,2,0);b=np.load(OUT/scene/'legacy_ray.npy').transpose(1,2,0)
    for col,img in enumerate([gt,np.clip(a,0,1),np.clip(b,0,1)]):axes[row,col].imshow(img,interpolation='nearest')
    diff=np.max(np.abs(a.astype(float)-b.astype(float)),axis=2)
    im=axes[row,3].imshow(diff,vmin=0,vmax=.2,cmap='cividis',interpolation='nearest')
    for col,title in enumerate([scene+' GT','Legacy center order','Legacy ray-plane order','Max channel change']):
        axes[row,col].set_title(title,fontsize=11);axes[row,col].axis('off')
fig.colorbar(im,ax=axes[:,3],shrink=.65,label='Absolute raw RGB change (saturates at 0.2)')
fig.suptitle('Fixed checkpoints, full frames; order change does not consistently improve GT metrics',fontsize=12)
fig.savefig(OUT/'full_frame_comparison.png',dpi=150);fig.savefig(OUT/'full_frame_comparison.pdf');plt.close(fig)
