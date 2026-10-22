"""Trusted prepared-ROI adapter for FP64 image-loss module, not renderer integration."""
import json
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,sha,module
from tgs_image_metrics_v11 import gaussian
OUT=ROOT/'experiments/tgs_local_image_loss_v24'

def layout(rois,W,H):
    assert W>0 and H>0 and 3*W*H<=np.iinfo(np.int32).max and len(rois)>0
    boxes=[];prefix=[[0] for _ in range(5)]
    for r in rois:
        x0,y0,x1,y1=map(int,r);assert 0<=x0<=x1<=W and 0<=y0<=y1<=H
        if x0==x1 or y0==y1:ax0=ay0=ax1=ay1=jx0=jx1=0
        else:ax0,ay0,ax1,ay1=max(0,x0-5),max(0,y0-5),min(W,x1+5),min(H,y1+5);jx0,jx1=max(0,ax0-5),min(W,ax1+5)
        boxes.append([x0,y0,x1,y1,ax0,ay0,ax1,ay1,jx0,jx1]);rp=3*(x1-x0)*(y1-y0);vi=3*(ay1-ay0)*(jx1-jx0);ai=3*(ay1-ay0)*(ax1-ax0)
        for p,count in zip(prefix,[rp,vi,(vi+255)//256,ai,(ai+255)//256]):p.append(p[-1]+count)
    return dict(boxes=np.array(boxes,dtype=np.int32),prefix=[np.array(p,dtype=np.int64) for p in prefix],totals=[p[-1] for p in prefix],W=W,H=H)

def device_layout(host):
    return dict(boxes=torch.from_numpy(host['boxes']).cuda(),prefix=[torch.from_numpy(p).cuda() for p in host['prefix']],totals=list(map(int,host['totals'])),
        weights=torch.from_numpy(gaussian()).cuda(),host=host)

def packed_patches(patches):
    return torch.from_numpy(np.concatenate([np.ascontiguousarray(x).reshape(-1) for x in patches])).cuda()

def evaluate(m,base,gt,patches,d,debug=False):
    assert base.is_contiguous() and gt.is_contiguous() and base.shape==gt.shape and tuple(base.shape)==(3,d['host']['H'],d['host']['W'])
    assert patches.numel()==d['totals'][0]
    return m.evaluate(base,gt,patches,d['boxes'],*d['prefix'],d['weights'],d['totals'],debug)

def backend():
    b=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(b['path']))==b['sha256'];return module(b['path'])
