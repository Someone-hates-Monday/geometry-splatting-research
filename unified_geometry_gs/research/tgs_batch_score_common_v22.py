"""Prepared fixed-state proposal bank and exact PNG8 score helpers; no optimization."""
import json
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,sha,module,pack,colors
from tgs_aux_ablation_common_v21 import V20
OUT=ROOT/'experiments/tgs_batch_score_v22'

def backend():
    r=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(r['path']))==r['sha256'];return module(r['path'])

def child_only(state,index):
    a,b,c=state['points'][index];ab,ac,bc=(a+b)/2,(a+c)/2,(b+c)/2
    points=torch.stack([torch.stack([a,ab,ac]),torch.stack([b,ab,bc]),torch.stack([c,ac,bc]),torch.stack([ab,ac,bc])])
    return pack(points,*[state[k][index:index+1].repeat((4,)+(1,)*(state[k].ndim-1)) for k in ['sigma','opacity','sh']])

def bank(state,col,cam,rows,helper):
    small=[];cc=[]
    for row in rows:
        index=row['parent_id'];child=child_only(state,index);color=col[index:index+1].repeat(4,1).contiguous() if row['appearance']=='hold' else colors(helper,child,cam)
        small.append(child);cc.append(color)
    merged=pack(*[torch.cat([x[k] for x in small]).contiguous() for k in ['points','sigma','opacity','sh']])
    parents=torch.tensor([r['parent_id'] for r in rows],device='cuda',dtype=torch.int32)
    return merged,torch.cat(cc).contiguous(),parents,small,cc

def schedule(m,bg,cg,parents,W,H):
    boxes,counts=m.plan(bg,cg,parents,W,H)
    offsets=torch.cat([torch.zeros(1,device='cuda',dtype=torch.int64),counts.cumsum(0,dtype=torch.int64)])
    return boxes,offsets

def cpu_sse(a,gt):
    d=a.astype(np.int64)-gt.astype(np.int64);return int(np.sum(d*d,dtype=np.int64))
