"""Structural influence helpers and simultaneous fixed midpoint edits."""
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,pack
OUT=ROOT/'experiments/tgs_joint_edit_v27'

def empty(r):return r[0]>=r[2] or r[1]>=r[3]
def expand(r,W,H):return [0,0,0,0] if empty(r) else [max(0,r[0]-5),max(0,r[1]-5),min(W,r[2]+5),min(H,r[3]+5)]
def overlaps(a,b):return not(empty(a) or empty(b)) and max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])
def union(a,b):
    rs=[r for r in [a,b] if not empty(r)]
    return [0,0,0,0] if not rs else [min(r[0] for r in rs),min(r[1] for r in rs),max(r[2] for r in rs),max(r[3] for r in rs)]
def paint(a,r,value):
    x0,y0,x1,y1=r;a[y0:y1,x0:x1]=value[y0:y1,x0:x1]
def round_bound(terms,n):
    eps=np.finfo(np.float64).eps;gamma=n*eps/(1-n*eps)
    return float((gamma+8*eps)*sum(float(np.abs(t).sum()) for t in terms)/n)

def split_pair(state,parents,children,colors,col):
    assert len(parents)==len(set(parents))==2
    keep=torch.ones(len(state['points']),device='cuda',dtype=torch.bool);keep[parents]=False
    result=pack(*[torch.cat([state[k][keep],children[0][k],children[1][k]]).contiguous() for k in ['points','sigma','opacity','sh']])
    result_colors=torch.cat([col[keep],colors[0],colors[1]]).contiguous()
    return result,result_colors,keep
