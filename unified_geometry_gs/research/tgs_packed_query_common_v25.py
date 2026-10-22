"""Full dynamic patch-layout/replay/loss integration; no proposal acceptance."""
import json
from pathlib import Path
import torch
from tgs_split_color_common_v14 import ROOT,sha,module
from tgs_batch_score_common_v22 import schedule
from tgs_local_image_loss_common_v24 import layout,device_layout,evaluate
OUT=ROOT/'experiments/tgs_packed_query_v25'

def backend():
    b=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(b['path']))==b['sha256'];return module(b['path'])

def dynamic_layout(planner,bg,cg,parents,W,H):
    boxes,offsets=schedule(planner,bg,cg,parents,W,H)
    host_boxes=boxes.cpu().tolist() # Measured sync/transfer in later pipeline timing.
    rois=[[x0*16,y0*16,min(W,x1*16),min(H,y1*16)] for x0,y0,x1,y1 in host_boxes]
    d=device_layout(layout(rois,W,H));return boxes,offsets,d,rois

def emit(m,bg,br,bl,col,cg,cc,parents,boxes,offsets,base,d):
    return m.emit(bg,br,bl,col,cg,cc,parents,boxes,offsets,d['boxes'],d['prefix'][0],base,d['host']['W'],d['host']['H'],d['totals'][0])

def query(emitter,losser,planner,bg,br,bl,col,cg,cc,parents,base,gt):
    H,W=base.shape[1:];boxes,offsets,d,_=dynamic_layout(planner,bg,cg,parents,W,H)
    patch=emit(emitter,bg,br,bl,col,cg,cc,parents,boxes,offsets,base,d)
    out=evaluate(losser,base,gt,patch[0],d,False)
    return out[0],[patch[1],out[1]]
