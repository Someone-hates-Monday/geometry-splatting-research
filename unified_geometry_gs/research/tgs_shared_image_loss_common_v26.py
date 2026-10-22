"""Shared cache must be rebuilt whenever baseline/GT/camera/weights change."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,module
from tgs_packed_query_common_v25 import dynamic_layout,emit
OUT=ROOT/'experiments/tgs_shared_loss_v26'

def backend():
    b=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(b['path']))==b['sha256'];return module(b['path'])

def evaluate(m,base,gt,patch,d,shared,debug=False):
    assert patch.numel()==d['totals'][0]
    return m.evaluate(base,gt,patch,shared,d['boxes'],*d['prefix'],d['weights'],d['totals'],debug)

def query(emitter,losser,planner,bg,br,bl,col,cg,cc,parents,base,gt,shared):
    H,W=base.shape[1:];boxes,offsets,d,_=dynamic_layout(planner,bg,cg,parents,W,H);patch=emit(emitter,bg,br,bl,col,cg,cc,parents,boxes,offsets,base,d)
    out=evaluate(losser,base,gt,patch[0],d,shared,False);return out[0],[patch[1],out[1]]
