"""Readonly binaries, per-view inputs and defined legacy image prefixes for v12."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
from tgs_cross_scene_common_v10 import ROOT,CrossWorkload
from run_tgs_real_events import camera_tensors
OUT=ROOT/'experiments/tgs_sh_isolation_v12';SOURCE=ROOT/'sources/tgs_sh_isolation_v12'
VARIANTS=['archived_native','sh_only','geometry_fixed','piecewise','coarse']
RECEIPTS={
 'archived_native':('experiments/tgs_subdivision_v1/native_build/receipt_9d974f09b721.json','9d974f09b7217a026aaad3ebc48af16c6a9571fcd17a885711b0517b9d409da1'),
 'sh_only':('experiments/tgs_subdivision_v1/sh_direction_v1/fixed_build_receipt.json','4fa625a41b94545010a579e9c98eb68fb557023c2429be890eaf20d750b00ef1'),
 'geometry_fixed':('experiments/tgs_backward_v1/geometry_fixed/build_receipt.json','955dd08a6082063253e188cd7504ab78cbbad8b60411c0b5c1e26d56cbf662c2'),
 'piecewise':('experiments/tgs_branch_aux_v1/piecewise_build/build_receipt.json','f06b1a1ccf5c11f99cebda0b0b5338ca273186ba316aa74fa619c4fa45ccd852'),
 'coarse':('experiments/tgs_window_candidates_v9/coarse/build_receipt.json','81d18c3cef204e9a68198481aaf22792696eb331d772735d8802e1cd51b5557c')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def modules():
    result={}
    for v,(rel,h) in RECEIPTS.items():
        receipt=json.loads((ROOT/rel).read_text());p=Path(receipt['module_file']);assert sha(p)==receipt['module_sha256']==h;spec=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);result[v]=m
    return result
def workload(scene):
    plan=json.loads((OUT/'input_plan.json').read_text());identity=json.loads((OUT/'identity_audit.json').read_text());rows=[r for r in plan['rows'] if r['scene']==scene];w=CrossWorkload(scene);before=w.input_hashes();old=json.loads((ROOT/'experiments/tgs_cross_scene_v10/real_plan.json').read_text())['model_inputs'][scene]
    for key in ['points','sigma','opacity','sh','counts','offsets','rgb_up','depth_up']:assert before[key]==old[key]
    w.camera_metadata=[r['camera'] for r in rows];w.cameras=[]
    for r in rows:
        idrow=next(i for i in identity['rows'] if i['scene']==scene and i['cohort']==r['cohort']);h,width=idrow['shape'][:2];w.cameras.append(camera_tensors(r['camera'],width,h))
    return w,rows
def forward(work,mod,ci,degree=3,colors=None):
    c=work.cameras[ci];return mod.rasterize_triangles(work.bg,work.points,work.sigma,work.counts,work.offsets,work.empty if colors is None else colors,work.opacity,work.scaling,work.density,c['view'],c['projection'],work.n,c['tx'],c['ty'],c['h'],c['w'],work.sh if colors is None else work.empty,degree,c['center'],False,False)
def paths(res,w,h):
    raw=res[6].cpu().numpy();off=(12*w*h+127)//128*128;assert len(raw)>=off+8*w*h;return raw[off:off+8*w*h].view(np.uint32).reshape(2,h,w).copy()
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and bool(np.array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8)))
def capture(res,w,h):return dict(rgb=res[1].cpu().numpy(),aux=res[2].cpu().numpy(),radii=res[3].cpu().numpy(),maxblend=res[9].cpu().numpy(),contributors=paths(res,w,h))
