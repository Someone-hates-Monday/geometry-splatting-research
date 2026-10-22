"""v9 candidate-specific layouts; existing v5 binaries are read-only references."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_window_candidates_v9'
BASES=['coarse','standard','oracle'];VARIANTS=['coarse_diag','standard_diag','oracle_diag','coarse','standard','oracle']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def align(n):return (n+127)//128*128
def stride(v):return 14 if v.startswith('coarse') else (48 if v.startswith('standard') else 10)
def modules():
    d={}
    for v in VARIANTS:
        root=ROOT/'experiments/tgs_phase_cost_v5' if v.startswith('oracle') else OUT
        r=json.loads((root/v/'build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256'];s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);d[v]=m
    return d
def offsets(length,n,v):
    mx=length-128-8*n;mn=mx-align(8*n);co=mn-align(12*n);er=co-align(4*stride(v)*n);ce=er-align(80*n);pre=ce-align(12*n);return mx,mn,co,er,ce,pre
def decode(r,n,v):
    raw=r[4].cpu().numpy();mx,mn,co,er,ce,pre=offsets(len(raw),n,v);s=stride(v);ct=raw[co:co+12*n].view(np.uint32).reshape(n,3).copy();e=raw[er:er+4*s*n].view(np.float32).reshape(n,s).copy();c=raw[ce:ce+80*n].view(np.float64).reshape(n,10).copy();vis=r[3].cpu().numpy()>0;e[~vis]=0;c[~vis]=0;assert (ct[:,1:]<=ct[:,:1]).all();return ct,e,c
def state(r,f,v):
    n,w,h=f['n'],f['w'],f['h'];hw=w*h;raw=r[6].cpu().numpy();flags=raw[-128-4*hw:-128].view(np.uint32).reshape(h,w).copy();off=align(12*hw);ids=raw[off:off+8*hw].view(np.uint32).reshape(2,h,w).copy();g=r[4].cpu().numpy();pre=offsets(len(g),n,v)[-1];pf=g[pre:pre+12*n].view(np.uint32).reshape(n,3).copy();assert (flags<=3).all();return flags,ids,pf
def workload():
    from benchmark_tgs_backward import Workload
    from run_tgs_real_events import camera_tensors
    work=Workload();p=json.loads((OUT/'input_plan.json').read_text());w,h=p['resolution'];work.camera_metadata=p['cameras'];work.cameras=[camera_tensors(c,w,h) for c in p['cameras']];work.metadata.update(camera_order=[c['img_name'] for c in p['cameras']],v9_holdout=p['cameras'][-1]['img_name']);return work
def strict_groups(groups):return all(g['relative']<=1e-5 if g['reference_l2']>1e-10 else g['absolute_max']<=1e-6 for g in groups)
