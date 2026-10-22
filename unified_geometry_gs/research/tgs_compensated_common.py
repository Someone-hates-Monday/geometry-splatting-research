"""v8 layout readers and frozen modules, separate from all sealed prior helpers."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compensated_renderer_v8';VARIANTS=['guard_diag','oracle_diag','guard','oracle']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def align(n):return (n+127)//128*128
def modules():
    d={}
    for v in VARIANTS:
        r=json.loads((OUT/v/'build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256'];s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);d[v]=m
    return d
def v5_oracle():
    r=json.loads((ROOT/'experiments/tgs_phase_cost_v5/oracle/build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256'];s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def workload():
    from benchmark_tgs_backward import Workload
    from run_tgs_real_events import camera_tensors
    work=Workload();plan=json.loads((OUT/'input_plan.json').read_text());w,h=plan['resolution'];work.camera_metadata=plan['cameras'];work.cameras=[camera_tensors(c,w,h) for c in plan['cameras']];work.metadata.update(camera_order=[c['img_name'] for c in plan['cameras']],v8_holdout=plan['cameras'][-1]['img_name']);return work
def offsets(rawlen,n):
    mx=rawlen-128-8*n;mn=mx-align(8*n);count=mn-align(12*n);err=count-align(192*n);coef=err-align(80*n);pre=coef-align(12*n);return mx,mn,count,err,coef,pre
def decode(res,n):
    raw=res[4].cpu().numpy();mx,mn,count,err,coef,pre=offsets(len(raw),n)
    counts=raw[count:count+12*n].view(np.uint32).reshape(n,3).copy();errors=raw[err:err+192*n].view(np.float32).reshape(n,48).copy();coeff=raw[coef:coef+80*n].view(np.float64).reshape(n,10).copy();visible=res[3].cpu().numpy()>0;errors[~visible]=0;coeff[~visible]=0;assert (counts[:,1:]<=counts[:,:1]).all();return counts,errors,coeff
def state(res,f):
    n,w,h=f['n'],f['w'],f['h'];hw=w*h;raw=res[6].cpu().numpy();flags=raw[-128-4*hw:-128].view(np.uint32).reshape(h,w).copy();off=align(12*hw);contrib=raw[off:off+8*hw].view(np.uint32).reshape(2,h,w).copy();geom=res[4].cpu().numpy();pre=offsets(len(geom),n)[-1];pref=geom[pre:pre+12*n].view(np.uint32).reshape(n,3).copy();assert (flags<=3).all();return flags,contrib,pref
