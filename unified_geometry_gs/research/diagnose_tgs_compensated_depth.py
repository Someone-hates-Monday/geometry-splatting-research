"""Freeze every real AOV violation and exact pixels before causal CUDA tracing."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compensated_renderer_v8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'depth_failure_plan.json';assert not target.exists();real=json.loads((OUT/'real_results.json').read_text());assert not real['all_gates'];rows=[]
    for ci,entry in enumerate(real['rows']):
        gp=OUT/f'real_{ci}_guard.npz';op=OUT/f'real_{ci}_oracle.npz'
        with np.load(gp) as a,np.load(op) as b:
            d=a['aux'].astype(float)-b['aux'].astype(float);bad=np.argwhere(np.abs(d)>3e-5);pixels=[]
            for ch,y,x in bad:
                pixels.append(dict(channel=int(ch),pixel=[int(x),int(y)],guard=float(a['aux'][ch,y,x]),oracle=float(b['aux'][ch,y,x]),difference=float(d[ch,y,x]),flag=int(a['flags'][y,x]),contributors_guard=a['contributors'][:,y,x].tolist(),contributors_oracle=b['contributors'][:,y,x].tolist(),RGB_guard=a['rgb'][:,y,x].tolist(),RGB_oracle=b['rgb'][:,y,x].tolist()))
            rows.append(dict(camera=ci,name=entry['image'],guard_sha256=sha(gp),oracle_sha256=sha(op),bad_count=len(bad),bad=pixels,all_paths_identical=bool(np.array_equal(a['contributors'],b['contributors'])),median_bitwise=bool(np.array_equal(a['aux'][5],b['aux'][5]))))
    result=dict(rows=rows,real_results_sha256=sha(OUT/'real_results.json'),script_sha256=sha(Path(__file__)),tolerance_unchanged=3e-5,performance_measurement_allowed=False,training=False)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
