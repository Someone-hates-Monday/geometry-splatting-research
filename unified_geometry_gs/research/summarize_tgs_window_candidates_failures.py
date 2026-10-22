"""Post-diagnostic census: all failure pixels, unchanged threshold, no candidate tuning."""
import json
from pathlib import Path
import numpy as np
from tgs_window_candidates_common import ROOT,OUT,sha

def main():
    target=OUT/'failure_census.json';assert not target.exists();real=json.loads((OUT/'real_results.json').read_text());rows=[]
    for ci,rr in enumerate(real['rows']):
        with np.load(OUT/f'real_{ci}_oracle.npz') as z:base=z['aux'].copy()
        variants={}
        for v in ['coarse','standard']:
            with np.load(OUT/f'real_{ci}_{v}.npz') as z:
                delta=z['aux'].astype(float)-base.astype(float);bad=np.any(np.abs(delta)>3e-5,axis=0);ys,xs=np.nonzero(bad)
                points=[dict(pixel=[int(x),int(y)],aux_delta=delta[:,y,x].tolist(),flag=int(z['flags'][y,x])) for y,x in zip(ys,xs)]
            reg=rr['candidates'][v]['regression_14_pixels'];assert all(r['depth_abs']<=3e-5 and r['paths_equal'] for r in reg)
            variants[v]=dict(bad_pixel_count=len(points),bad_by_channel=(np.abs(delta)>3e-5).sum(axis=(1,2)).tolist(),bad_pixels=points,old_14_count=len(reg),old_14_max_abs=max((r['depth_abs'] for r in reg),default=0.),old_14_bitwise_count=sum(r['depth_abs']==0 for r in reg))
        rows.append(dict(camera=ci,name=rr['name'],candidates=variants))
    fd=json.loads((OUT/'fd_results.json').read_text());by={v:[] for v in ['coarse','standard','oracle']}
    for r in fd['records']:by[r['variant']].append(r)
    assert len(by['coarse'])==len(by['oracle'])==2592
    for a,b in zip(by['coarse'],by['oracle']):
        assert all(a[k]==b[k] for k in ['name','pattern','direction','step','plus_samples','minus_samples','native_fd','expected','smooth'])
    cache=json.loads((OUT/'cache_audit.json').read_text())
    sums={v:dict(small_edges=sum(r['candidates'][v]['checked_edges'] for r in cache['small']),real_edges=sum(r['candidates'][v]['checked_edges'] for r in cache['real'])) for v in ['coarse','standard']}
    out=dict(rows=rows,threshold=3e-5,all_coarse_FD_sample_endpoints_bitwise_oracle=True,FD_pairs_compared=2592,cache_exact_edge_counts=sums,script_sha256=sha(Path(__file__)),real_sha256=sha(OUT/'real_results.json'),fd_sha256=sha(OUT/'fd_results.json'),scope='Post-diagnostic failure census, not a new holdout or tuning experiment.',training=False)
    target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':main()
