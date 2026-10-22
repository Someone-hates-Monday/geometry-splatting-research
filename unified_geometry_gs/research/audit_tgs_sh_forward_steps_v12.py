"""Literal forward comparison of already rendered successive diagnostic versions."""
import json
from pathlib import Path
import numpy as np
from tgs_sh_isolation_common_v12 import OUT,sha,exact
def main():
    target=OUT/'forward_step_audit.json';assert not target.exists();render=json.loads((OUT/'render_results.json').read_text());rows=[]
    for case in render['comparisons']:
        base=OUT/case['scene']/case['cohort']/'main';pairs={}
        for left,right in [('archived_native','sh_only'),('sh_only','geometry_fixed'),('geometry_fixed','piecewise'),('piecewise','coarse')]:
            fields={}
            for key in ['rgb','aux','radii','maxblend','contributors']:
                a=np.load(base/left/(key+'.npy'),mmap_mode='r');b=np.load(base/right/(key+'.npy'),mmap_mode='r');fields[key]=dict(literal_equal=exact(a,b),maximum_numeric_abs=float(np.abs(a.astype(float)-b.astype(float)).max(initial=0)))
            pairs[left+'_to_'+right]=fields
        rows.append(dict(scene=case['scene'],cohort=case['cohort'],camera=case['camera'],steps=pairs))
    equal_mid=all(all(f['literal_equal'] for f in r['steps'][name].values()) for r in rows for name in ['sh_only_to_geometry_fixed','geometry_fixed_to_piecewise']);target.write_text(json.dumps(dict(rows=rows,all_mid_step_forwards_literal_equal=equal_mid,render_results_sha256=sha(OUT/'render_results.json'),script_sha256=sha(Path(__file__)),scope='Post-render diagnostic of full field equality, not a new held-out test or changed gate.',training=False),indent=2)+'\n');print(json.dumps(dict(all_mid_step_forwards_literal_equal=equal_mid)),flush=True)
if __name__=='__main__':main()
