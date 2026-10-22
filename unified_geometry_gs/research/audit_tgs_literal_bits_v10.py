"""Supplement numeric array_equal with actual byte equality, including signed zero."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_cross_scene_v10';V9=ROOT/'experiments/tgs_window_candidates_v9'
from tgs_cross_scene_common_v10 import sha

FORWARD=['rgb','aux','radii','maxblend','contributors'];PARITY=FORWARD+['flags','double_coeff','cache_errors']
def bits(a,b):
    shape=a.shape==b.shape and a.dtype==b.dtype
    if not shape:return dict(numeric_equal=False,literal_bits_equal=False,byte_differences=None)
    aa=np.ascontiguousarray(a).reshape(-1).view(np.uint8);bb=np.ascontiguousarray(b).reshape(-1).view(np.uint8);count=0
    for lo in range(0,len(aa),16*1024*1024):count+=int(np.count_nonzero(aa[lo:lo+16*1024*1024]!=bb[lo:lo+16*1024*1024]))
    return dict(numeric_equal=bool(np.array_equal(a,b)),literal_bits_equal=count==0,byte_differences=count)

def v10():
    result=json.loads((OUT/'real_results.json').read_text());rows=[]
    for row in result['rows']:
        directory=OUT/row['scene']/f"camera_{row['camera']}";comparisons={}
        for v,ref,keys in [('coarse','oracle',FORWARD+['double_coeff']),('coarse_diag','coarse',PARITY),('oracle_diag','oracle',PARITY)]:
            q={k:bits(np.load(directory/v/(k+'.npy'),mmap_mode='r'),np.load(directory/ref/(k+'.npy'),mmap_mode='r')) for k in keys};comparisons[v]=q
        rows.append(dict(scene=row['scene'],camera=row['camera'],name=row['name'],comparisons=comparisons));print(json.dumps(dict(scope='v10',scene=row['scene'],camera=row['camera'])),flush=True)
    return rows

def v9():
    rows=[];plan=json.loads((V9/'input_plan.json').read_text());mapping={'rgb':'native_rgb','aux':'native_aux','radii':'native_radii'}
    for en in plan['cases']:
        q={}
        for v,ref,keys in [('coarse','oracle',FORWARD+['double_coeff']),('coarse_diag','coarse',PARITY),('oracle_diag','oracle',PARITY)]:
            with np.load(V9/(v+'_'+en['name']+'.npz')) as a,np.load(V9/(ref+'_'+en['name']+'.npz')) as b:q[v]={k:bits(a[mapping.get(k,k)],b[mapping.get(k,k)]) for k in keys}
        rows.append(dict(family='case',name=en['name'],comparisons=q))
    for ci,row in enumerate(json.loads((V9/'real_results.json').read_text())['rows']):
        q={}
        for v,ref,keys in [('coarse','oracle',FORWARD+['double_coeff']),('coarse_diag','coarse',PARITY),('oracle_diag','oracle',PARITY)]:
            with np.load(V9/f'real_{ci}_{v}.npz') as a,np.load(V9/f'real_{ci}_{ref}.npz') as b:q[v]={k:bits(a[k],b[k]) for k in keys}
        rows.append(dict(family='real',name=row['name'],comparisons=q));print(json.dumps(dict(scope='v9',camera=ci)),flush=True)
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['v9','v10'],required=True);args=parser.parse_args();target=OUT/('literal_bit_audit.json' if args.scope=='v10' else 'v9_literal_bit_audit.json');assert not target.exists();freeze=OUT/'literal_bit_audit_plan.json';assert json.loads(freeze.read_text())['script_sha256']==sha(Path(__file__));rows=v10() if args.scope=='v10' else v9();comparisons=[r for row in rows for q in row['comparisons'].values() for r in q.values()];out=dict(scope=args.scope,rows=rows,literal_bits_equal=all(r['literal_bits_equal'] for r in comparisons),numeric_equal=all(r['numeric_equal'] for r in comparisons),byte_differences=sum(r['byte_differences'] or 0 for r in comparisons),fields_compared=len(comparisons),script_sha256=sha(Path(__file__)),plan_sha256=sha(freeze),reason='np.array_equal alone equates +0 and -0. This supplementary check compares actual stored bytes, without changing rendering, thresholds or old sealed evidence.',training=False);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
