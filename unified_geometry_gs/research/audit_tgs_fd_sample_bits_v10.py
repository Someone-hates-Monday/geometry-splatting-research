"""Literal IEEE64 bits of retained JSON sample values, preserving signed zero."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_cross_scene_v10';V9=ROOT/'experiments/tgs_window_candidates_v9'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'v9_FD_sample_bit_audit.json';assert not target.exists();path=V9/'fd_results.json';fd=json.loads(path.read_text());index={(r['variant'],r['name'],r['pattern'],r['direction'],r['step']):r for r in fd['records']};comparisons=0;changed=0
    for key,a in index.items():
        if key[0]!='coarse':continue
        b=index[('oracle',*key[1:])]
        for sample in ['plus_samples','minus_samples']:
            x=np.array(a[sample],dtype=np.float64);y=np.array(b[sample],dtype=np.float64);assert x.shape==y.shape;changed+=int(np.count_nonzero(x.view(np.uint64)!=y.view(np.uint64)));comparisons+=1
    assert comparisons==5184
    out=dict(FD_pairs=2592,sample_arrays=5184,differing_IEEE64_sample_values=changed,literal_bits_equal=changed==0,scope='Bits of finite JSON-decoded sample numbers; original binary32 values are exactly representable in binary64, and signed zero is preserved. This is not a claim that all finite-difference numeric errors satisfy1e-3.',fd_sha256=sha(path),script_sha256=sha(Path(__file__)),training=False);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
