"""Audit trace endpoints, first branch divergence and omitted depth contribution."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compositing_trace_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'independent_audit.json';assert not target.exists()
    result=json.loads((OUT/'results.json').read_text());assert result['all_endpoints_bitwise_equal']
    receipt=json.loads((OUT/'build_receipt.json').read_text());assert sha(OUT/'build'/Path(receipt['module_file']).name)==receipt['module_sha256']
    assert receipt['source_sha256']==sha(OUT/'build/trace.cu') and receipt['helper_sha256']==sha(ROOT/'tgs_adaptive_window.cuh')
    rows=[]
    for ci in range(2):
        arrays={}
        for v in ['adaptive','oracle64']:
            with np.load(OUT/f'camera_{ci}_{v}.npz') as z:arrays[v]={k:z[k].copy() for k in z.files}
        a,b=arrays['adaptive'],arrays['oracle64'];assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['packed'],b['packed']) and np.array_equal(a['double_coeff'],b['double_coeff'])
        ta,tb=a['trace'],b['trace'];valid=np.isfinite(ta[:,0])&np.isfinite(tb[:,0]);different=np.flatnonzero(valid&(ta[:,13]!=tb[:,13]))
        assert len(different)==1;index=int(different[0]);aa=ta[index];bb=tb[index]
        assert aa[13]==1 and bb[12]==1 and aa[11]<np.float32(1e-4)<=bb[11]
        lasta=np.flatnonzero(ta[:,12]==1);lastb=np.flatnonzero(tb[:,12]==1)
        onlyb=np.setdiff1d(lastb,lasta);assert np.array_equal(onlyb,np.array([index]))
        extra=float(tb[index,15]*tb[index,21]);actual=float(b['native'][3]-a['native'][3])
        shared=lasta;common_difference=float(np.sum(tb[shared,15]*tb[shared,21]-ta[shared,15]*ta[shared,21]))
        residual=actual-extra-common_difference
        replay_a=a['summary'][[0,1,2,3,4,5,6,8]];replay_b=b['summary'][[0,1,2,3,4,5,6,8]]
        assert np.array_equal(replay_a.astype(np.float32),a['native'].astype(np.float32)) and np.array_equal(replay_b.astype(np.float32),b['native'].astype(np.float32))
        rows.append(dict(camera=ci,first_stop_disagreement_ordinal=index+1,triangle_id=int(aa[0]),depth=float(aa[15]),
            adaptive=dict(incoming_T=float(aa[10]),alpha=float(aa[5]),test_T=float(aa[11]),stop=True),
            oracle=dict(incoming_T=float(bb[10]),alpha=float(bb[5]),test_T=float(bb[11]),included=True),
            added_oracle_depth_weight=extra,native_depth_difference=actual,common_prefix_weight_difference=common_difference,float_accumulation_residual=residual,
            same_median_ordinal=bool(a['summary'][8]==b['summary'][8]),same_coefficients=True,endpoint_bitwise_equal=True))
    target.write_text(json.dumps(dict(rows=rows,source_results_sha256=sha(OUT/'results.json'),script_sha256=sha(Path(__file__)),causal_scope='Same sorted IDs/cache and bitwise endpoint replay; first different stop decision omits one reference contribution. Prefix differences and float accumulation residual separately retained.',training=False),indent=2)+'\n')
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
