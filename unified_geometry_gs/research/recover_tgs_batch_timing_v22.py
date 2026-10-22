"""Recover complete raw cells after a final bookkeeping assertion, without GPU rerun."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_batch_score_v22/timing'

def main():
    p=json.loads((OUT/'plan.json').read_text());script=ROOT/'benchmark_tgs_batch_score_v22.py'
    assert sha(script)==p['inputs']['benchmark_tgs_batch_score_v22.py']
    failure=(OUT/'run_failure.txt').read_text();assert 'assert forwards==954 and len(rows)==8' in failure and 'AssertionError' in failure
    assert not (OUT/'results.json').exists();rows=[];files=[OUT/'plan.json',OUT/'run_failure.txt',script,Path(__file__)]
    for scene in ['flowers','room']:
        for K in p['sizes']:
            q=OUT/f'{scene}_K{K}.json';files.append(q);row=json.loads(q.read_text())
            assert row['scene']==scene and row['K']==K and len(row['blocks'])==8 and len(row['proposals'])==K
            assert [b['order'] for b in row['blocks']]==p['orders'][f'{scene}_K{K}'];rows.append(row)
    actual=2*(1+10*sum(3*K+1 for K in p['sizes']));assert actual==982 and p['full_forward_cap']==954
    write_new(OUT/'recovery_plan.json',dict(inputs={str(f.relative_to(ROOT)):sha(f) for f in files},
        original_run_exit_code=1,failure_stage='Final expected-call-count assertion after all cells and input hash checks',
        planned_forwards=954,derived_actual_forwards=actual,excess_over_written_count=actual-954,no_GPU_rerun=True))
    write_new(OUT/'results_recovered.json',dict(rows=rows,full_forwards=actual,original_run_exit_code=1,
        all8_raw_cells_present=True,setups_not_persisted=True,final_telemetry_not_persisted=True,training=False,
        recovery_scope='Raw measurements only; do not infer missing original setup/telemetry values'))
    print(json.dumps(dict(recovered_cells=8,actual_forwards=actual,original_run_exit_code=1,missing=['setups','final telemetry'])),flush=True)

if __name__=='__main__':main()
