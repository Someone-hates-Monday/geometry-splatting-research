"""Separate initial idle telemetry from block-boundary samples, without filtering timings."""
import csv,json
from pathlib import Path
from tgs_cross_scene_common_v10 import OUT,sha
def numeric(t):
    if t['returncode']!=0:return {}
    lines=t['stdout'].splitlines();assert len(lines)==1;vals=next(csv.reader(lines));out={}
    for key,value in zip(t['fields'].split(','),vals):
        try:out[key]=float(value)
        except ValueError:pass
    return out
def ranges(rows):
    keys=['temperature.gpu','clocks.current.graphics','clocks.current.memory','memory.used','utilization.gpu'];return {key:dict(min=min(r[key] for r in rows if key in r),max=max(r[key] for r in rows if key in r)) for key in keys if any(key in r for r in rows)}
def main():
    cost=OUT/'cost';target=cost/'telemetry_summary.json';assert not target.exists();p=json.loads((cost/'plan.json').read_text());records=json.loads((cost/'telemetry.json').read_text());assert len(records)==72;allrows=[];by={s:[] for s in p['scenes']}
    for record in records:
        scene=p['blocks'][record['block']]['scene'];sample=[numeric(record['before']),numeric(record['after'])];allrows.extend(sample);by[scene].extend(sample)
    out=dict(initial_before_scene_workload=numeric(p['telemetry_start']),all_block_boundaries=ranges(allrows),by_scene={s:ranges(rs) for s,rs in by.items()},boundary_sample_count=len(allrows),plan_sha256=sha(cost/'plan.json'),telemetry_sha256=sha(cost/'telemetry.json'),script_sha256=sha(Path(__file__)),scope='Before/after block observations, not continuous in-call clock measurement. Initial idle snapshot kept separately; no timing sample is removed or recalibrated.')
    target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
