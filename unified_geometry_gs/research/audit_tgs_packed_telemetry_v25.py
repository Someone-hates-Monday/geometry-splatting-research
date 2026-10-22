"""Read-only after-run telemetry addendum; no timing rerun or clock changes."""
import json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_packed_query_v25';TIM=OUT/'timing'
fields=['name','driver_version','clocks.max.sm','clocks.sm','pstate','utilization.gpu']
command=['/usr/lib/wsl/lib/nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits']
r=subprocess.run(command,capture_output=True,text=True,timeout=15)
start=json.loads((TIM/'telemetry_start.json').read_text());initial_sm=int(start['stdout'].split(',')[3].strip())
current=dict(zip(fields,[x.strip() for x in r.stdout.strip().split(',')])) if r.returncode==0 else {}
out=dict(observed_at_UTC=datetime.now(timezone.utc).isoformat(),main_start_SM_MHz=initial_sm,after_run_query=command,
    exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,parsed=current,
    preceding_Windows_probe=dict(command='nvidia-smi --query-gpu=name,clocks.max.sm,clocks.sm,pstate,utilization.gpu --format=csv,noheader,nounits',
        stdout='NVIDIA GeForce RTX 5090 Laptop GPU, 3090, 2610, P0, 0',not_same_instant_as_main_timing=True),
    main_start_SM_exceeds_after_run_reported_max=initial_sm>int(current['clocks.max.sm']) if current.get('clocks.max.sm','').isdigit() else None,
    conclusion='Start SM telemetry8040 exceeds later native-Windows reported max3090; cause undetermined. Do not infer stable clocks from these telemetry samples.',
    no_clock_changes=True,no_GPU_timing_rerun=True,inputs={str(p.relative_to(ROOT)):sha(p) for p in [TIM/'telemetry_start.json',TIM/'results.json',Path(__file__)]})
write_new(TIM/'telemetry_audit_addendum.json',out);print(json.dumps(out),flush=True)
