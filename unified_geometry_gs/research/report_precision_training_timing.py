"""Audit counterbalanced actual-training timing and retain its preset verdict."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required.')
    used={}
    def read(p):used[str(p.resolve())]=sha(p);return json.loads(p.read_text())
    d=args.run.resolve();m=read(d/'manifest.json');r=read(d/'results.json')
    if m['status']!='complete' or len(m['steps'])!=6 or any(s['status']!='complete' or s['returncode']!=0 for s in m['steps']):raise ValueError('All six runs must complete.')
    for n,h in m['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError(f'Actual-training source changed: {n}')
    archive=d/'sources.zip'
    if sha(archive)!=m['source_archive_sha256']:raise ValueError('Source archive changed.')
    with zipfile.ZipFile(archive) as z:
        for n,h in m['source_sha256'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Archived source mismatch.')
    used[str(archive)]=sha(archive)
    invariants=[];rows=[];backend_hashes={}
    for row in r['rows']:
        variants={}
        for variant in ('control','mixed1'):
            model=d/f'round{row["round"]}_{variant}'
            a=read(model/'precision_training_integration.json');b=read(model/'solver_training_integration.json')
            if a['status']!='complete' or not a['intentional_stop_observed'] or a['optimizer_steps_observed']!=500:raise ValueError('Intentional timing boundary mismatch.')
            if a['initial_state_sha256']!=m['common_initial_state_sha256'] or not b['resume_state_identical']:raise ValueError('Initial state mismatch.')
            if a['measured_iterations']!=list(range(7050,7500)) or len(a['iteration_gpu_ms'])!=450:raise ValueError('Measured phase changed.')
            if any(not math.isfinite(x) or x<=0 for x in a['iteration_gpu_ms']):raise ValueError('Invalid GPU time.')
            invariants.append({k:a[k] for k in ('initial_state_sha256','common_checkpoint_sha256','sidecar_sha256','first_camera')})
            backend_hashes.setdefault(variant,set()).add(a['backend_sha256'])
            recorded=row['variants'][variant]
            if abs(statistics.mean(a['iteration_gpu_ms'])-recorded['mean_iteration_gpu_ms'])>1e-10 or a['measured_block_wall_seconds']!=recorded['block_wall_seconds']:raise ValueError('Timing aggregate mismatch.')
            variants[variant]=recorded
        ratio=variants['mixed1']['block_wall_seconds']/variants['control']['block_wall_seconds']
        if ratio!=row['mixed1_control_block_wall_ratio']:raise ValueError('Ratio mismatch.')
        rows.append({'round':row['round'],'order':row['order'],'variants':variants,'mixed1_control_block_wall_ratio':ratio})
    if any(x!=invariants[0] for x in invariants) or any(len(x)!=1 for x in backend_hashes.values()):raise ValueError('State/backend invariants failed.')
    geom=math.exp(statistics.mean(math.log(x['mixed1_control_block_wall_ratio']) for x in rows))
    gate=geom<=.97 and all(x['mixed1_control_block_wall_ratio']<=1.01 for x in rows)
    if geom!=r['geometric_mean_paired_block_wall_ratio'] or gate!=r['short_phase_timing_gate_pass']:raise ValueError('Preset verdict mismatch.')
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'all_six_common_state_invariants_verified':True,
        'common_state':invariants[0],'rows':rows,'geometric_mean_paired_block_wall_ratio':geom,
        'preset_short_phase_speed_gate_pass':gate,
        'decision':'Do not adopt as an efficiency improvement; preserve mathematical/stability diagnostics separately.',
        'limitations':['Only scan55 early regularized iterations 7050-7499, 450 timed steps after 50 warmup steps per process.',
            'Three timing process pairs from identical state are not independent random-seed quality replications.',
            'Gaussian counts evolve differently; this is actual short-training trajectory cost, not a frozen fixed-count microbenchmark.',
            'No final model, mesh, held-out image or all-phase speed claim.'],
        'source_sha256':used}
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'ratios':[x['mixed1_control_block_wall_ratio'] for x in rows],
        'geometric_mean':geom,'speed_gate_pass':gate},indent=2))


if __name__=='__main__':main()
