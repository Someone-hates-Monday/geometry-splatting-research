"""Validate completed work-grouping/gradient evidence and preserve failed gates."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import zipfile
from research.report_adjquad55 import host_path
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Fresh report required.')
    used={}
    def read(p):
        used[str(p.resolve())]=sha(p);return json.loads(p.read_text())
    def experiment(name):
        d=ROOT/'research/experiments'/name;m=read(d/'manifest.json')
        if m['status']!='complete':raise ValueError(f'Incomplete experiment: {name}')
        for name,expected in m['source_sha256'].items():
            p=host_path(name) if name.startswith('/') else ROOT/name
            if sha(p)!=expected:raise ValueError(f'Source changed: {p}')
        archive=d/'sources.zip'
        if sha(archive)!=m['source_archive_sha256']:raise ValueError('Source archive changed.')
        used[str(archive.resolve())]=sha(archive)
        # Check actual archived bytes, including manifests with WSL absolute keys.
        expected={str((host_path(k) if k.startswith('/') else ROOT/k).resolve()):v for k,v in m['source_sha256'].items()}
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                key=str((ROOT/n).resolve())
                if key not in expected or hashlib.sha256(z.read(n)).hexdigest()!=expected[key]:raise ValueError(f'Archive content mismatch: {n}')
        return read(d/'results.json')
    batch=experiment('sampling_batch_v2')
    loss=experiment('sampling_full_loss_v1')
    diagnosis=experiment('sampling_gradient_diagnostic_v1')
    timing=[]
    for row in batch['rows']:
        for variant in ('batch1','batch4'):
            ratios={k:statistics.median(v[variant])/statistics.median(v['control']) for k,v in row['gpu_ms_rounds'].items()}
            timing.append({'model':Path(row['model']).name,'camera':row['camera'],'variant':variant,'median_round_latency_ratio':ratios})
    full_groups=[(Path(r['model']).name,r['camera'],f,c) for r in loss['rows'] for f,c in r['gradient_comparison'].items() if c['participates']]
    failures=[{'model':m,'camera':c,'field':f,**v} for m,c,f,v in full_groups if not v['finite'] or v['relative_l2_drift']>.001]
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),
        'sample_output_bitwise_gate_pass':batch['all_sample_outputs_bitwise_equal'],
        'sparse_primitive_gradient_gate_pass':batch['all_gradient_groups_finite_and_drift_le_0_001'],
        'frozen_camera_timing':timing,
        'batch1_sample_latency_ratio_range':[min(r['median_round_latency_ratio']['sample_forward_backward'] for r in timing if r['variant']=='batch1'),max(r['median_round_latency_ratio']['sample_forward_backward'] for r in timing if r['variant']=='batch1')],
        'batch1_full_loss_latency_ratio_range':[min(r['median_round_latency_ratio']['render_patchmatch_loss_backward'] for r in timing if r['variant']=='batch1'),max(r['median_round_latency_ratio']['render_patchmatch_loss_backward'] for r in timing if r['variant']=='batch1')],
        'expanded_audit_cameras':len(loss['rows']),'expanded_audit_participating_groups':len(full_groups),
        'all_complete_loss_components_bitwise_equal':loss['all_loss_components_bitwise_equal'],
        'expanded_full_gradient_gate_pass':loss['all_gradients_finite_and_drift_le_0_001'],
        'expanded_failed_groups':failures,
        'expanded_max_gradient_relative_l2_drift':loss['maximum_gradient_relative_l2_drift'],
        'expanded_control_repeat_max_relative_l2_drift':loss['maximum_control_repeat_relative_l2_drift'],
        'same_graph_component_diagnostic':diagnosis,
        'decision':'Do not start the prepared actual-training timing runner: expanded full-loss gradient gate failed. Diagnose numerical stability first; retain the 0.001 gate.',
        'limitations':['Frozen outputs and timings do not establish full-training speed or quality.',
            'Same-backend variability affects several world-parameter gradients; no attribution to CUDA atomics alone.',
            'Two worst-case diagnostic cameras were selected after the failure and are not independent validation.'],
        'source_sha256':used}
    args.out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k in ('sample_output_bitwise_gate_pass','expanded_full_gradient_gate_pass','expanded_audit_cameras','expanded_audit_participating_groups','batch1_full_loss_latency_ratio_range')},indent=2))


if __name__=='__main__':main()
