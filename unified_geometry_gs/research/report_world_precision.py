"""Validate frozen precision experiments without treating repeatability as truth."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mixed',type=Path);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh report required.')
    used={}
    def read(p):
        used[str(p.resolve())]=sha(p);return json.loads(p.read_text())
    def check_archive(folder,current=True):
        m=read(folder/'manifest.json')
        if m['status']!='complete':raise ValueError(f'Experiment not complete: {folder}')
        paths={}
        for name,expected in m['source_sha256'].items():
            local=host_path(name) if name.startswith('/') else ROOT/name
            paths[str(local.resolve())]=expected
            if current and sha(local)!=expected:raise ValueError(f'Current source changed: {local}')
        zpath=folder/'sources.zip'
        if sha(zpath)!=m['source_archive_sha256']:raise ValueError('Source archive changed.')
        used[str(zpath.resolve())]=sha(zpath)
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if hashlib.sha256(z.read(name)).hexdigest()!=paths[str((ROOT/name).resolve())]:raise ValueError('Archived source differs.')
        return m
    audit_manifest=read(ROOT/'research/experiments/sampling_full_loss_v1/manifest.json')
    for name,expected in audit_manifest['input_sha256'].items():
        if sha(host_path(name))!=expected:raise ValueError('Frozen PLY changed.')
    diagnostic={}
    for label,name in (('original','sampling_gradient_diagnostic_v1'),('inverse_factor','inverse_gradient_diagnostic_v1'),('fp64','precise_gradient_diagnostic_v2')):
        d=ROOT/'research/experiments'/name;check_archive(d)
        diagnostic[label]=read(d/'results.json')
    comparisons=[]
    for old,new in zip(diagnostic['original']['rows'],diagnostic['fp64']['rows']):
        if old['camera']!=new['camera'] or old['model']!=new['model']:raise ValueError('Diagnostic camera mismatch.')
        for component in old['component_gradient_comparisons']:
            for field,o in old['component_gradient_comparisons'][component].items():
                n=new['component_gradient_comparisons'][component][field]
                if o['participates']:
                    comparisons.append({'model':Path(old['model']).name,'camera':old['camera'],'component':component,'field':field,
                        'original_repeat_relative_l2':o['baseline_repeat_pairwise_max_relative_l2'],
                        'fp64_repeat_relative_l2':n['baseline_repeat_pairwise_max_relative_l2'],
                        'fp64_batch1_vs_batch2_cross_relative_l2':n['candidate_control_cross_max_relative_l2']})
    def development(d):
        parent=check_archive(d);check_archive(d/'audit');check_archive(d/'timing')
        a=read(d/'audit/results.json');t=read(d/'timing/results.json')
        if any(s['status']!='complete' for s in parent['steps']):raise ValueError('Incomplete stages.')
        rows=[]
        for row in t['rows']:
            times=row['gpu_ms_rounds']['render_patchmatch_loss_backward']
            med={v:statistics.median(x) for v,x in times.items()}
            if len(set(row['full_frozen_losses'].values()))!=1:raise ValueError('Complete frozen loss values differ.')
            rows.append({'model':Path(row['model']).name,'camera':row['camera'],
                'median_round_gpu_ms':med,'latency_ratio_to_original':{v:x/med['control'] for v,x in med.items()}})
        return {'participating_groups':a['participating_groups'],
            'all_loss_components_bitwise_equal':a['all_loss_components_bitwise_equal'],
            'full_gradient_gate_pass':a['all_gradients_finite_and_drift_le_0_001'],
            'max_gradient_relative_l2':a['maximum_gradient_relative_l2_drift'],
            'max_reference_repeat_relative_l2':a['maximum_precise2_repeat_relative_l2_drift'],
            'sample_output_bitwise_gate_pass':t['all_sample_outputs_bitwise_equal'],
            'timing':rows}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),
        'precision_diagnostic_comparison':comparisons,
        'fp64_development':development(ROOT/'research/experiments/precise_development_v1'),
        'normalization_local_proof':read(ROOT/'research/results/normal_preprocess_vjp_v1.json'),
        'limitations':['FP64 reference shares analytic formulas, masks, IFT and float32 contributor accumulations; repeatability is not a correctness oracle.',
            'Selected diagnostic camera comparisons use each run first-reference gradient norm and only five replays, not a noise distribution.',
            'No full-training speed, quality, statistical significance or novelty claim.',
            'The isolated normal-normalization derivative discrepancy does not explain geometry-only variability.']}
    if args.mixed:result['mixed_development']=development(args.mixed.resolve())
    result['source_sha256']=used
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result['fp64_development'].items() if k!='timing'},indent=2))
    if args.mixed:print(json.dumps({k:v for k,v in result['mixed_development'].items() if k!='timing'},indent=2))


if __name__=='__main__':main()
