"""Describe a verified same-state control repeat without replacing preset gates."""
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
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--repeat',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh output required.')
    sources={}
    def read(p):
        sources[str(p.resolve())]=digest(p);return json.loads(p.read_text())
    pilot=read(args.report)
    for name,sha in pilot['source_sha256'].items():
        if digest(host_path(name))!=sha:raise ValueError(f'Pilot report source changed: {name}')
    repeat=args.repeat.resolve()
    m=read(repeat/'manifest.json')
    if m['status']!='complete' or not m['branch_state_verified'] or any(s['status']!='complete' for s in m['steps']):
        raise ValueError('Repeat incomplete.')
    for name,sha in m['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError(f'Repeat source changed: {name}')
    with zipfile.ZipFile(repeat/'sources.zip') as z:
        if any(hashlib.sha256(z.read(k)).hexdigest()!=v for k,v in m['source_sha256'].items()):
            raise ValueError('Repeat archived sources changed.')
    base=host_path(m['base_pair'])
    a=read(base/'control/solver_training_integration.json')
    b=read(repeat/'control_repeat/solver_training_integration.json')
    invariants=('initial_state_sha256','common_checkpoint_sha256','sidecar_sha256','first_camera','backend_sha256','initial_gaussians')
    equal={k:a[k]==b[k] for k in invariants}
    if not all(equal.values()) or not b['completed'] or not b['resume_state_identical']:
        raise ValueError('Same-state/backend comparison not verified.')
    metrics=read(repeat/'paired_mesh_metrics.json')
    for seed in (0,1,2):
        e=read(repeat/f'eval_seed{seed}'/'manifest.json')
        values=read(repeat/f'eval_seed{seed}'/'results.json')
        if e['status']!='complete' or values!=e['metrics'] or values!=metrics['rows'][seed]['control_repeat']:
            raise ValueError('Repeat evaluation result mismatch.')
        for name,sha in e['source_sha256'].items():
            if digest(host_path(name))!=sha:raise ValueError(f'Evaluation input changed: {name}')
    image=read(repeat/'images_control_repeat.json')
    original=read(base/'images_control.json')
    if image['camera_count']!=49 or image['is_novel_view_evaluation'] or image['partial_smoke']:
        raise ValueError('Unexpected image evaluation scope.')
    if [r['camera'] for r in image['rows']]!=[r['camera'] for r in original['rows']]:raise ValueError('Camera list mismatch.')
    for k,v in image['mean'].items():
        if abs(statistics.mean(r[k] for r in image['rows'])-v)>1e-10:raise ValueError('Image aggregate mismatch.')
    paths={'point_cloud.ply':repeat/'control_repeat/point_cloud/iteration_30000/point_cloud.ply',
        'cfg_args':repeat/'control_repeat/cfg_args','evaluate_images.py':ROOT/'research/evaluate_images.py'}
    if any(digest(paths[k])!=v for k,v in image['source_sha256'].items()):raise ValueError('Repeat image input changed.')
    keys=('overall','mean_d2s','mean_s2d')
    delta={k:statistics.mean(r['delta_repeat_minus_control'][k] for r in metrics['rows']) for k in keys}
    candidate_vs_repeat={k:pilot['mean_mesh_delta_adjquad_minus_control'][k]-delta[k] for k in keys}
    candidate_vs_control_average={k:pilot['mean_mesh_delta_adjquad_minus_control'][k]-.5*delta[k] for k in keys}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'same_state_invariants':equal,
        'control_repeat_mesh_means':{k:statistics.mean(r['control_repeat'][k] for r in metrics['rows']) for k in keys},
        'mean_mesh_delta_repeat_minus_original_control':delta,
        'original_candidate_delta':pilot['mean_mesh_delta_adjquad_minus_control'],
        'candidate_minus_repeat_diagnostic_only':candidate_vs_repeat,
        'candidate_minus_two_control_average_exploratory_only':candidate_vs_control_average,
        'repeat_difference_to_original_candidate_overall_difference_ratio':abs(delta['overall'])/abs(pilot['mean_mesh_delta_adjquad_minus_control']['overall']),
        'original_preset_quality_gates_retained':pilot['preset_quality_gates'],
        'original_preset_gate_pass_retained':pilot['all_preset_quality_gates_pass'],
        'repeat_training_view_image_means':image['mean'],
        'repeat_training_view_image_delta':{k:image['mean'][k]-original['mean'][k] for k in image['mean']},
        'resource_records':{'original_control_wall_seconds':a['wall_seconds'],'control_repeat_wall_seconds':b['wall_seconds'],
            'note':'Even identical solver/state repeats have very different sequential wall clocks; no end-to-end speed inference.'},
        'interpretation':'One same-state control repeat produces a larger geometric difference than the original candidate contrast; a single pair cannot isolate the method effect.',
        'limitations':['One repeated control trajectory provides neither a variance distribution nor a significance/equivalence test.',
            'Do not choose the more favorable repeated baseline to reverse the original failed adoption verdict.',
            'Initial recorded states and backend match; subsequent CUDA scheduling and unrecorded system state can differ.',
            'Three mesh evaluator seeds are not three training repetitions; both controls share the same training seed/warmup.'],
        'source_sha256':sources}
    sources[str(Path(__file__).resolve())]=digest(Path(__file__))
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('mean_mesh_delta_repeat_minus_original_control','candidate_minus_repeat_diagnostic_only','original_preset_gate_pass_retained','interpretation')},indent=2))


if __name__=='__main__':main()
