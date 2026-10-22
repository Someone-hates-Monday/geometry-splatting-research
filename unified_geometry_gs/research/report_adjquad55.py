"""Verify complete shared-state pilot evidence and report all preset gates."""
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


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def host_path(value):
    if value.startswith('/mnt/c/') and ROOT.drive:
        return Path('C:/'+value[len('/mnt/c/'):])
    return Path(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh report required.')
    sources={}
    def read(p):
        sources[str(p.resolve())]=digest(p)
        return json.loads(p.read_text())
    pair=args.pair.resolve()
    m=read(pair/'manifest.json')
    if m['status']!='complete' or not m['branch_state_verified'] or any(s['status']!='complete' or s['returncode']!=0 for s in m['steps']):
        raise ValueError('Full pipeline or branch-state verification incomplete.')
    mismatched=[name for name,sha in m['source_sha256'].items() if digest(ROOT/name)!=sha]
    if mismatched:raise ValueError(f'Experiment source changed: {mismatched}')
    archive=pair/'sources.zip'
    if digest(archive)!=m['source_archive_sha256']:raise ValueError('Source archive changed.')
    with zipfile.ZipFile(archive) as z:
        if any(hashlib.sha256(z.read(name)).hexdigest()!=sha for name,sha in m['source_sha256'].items()):
            raise ValueError('Archived source content mismatch.')
    raw=read(pair/'paired_mesh_metrics.json')
    expected_seeds=[0,1,2]
    if [r['seed'] for r in raw['rows']]!=expected_seeds:raise ValueError('Evaluator seed list changed.')
    images={};integrations={};metrics=[]
    for v in ('control','adjquad4'):
        integrations[v]=read(pair/v/'solver_training_integration.json')
        i=integrations[v]
        if not (i['completed'] and i['resume_state_identical'] and i['initial_state_sha256']==m['initial_state_sha256'] and i['common_checkpoint_sha256']==m['common_checkpoint_sha256']):
            raise ValueError('Runtime branch state mismatch.')
        images[v]=read(pair/f'images_{v}.json')
        im=images[v]
        if im['camera_count']!=49 or len(im['rows'])!=49 or im['is_novel_view_evaluation'] or im['partial_smoke']:
            raise ValueError('Expected all 49 training views, no relabeling.')
        for key,mean in im['mean'].items():
            vals=[r[key] for r in im['rows']]
            if not all(math.isfinite(x) for x in vals) or abs(statistics.mean(vals)-mean)>1e-10:
                raise ValueError('Invalid image aggregate.')
        image_sources={'point_cloud.ply':pair/v/'point_cloud/iteration_30000/point_cloud.ply',
            'cfg_args':pair/v/'cfg_args','evaluate_images.py':ROOT/'research/evaluate_images.py'}
        if set(im['source_sha256'])!=set(image_sources):raise ValueError('Unexpected image source keys.')
        for value,sha in im['source_sha256'].items():
            if digest(image_sources[value])!=sha:raise ValueError(f'Image source changed: {value}')
        for seed in expected_seeds:
            e=read(pair/f'eval_{v}_seed{seed}'/'manifest.json')
            result=read(pair/f'eval_{v}_seed{seed}'/'results.json')
            if e['status']!='complete' or e['returncode'] or e['shuffle_seed']!=seed:
                raise ValueError('Evaluation incomplete.')
            if e['metrics']!=result or result!=raw['rows'][seed][v]:raise ValueError('Metric source mismatch.')
            if abs(result['overall']-(result['mean_d2s']+result['mean_s2d'])/2)>1e-12:
                raise ValueError('Overall convention mismatch.')
            for name,sha in e['source_sha256'].items():
                if digest(host_path(name))!=sha:raise ValueError(f'Evaluation input changed: {name}')
    if [r['camera'] for r in images['control']['rows']]!=[r['camera'] for r in images['adjquad4']['rows']]:
        raise ValueError('Camera lists differ.')
    if len({i['sidecar_sha256'] for i in integrations.values()})!=1 or len({i['first_camera'] for i in integrations.values()})!=1:
        raise ValueError('Sidecar or initial camera mismatch.')
    keys=('overall','mean_d2s','mean_s2d')
    means={v:{k:statistics.mean(row[v][k] for row in raw['rows']) for k in keys} for v in images}
    delta={k:means['adjquad4'][k]-means['control'][k] for k in keys}
    if any(abs(delta[k]-raw['mean_delta_adjquad_minus_control'][k])>1e-12 for k in keys):
        raise ValueError('Pair mean mismatch.')
    image_delta={k:images['adjquad4']['mean'][k]-images['control']['mean'][k] for k in images['control']['mean']}
    gates={'mean_overall_delta_le_minus_0_0005_mm':delta['overall']<=-.0005,
        'mean_s2d_delta_le_zero':delta['mean_s2d']<=0,
        'training_psnr_delta_ge_minus_0_2_db':image_delta['psnr_global_rgb_db']>=-.2}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'scan':m['scan'],
        'pipeline_complete_and_sources_verified':True,'common_state_verified':True,
        'initial_state_sha256':m['initial_state_sha256'],'first_camera':m['first_camera'],
        'fixed_evaluator_seeds':expected_seeds,'mesh_means':means,'mean_mesh_delta_adjquad_minus_control':delta,
        'overall_delta_range':[min(r['delta_adjquad_minus_control']['overall'] for r in raw['rows']),max(r['delta_adjquad_minus_control']['overall'] for r in raw['rows'])],
        'all_evaluator_overall_deltas_positive':all(r['delta_adjquad_minus_control']['overall']>0 for r in raw['rows']),
        'training_view_image_means':{v:images[v]['mean'] for v in images},'training_view_image_delta':image_delta,
        'gaussian_count':{v:images[v]['gaussian_count'] for v in images},
        'gaussian_ply_bytes':{v:images[v]['gaussian_ply_bytes'] for v in images},
        'training_resource_records':{v:{k:integrations[v][k] for k in ('wall_seconds','peak_torch_allocated_bytes','render_calls','depth_render_calls','ncc_calls')} for v in integrations},
        'preset_quality_gates':gates,'all_preset_quality_gates_pass':all(gates.values()),
        'decision':'Do not adopt as a geometric quality improvement; keep frozen efficiency/conditional math evidence separate.',
        'next_diagnostic':'Repeat the control suffix from the identical common state to measure CUDA training variation; no gate changes or independent-seed claim.',
        'limitations':['One development scene, one trained candidate/control pair; evaluation seeds do not increase training replication count.',
            'Initial model/optimizer/RNG/camera queue equality is verified, subsequent CUDA atomics are not deterministic.',
            'Without a repeated control trajectory this small quality delta cannot be cleanly separated from training variation.',
            'All image views were used in training; there is no novel-view or statistical significance claim.',
            'Sequential single-run wall clocks are resource records, not an end-to-end acceleration experiment.'],
        'source_sha256':sources}
    sources[str(Path(__file__).resolve())]=digest(Path(__file__))
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('mean_mesh_delta_adjquad_minus_control','training_view_image_delta','preset_quality_gates','decision')},indent=2))


if __name__=='__main__':main()
