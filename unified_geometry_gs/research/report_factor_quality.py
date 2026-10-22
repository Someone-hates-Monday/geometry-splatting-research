"""Audit the complete paired pilot against actual sources, models and metrics."""
from __future__ import annotations
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def local(s):return Path('C:/'+s[7:]) if s.startswith('/mnt/c/') else Path(s)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve();s=read(run/'manifest.json');q=read(run/'paired_quality.json')
    if s['status']!='complete' or sha(run/'paired_quality.json')!=s['quality_results_sha256']:raise ValueError('Complete unchanged quality result required.')
    if s['pilot_gate']!={'mean_overall_candidate_control_ratio_le':.99,'mean_s2d_delta_le':0.,'train_psnr_delta_ge':-.2}:raise ValueError('Predefined quality gates changed.')
    expected=['train_factor64_smoke','train_factor64']
    for v in ('legacy64','factor64'):expected+=['extract_'+v,'align_'+v,*[f'eval_{v}_seed{i}' for i in (0,1,2)],'images_'+v]
    if [x['stage'] for x in s['steps']]!=expected or any(x['status']!='complete' or x['returncode']!=0 for x in s['steps']):raise ValueError('Stage sequence or completion differs.')
    with zipfile.ZipFile(run/'sources.zip') as z:
        if sha(run/'sources.zip')!=s['source_archive_sha256']:raise ValueError('Archive changed.')
        for n,h in s['source_sha256'].items():
            if sha(ROOT/n)!=h or hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Actual/archived source changed: '+n)
    for n,h in q['artifact_sha256'].items():
        if sha(local(n))!=h:raise ValueError('Quality artifact changed: '+n)
    control=local(s['control_reference']);candidate=run/'factor64'
    a=read(control/'corrected_training_integration.json');b=read(candidate/'factor_training_integration.json')
    for k in ('initial_state_sha256','common_checkpoint_sha256','sidecar_sha256','first_camera','initial_gaussians','first_loss_scalars'):
        if a[k]!=b[k]:raise ValueError('Matched full starting state differs: '+k)
    if any(x['status']!='complete' or x['steps_observed']!=23000 for x in (a,b)):raise ValueError('Full update count differs.')
    if sha(control/'point_cloud/iteration_30000/point_cloud.ply')!=s['control_model_sha256_before_candidate']:raise ValueError('Preselected control changed.')
    if b['backend_sha256']!=s['factor_backend']['binary_sha256']:raise ValueError('Candidate binary differs from preflight.')
    for x in (a,b):
        if sha(local(str(ROOT/'research/experiments/adjquad55_v1/common/chkpnt6999.pth')))!=x['common_checkpoint_sha256']:raise ValueError('Checkpoint input changed.')
        if sha(ROOT/'research/experiments/adjquad55_v1/common/chkpnt6999.runtime.pth')!=x['sidecar_sha256']:raise ValueError('Sidecar input changed.')
    metrics={};images={};alignments={}
    for v,model in (('legacy64',control),('factor64',candidate)):
        align=read(model/'alignment.json');alignments[v]=align
        if sha(model/'recon_post.ply')!=align['source_mesh_sha256'] or sha(model/'recon_aligned.ply')!=align['aligned_mesh_sha256']:raise ValueError('Alignment artifacts changed.')
        for path,h in align['source_sha256'].items():
            if sha(local(path))!=h:raise ValueError('Alignment helper source changed.')
        metrics[v]=[]
        for seed in (0,1,2):
            d=run/f'eval_{v}_seed{seed}';ev=read(d/'manifest.json');raw=read(d/'results.json')
            if ev['status']!='complete' or ev['shuffle_seed']!=seed or ev['scan']!=55:raise ValueError('Evaluator status/seed/scan differs.')
            for path,h in ev['source_sha256'].items():
                if sha(local(path))!=h:raise ValueError('Evaluator, GT or mesh changed: '+path)
            if raw!=q['mesh_metrics'][v][seed]:raise ValueError('Raw mesh result differs from pair.')
            if abs(raw['overall']-.5*(raw['mean_d2s']+raw['mean_s2d']))>1e-12:raise ValueError('Metric identity failed.')
            metrics[v].append(raw)
        image=read(run/f'images_{v}.json');images[v]=image
        if image['camera_count']!=49 or image['split']!='train' or image['is_novel_view_evaluation'] or image['partial_smoke'] or image['iteration']!=30000:raise ValueError('Image split/count/iteration differs.')
        if sha(model/'point_cloud/iteration_30000/point_cloud.ply')!=image['source_sha256']['point_cloud.ply'] or sha(model/'cfg_args')!=image['source_sha256']['cfg_args']:raise ValueError('Image evaluation model/config changed.')
        for k in image['mean']:
            if abs(statistics.mean(x[k] for x in image['rows'])-image['mean'][k])>1e-12:raise ValueError('Image mean differs from actual camera rows.')
        extract=next(x['argv'] for x in s['steps'] if x['stage']=='extract_'+v)
        for key,value in (('--iteration','30000'),('--voxel_size','.002'),('--trunc_voxel_multiplier','4'),('--depth_max','8'),('--num_cluster','1')):
            if extract[extract.index(key)+1]!=value:raise ValueError('Extraction protocol differs.')
    cameras=lambda v:[(x['camera'],x['height'],x['width']) for x in images[v]['rows']]
    for k in ('scale','rotation','translation','convention'):
        if alignments['legacy64'][k]!=alignments['factor64'][k]:raise ValueError('Same-camera alignment transform differs: '+k)
    if cameras('legacy64')!=cameras('factor64'):raise ValueError('Image cameras or dimensions differ.')
    for k in ('background','resolution_parameter','appearance'):
        if images['legacy64'][k]!=images['factor64'][k]:raise ValueError('Image renderer settings differ: '+k)
    means={v:{k:statistics.mean(x[k] for x in metrics[v]) for k in ('overall','mean_d2s','mean_s2d')} for v in metrics}
    delta={k:means['factor64'][k]-means['legacy64'][k] for k in means['legacy64']}
    ratio=means['factor64']['overall']/means['legacy64']['overall'];psnr=images['factor64']['mean']['psnr_global_rgb_db']-images['legacy64']['mean']['psnr_global_rgb_db']
    gate=ratio<=.99 and delta['mean_s2d']<=0 and psnr>=-.2
    if gate!=q['pilot_gate_pass'] or gate!=s['pilot_gate_pass'] or abs(ratio-q['mean_overall_ratio'])>1e-12 or abs(psnr-q['train_psnr_delta'])>1e-12:raise ValueError('Recomputed pilot verdict differs.')
    report={'checked_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'actual_sources_models_GT_and_metrics_verified':True,
        'source_count':len(s['source_sha256']),'source_archive_sha256':s['source_archive_sha256'],'paired_results_sha256':sha(run/'paired_quality.json'),
        'complete_initial_state_verified':True,'fixed_preselected_control_verified':True,'mesh_mean_mm':means,'mean_delta_factor_minus_legacy_mm':delta,
        'mean_overall_ratio':ratio,'relative_overall_improvement_percent':100*(1-ratio),'train_psnr_delta_db':psnr,'pilot_gate_pass':gate,
        'individual_gates':{'overall_at_least_one_percent':ratio<=.99,'s2d_not_increased':delta['mean_s2d']<=0,'train_psnr_drop_at_most_point2_db':psnr>=-.2},
        'images':{v:{'camera_count':x['camera_count'],'split':x['split'],'mean':x['mean'],'gaussian_count':x['gaussian_count']} for v,x in images.items()},
        'limits':['One scan55 pair from one fixed state; no independent training replications or significance claim.',
            'Evaluator seeds are mesh-sampling repetitions, not training repetitions.','49 TRAIN views do not establish novel-view quality.',
            'Combined intervention; feature-level attribution requires ablation.','Wall clock is sequential resource accounting, not a speed comparison.',
            'Gram forward was a separate 20-update smoke and is absent from this quality candidate.']}
    if args.out.exists():raise ValueError('Keep prior reports; use fresh out path.')
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('limits','images')},indent=2))
if __name__=='__main__':main()
