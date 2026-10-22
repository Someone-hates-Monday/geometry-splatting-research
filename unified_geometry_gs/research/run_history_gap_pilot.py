"""Frozen-budget paired training followed automatically by all fixed evaluations."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback,zipfile
import numpy as np
from research.run_history_gap_full_smoke import recipe
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve();assert not out.exists();out.mkdir(parents=True)
    smoke=read(R/'experiments/history_gap_full_smoke_v1/summary.json')
    assert smoke['status']=='complete' and smoke['gate_pass']
    pre=read(R/'results/history_gap_full_train_preflight_v1.json');assert pre['status']=='complete'
    data=read(R/'results/factor_holdout_preflight_v1.json')
    input_hashes=data['input_sha256'].copy()
    common=R/'experiments/factor_holdout55_v2/common'
    input_hashes[str(common/'chkpnt6999.pth')]=pre['checkpoint_sha256']
    input_hashes[str(common/'chkpnt6999.runtime.pth')]=pre['sidecar_sha256']
    gt=ROOT/'dtu_eval/Offical_DTU_Dataset'
    for path in [gt/'ObsMask/ObsMask55_10.mat',gt/'ObsMask/Plane55.mat',gt/'Points/stl/stl055_total.ply',*sorted((gt/'Calibration/cal18').glob('*'))]:
        assert path.is_file();input_hashes[str(path)]=sha(path)
    names=['run_history_gap_pilot.py','train_history_gap_pilot.py','run_history_gap_full_smoke.py',
        'HISTORY_GAP_PILOT_PROTOCOL.md','HISTORY_GAP_FULL_TRAIN_PROTOCOL.md','history_gap_live_adapter.py',
        'history_gap_gpu.py','history_gap_gpu_v2.py','smoke_history_gap_adapter.py','train_solver_branch.py',
        'train_holdout_branch.py','checkpoint_state.py','median_cuda_backend.py','median_adjquad_backend.py',
        'median_adjquad_accumulate.inc','median_adjquad_refine.inc','preflight_factor_holdout.py',
        'exact_outlier_capture_rebuilt.py','benchmark_csr_capture.py','evaluate_images.py',
        'align_holdout_mesh.py','evaluate_mesh_seeded.py','benchmark_median_cuda.py']
    files={R/name for name in names}
    for folder in ['scene','arguments','utils','gaussian_renderer','warp_patch_ncc']:
        files.update((ROOT/folder).rglob('*.py'))
    source=ROOT/'submodules/diff-gaussian-rasterization'
    for pattern in ['*.cu','*.h','*.cpp','*.py']:files.update(source.rglob(pattern))
    files.update([ROOT/'train.py',ROOT/'mesh_extract.py',ROOT/'evaluate_dtu_mesh.py',ROOT/'dtu_eval/eval.py',
        R/'results/history_gap_full_train_preflight_v1.json',R/'results/factor_holdout_preflight_v1.json',
        R/'experiments/history_gap_full_smoke_v1/summary.json'])
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    m=dict(status='preflight',pid=os.getpid(),created_epoch=time.time(),current_stage='preflight',jobs=[],
        source_sha256=sources,input_sha256=input_hashes,initial_state_sha256=pre['initial_state_sha256'],
        train_names=data['pool']['train_names'],test_names=data['pool']['test_names'],
        order=['control','history_gap'],target_iteration=30000,
        gate=dict(overall_ratio_le=.99,directed_ratios_le=1.,test_psnr_delta_ge=-.1,
            test_ssim_delta_ge=-.001,gaussian_count_ratio_le=1.05,training_wall_ratio_le=2.),
        limitations=['Exploratory scan55, prior research exposure and transductive SfM.',
            'One shared warmup pair; evaluator seeds are not independent training replicates.',
            'Sequential resource accounting including checks and CSR; not a speed benchmark.'])
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    m['source_archive_sha256']=sha(out/'sources.zip');save(out/'manifest.json',m)
    def guard():
        for name,h in sources.items():assert sha(ROOT/name)==h,('source changed',name)
        for name,h in input_hashes.items():assert sha(Path(name))==h,('input changed',name)
    def run(stage,argv):
        guard();entry=dict(stage=stage,status='starting',argv=argv,started_epoch=time.time())
        m['jobs'].append(entry);m.update(status='running',current_stage=stage);save(out/'manifest.json',m)
        with (out/(stage+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            entry.update(status='running',pid=child.pid);save(out/'manifest.json',m)
            print(f'{stage} running pid={child.pid}',flush=True);code=child.wait()
        entry.update(status='complete' if code==0 else 'failed',returncode=code,wall_seconds=time.time()-entry['started_epoch'])
        save(out/'manifest.json',m);print(f'{stage} exit={code} wall={entry["wall_seconds"]:.2f}s',flush=True)
        if code:raise RuntimeError(stage+' failed; outputs retained, no automatic retry')
        return entry
    try:
        guard();m['input_hashes_reverified_before_training']=True;save(out/'manifest.json',m)
        training={};records={}
        for label,weight in [('control','disabled'),('history_gap','.01')]:
            model=out/label
            training[label]=run('train_'+label,[sys.executable,'-u','-m','research.train_history_gap_pilot',
                '--hg-label',label,'--hg-weight',weight,*recipe(model)])
            rec=records[label]=read(model/'history_gap_training_integration.json')
            assert rec['status']=='complete' and rec['full_training_completed'] and rec['steps_observed']==23000
            assert rec['parent_guards_verified'] and rec['initial_state_sha256']==pre['initial_state_sha256']
            assert (model/'point_cloud/iteration_30000/point_cloud.ply').is_file()
            assert rec['final_backward_without_optimizer']['iteration']==30000
        for name in ['baseline_loss','rgb','ncc','geo','depth_normal','camera']:
            assert records['control']['steps'][0][name]==records['history_gap']['steps'][0][name],name
        a=read(out/'control/solver_training_integration.json');b=read(out/'history_gap/solver_training_integration.json')
        assert a['backend_sha256']==b['backend_sha256']
        images={};meshes={}
        for label in m['order']:
            model=out/label
            run('extract_'+label,[sys.executable,'mesh_extract.py','-m',str(model),'--iteration','30000',
                '--voxel_size','.002','--trunc_voxel_multiplier','4','--depth_max','8','--num_cluster','1'])
            run('align_'+label,[sys.executable,'-m','research.align_holdout_mesh','--model',str(model)])
            meshes[label]=[]
            for seed in [0,1,2]:
                dest=out/f'eval_{label}_seed{seed}'
                run(f'eval_{label}_seed{seed}',[sys.executable,'-m','research.evaluate_mesh_seeded',
                    '--mesh',str(model/'recon_aligned.ply'),'--scan','55','--seed',str(seed),'--out',str(dest)])
                meshes[label].append(read(dest/'results.json'))
            images[label]={}
            for split in ['train','test']:
                dest=out/f'images_{label}_{split}.json'
                run(f'images_{label}_{split}',[sys.executable,'-m','research.evaluate_images','--model',str(model),
                    '--iteration','30000','--split',split,'--out',str(dest)])
                images[label][split]=read(dest)
                assert [r['camera'] for r in images[label][split]['rows']]==m[split+'_names']
        alignment=[read(out/label/'alignment.json') for label in m['order']]
        for key in ['scale','rotation','translation']:
            assert np.array_equal(np.asarray(alignment[0][key]),np.asarray(alignment[1][key])),key
        ratios={key:float(np.mean([x[key] for x in meshes['history_gap']])/np.mean([x[key] for x in meshes['control']])) for key in ['overall','mean_d2s','mean_s2d']}
        deltas={key:images['history_gap']['test']['mean'][key]-images['control']['test']['mean'][key] for key in ['psnr_global_rgb_db','ssim_repository']}
        capacity=images['history_gap']['test']['gaussian_count']/images['control']['test']['gaussian_count']
        cost=training['history_gap']['wall_seconds']/training['control']['wall_seconds']
        gates=dict(geometry=ratios['overall']<=.99,d2s=ratios['mean_d2s']<=1.,s2d=ratios['mean_s2d']<=1.,
            test_psnr=deltas['psnr_global_rgb_db']>=-.1,test_ssim=deltas['ssim_repository']>=-.001,
            capacity=capacity<=1.05,resource_cost=cost<=2.)
        result=dict(status='complete',gate_pass=all(gates.values()),gates=gates,mesh_ratios=ratios,
            test_deltas=deltas,gaussian_count_ratio=capacity,training_wall_ratio=cost,
            images=images,meshes=meshes,scope=m['limitations'])
        save(out/'paired_quality.json',result);guard()
        m.update(status='complete',current_stage='complete',gate_pass=result['gate_pass'],
            quality_sha256=sha(out/'paired_quality.json'),finished_epoch=time.time())
        m['artifacts_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json' and not p.name.startswith('events.out.tfevents')}
        save(out/'manifest.json',m);print(json.dumps({k:result[k] for k in ['gate_pass','gates','mesh_ratios','test_deltas','gaussian_count_ratio','training_wall_ratio']}),flush=True)
    except BaseException as exc:
        m.update(status='failed',failure=repr(exc),finished_epoch=time.time());save(out/'manifest.json',m)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
