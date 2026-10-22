"""Independent source/stream/state/quality audit; running mode makes no final claim."""
from pathlib import Path
import argparse,ast,collections,hashlib,json,math,statistics,sys,time,zipfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
LABELS=['control','history_gap']
FIXED_GATE=dict(overall_ratio_le=.99,directed_ratios_le=1.,test_psnr_delta_ge=-.1,
    test_ssim_delta_ge=-.001,gaussian_count_ratio_le=1.05,training_wall_ratio_le=2.)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def local(name):return Path('C:/'+name[7:]) if sys.platform=='win32' and name.startswith('/mnt/c/') else Path(name)
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
def config(p):
    expression=ast.parse(p.read_text(),mode='eval').body
    assert isinstance(expression,ast.Call) and isinstance(expression.func,ast.Name) and expression.func.id=='Namespace'
    return {k.arg:ast.literal_eval(k.value) for k in expression.keywords}
def vertex_count(p):
    with p.open('rb') as stream:
        assert stream.readline().strip()==b'ply'
        for _ in range(1000):
            line=stream.readline().decode('ascii').strip()
            if line.startswith('element vertex '):return int(line.split()[-1])
            assert line!='end_header','Vertex count missing'
    raise ValueError('Invalid PLY header')
def stream_audit(folder,label,train_names,allow_partial):
    cfg=config(folder/'cfg_args')
    assert cfg['eval'] is True and cfg['resolution']==2 and cfg['kernel_size']==0. and cfg['white_background'] is False
    assert cfg['use_decoupled_appearance']==3
    raw=(folder/'history_gap_steps.jsonl').read_bytes();lines=raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        assert allow_partial;lines.pop()
    rows=[json.loads(line) for line in lines];assert rows
    codes=collections.Counter();empty=0;node_count=0;grad_absent=collections.Counter();camera=[]
    for i,row in enumerate(rows):
        assert row['iteration']==7000+i and row['camera'] in train_names
        assert row['opacity_mask_unchanged_by_Adam'] is True
        for key in ['baseline_loss','rgb','ncc','geo','depth_normal','objective','combined_backward_wall_ms','iteration_tail_wall_ms']:
            assert math.isfinite(row[key]),(i,key)
        assert row['gaussians_before_backward']>0 and row['gaussians_at_optimizer']>0
        assert row['opaque_before_backward']>=0 and row['opaque_at_optimizer']>=0
        assert row['depth_normal']==0.
        # train.py uses separate Float32 tensor operations, not Python Float64.
        f32_sum=np.float32(np.float32(row['rgb'])+np.float32(np.float32(.6)*np.float32(row['ncc'])))
        f32_sum=np.float32(f32_sum+np.float32(np.float32(.02)*np.float32(row['geo'])))
        assert float(f32_sum)==row['baseline_loss'],(i,float(f32_sum),row['baseline_loss'])
        camera.append(row['camera']);grad_absent.update(row['gradient_groups_absent_at_optimizer'])
        if label=='control':
            assert row['regularizer'] is None;close(row['objective'],row['baseline_loss'])
        else:
            assert row['RNG_unchanged'] is True and 0<=row['regularizer']<=math.log(4)**2+1e-12
            # Original loss is a Float32 one-element tensor; adding the zero-dim
            # Float64 regularizer follows PyTorch's dimensioned-tensor promotion.
            objective=np.float32(np.float32(row['baseline_loss'])+np.float32(.01*row['regularizer']))
            assert float(objective)==row['objective'],(i,float(objective),row['objective'])
            pixels=row['pixels'];assert len(pixels)==len(set(pixels))==128
            image=local(cfg['source_path'])/cfg['images']/(row['camera']+'.png')
            with Image.open(image) as im:w,h=im.size
            H,W=round(h/2),round(w/2)
            predicted=np.random.default_rng(np.random.SeedSequence([13023,row['iteration'],int(row['camera'])])).choice(H*W,128,replace=False)
            assert np.array_equal(pixels,predicted)
            assert sum(row['status_counts'].values())+row['empty']==128
            assert set(row['status_counts'])<=set(['1','2','3','4'])
            assert 0<=row['empty']<=128 and row['nodes']>=128-row['empty']
            codes.update(row['status_counts']);empty+=row['empty'];node_count+=row['nodes']
    return rows,dict(updates=len(rows),last_iteration=rows[-1]['iteration'],camera_sequence_sha256=hashlib.sha256(json.dumps(camera,separators=(',',':')).encode()).hexdigest(),
        native_status_counts=dict(codes),empty_samples=empty,node_count=node_count,gradient_absent_at_optimizer=dict(grad_absent),
        final_gaussians=rows[-1]['gaussians_at_optimizer'],interval_wall_ms_median=statistics.median(x['iteration_tail_wall_ms'] for x in rows[1:]) if len(rows)>1 else None,
        timing_scope='Intervals after first include original forward/NCC/topology, capture, backward, Adam and checks; excludes separate JSONL/progress write and next clock reset. Not a speed benchmark.',
        stream_sha256=hashlib.sha256(raw).hexdigest())
def verify_final_state(folder,record):
    import torch
    from research.checkpoint_state import state_digest
    cp=folder/'chkpnt30000.pth';sidepath=folder/'chkpnt30000.runtime.pth'
    assert sha(cp)==record['final_checkpoint']['sha256'] and sha(sidepath)==record['final_checkpoint']['sidecar_sha256']
    params,it=torch.load(cp,map_location='cpu',weights_only=False)
    side=torch.load(sidepath,map_location='cpu',weights_only=False)
    assert it==side['iteration']==30000 and side['checkpoint_sha256']==sha(cp)
    digest=state_digest((params,side['supplemental']))
    assert digest==side['initial_state_sha256']==record['final_checkpoint']['state_sha256']
    assert len(params[2])==vertex_count(folder/'point_cloud/iteration_30000/point_cloud.ply')
    return dict(state_sha256=digest,checkpoint_sha256=sha(cp),sidecar_sha256=sha(sidepath),gaussians=len(params[2]))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=R/'experiments/history_gap_pilot_v1')
    parser.add_argument('--running',action='store_true');parser.add_argument('--out',type=Path,required=True)
    opts=parser.parse_args();run=opts.run.resolve();assert not opts.out.exists()
    m=read(run/'manifest.json');assert m['gate']==FIXED_GATE
    assert len(m['train_names'])==42 and len(m['test_names'])==7 and not set(m['train_names'])&set(m['test_names'])
    for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,('source',name)
    for name,h in m['input_sha256'].items():assert sha(local(name))==h,('input',name)
    assert sha(run/'sources.zip')==m['source_archive_sha256']
    with zipfile.ZipFile(run/'sources.zip') as z:
        for name,h in m['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    report=dict(epoch=time.time(),source_checks=len(m['source_sha256']),input_checks=len(m['input_sha256']),streams={},
        verifier_sha256=sha(Path(__file__)),run_manifest_observed_sha256=sha(run/'manifest.json'),
        scope='Actual source/archive/input hashes and serialized scalar/ID records. Does not independently replay all floating-point training updates.')
    rows={};records={}
    for label in LABELS:
        folder=run/label
        if not folder.exists():assert opts.running;continue
        rows[label],report['streams'][label]=stream_audit(folder,label,m['train_names'],opts.running)
        record=records[label]=read(folder/'history_gap_training_integration.json')
        for name,h in record['sources'].items():assert sha(ROOT/name)==h
        if opts.running:continue
        assert record['status']=='complete' and record['full_training_completed'] and record['parent_guards_verified']
        assert record['steps_observed']==len(rows[label])==23000 and rows[label][-1]['iteration']==29999
        assert record['steps']==rows[label][:6] and record['last_step']==rows[label][-1]
        assert record['final_backward_without_optimizer']['iteration']==30000
        assert record['weight']==(None if label=='control' else .01)
        parent=read(folder/'solver_training_integration.json');camera=read(folder/'holdout_camera_integration.json')
        assert parent['completed'] and parent['resume_state_identical'] and camera['status']=='complete'
        assert parent['initial_state_sha256']==record['initial_state_sha256']==m['initial_state_sha256']
        assert camera['pool']['train_names']==m['train_names'] and camera['pool']['test_names']==m['test_names']
        assert camera['final_test_report_render_calls']==7
        final=record['final_backward_without_optimizer']
        empty_mask_candidates=[x['iteration'] for x in [*rows[label],final] if x['ncc']==0. and x['geo']==0.]
        # PatchMatch returns zero before calling warp NCC if its d_mask is empty.
        # This log/source reconciliation is not a saved-mask replay.
        assert camera['ncc_calls']+len(empty_mask_candidates)==23001
        report['streams'][label]['zero_NCC_geo_iterations']=empty_mask_candidates
        report['streams'][label]['actual_warp_NCC_calls']=camera['ncc_calls']
        assert not set(camera['ncc_target_names_seen'])&set(m['test_names'])
        for name,h in record['artifacts_sha256'].items():assert sha(folder/name)==h
        for b in [record['binaries']['query'],record['binaries']['csr']]:
            assert sha(local(b['binary_path']) if 'binary_path' in b else ROOT/b['binary_relative_path'])==b['binary_sha256']
        assert sha(ROOT/record['binaries']['loss']['path'])==record['binaries']['loss']['sha256']
        with zipfile.ZipFile(folder/'history_gap_sources.zip') as z:
            for name,h in record['sources'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        report['streams'][label]['final_state']=verify_final_state(folder,record)
    if opts.running:
        report.update(status='partial',audit_gate_pass=False,quality_audited=False,observed_controller_status=m['status'])
    else:
        assert m['status']=='complete' and len(rows)==2
        assert [r['camera'] for r in rows['control']]==[r['camera'] for r in rows['history_gap']]
        assert records['control']['binaries']==records['history_gap']['binaries']
        assert all(records['history_gap']['steps'][0]['initial_query_CSR_public_bit_parity'].values())
        for name,h in m['artifacts_sha256'].items():assert sha(run/name)==h
        expected=['train_control','train_history_gap']
        for label in LABELS:expected += ['extract_'+label,'align_'+label,*[f'eval_{label}_seed{s}' for s in [0,1,2]],f'images_{label}_train',f'images_{label}_test']
        assert [j['stage'] for j in m['jobs']]==expected
        assert all(j['status']=='complete' and j['returncode']==0 for j in m['jobs'])
        q=read(run/'paired_quality.json');assert sha(run/'paired_quality.json')==m['quality_sha256']
        images={};means={};align=[]
        for label in LABELS:
            folder=run/label;images[label]={}
            ply=folder/'point_cloud/iteration_30000/point_cloud.ply'
            a=read(folder/'alignment.json');align.append(a)
            assert sha(folder/'recon_post.ply')==a['source_mesh_sha256'] and sha(folder/'recon_aligned.ply')==a['aligned_mesh_sha256']
            assert a['camera_names']==m['train_names'] and a['official_zero_based_ids']==[int(n) for n in m['train_names']]
            values=[]
            for seed in [0,1,2]:
                directory=run/f'eval_{label}_seed{seed}';ev=read(directory/'manifest.json');raw=read(directory/'results.json')
                assert ev['status']=='complete' and ev['returncode']==0 and ev['shuffle_seed']==seed and ev['scan']==55
                for name,h in ev['source_sha256'].items():assert sha(local(name))==h
                close(raw['overall'],(raw['mean_d2s']+raw['mean_s2d'])/2)
                assert raw==q['meshes'][label][seed];values.append(raw)
            means[label]={k:statistics.mean(v[k] for v in values) for k in ['overall','mean_d2s','mean_s2d']}
            for split in ['train','test']:
                im=images[label][split]=read(run/f'images_{label}_{split}.json')
                assert im['rows'] and [r['camera'] for r in im['rows']]==m[split+'_names']
                assert im['camera_count']==len(im['rows']) and not im['partial_smoke']
                assert im['is_novel_view_evaluation']==(split=='test') and im['split']==split and im['iteration']==30000
                assert im['gaussian_count']==vertex_count(ply) and im['source_sha256']['point_cloud.ply']==sha(ply)
                assert im['source_sha256']['cfg_args']==sha(folder/'cfg_args') and im['source_sha256']['evaluate_images.py']==sha(R/'evaluate_images.py')
                for k,v in im['mean'].items():close(v,statistics.mean(r[k] for r in im['rows']))
                assert im==q['images'][label][split]
        for key in ['scale','rotation','translation']:assert np.array_equal(align[0][key],align[1][key])
        ratios={k:means['history_gap'][k]/means['control'][k] for k in means['control']}
        deltas={k:images['history_gap']['test']['mean'][k]-images['control']['test']['mean'][k] for k in ['psnr_global_rgb_db','ssim_repository']}
        capacity=images['history_gap']['test']['gaussian_count']/images['control']['test']['gaussian_count']
        wall=m['jobs'][1]['wall_seconds']/m['jobs'][0]['wall_seconds']
        gates=dict(geometry=ratios['overall']<=.99,d2s=ratios['mean_d2s']<=1.,s2d=ratios['mean_s2d']<=1.,
            test_psnr=deltas['psnr_global_rgb_db']>=-.1,test_ssim=deltas['ssim_repository']>=-.001,capacity=capacity<=1.05,resource_cost=wall<=2.)
        assert gates==q['gates'] and all(gates.values())==q['gate_pass']==m['gate_pass']
        for k,v in ratios.items():close(v,q['mesh_ratios'][k])
        for k,v in deltas.items():close(v,q['test_deltas'][k])
        close(capacity,q['gaussian_count_ratio']);close(wall,q['training_wall_ratio'])
        report.update(status='complete',audit_gate_pass=True,pilot_gate_pass=all(gates.values()),gates=gates,
            mesh_ratios=ratios,test_deltas=deltas,gaussian_count_ratio=capacity,training_wall_ratio=wall,
            quality_audited=True,limitations=m['limitations'])
    opts.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
