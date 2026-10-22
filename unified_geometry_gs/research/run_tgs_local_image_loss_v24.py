"""Freeze/build/validate FP64 local loss on all32 existing real images; no renderer."""
import argparse,gc,json,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,exact
from tgs_image_metrics_v11 import ssim_map
from tgs_local_image_loss_common_v24 import OUT,layout,device_layout,packed_patches,evaluate,backend
V23=ROOT/'experiments/tgs_view_transfer_v23';V20=ROOT/'experiments/tgs_local_stress_v20'
MAP_TOL=1e-10;SCORE_ATOL=1e-13;SCORE_RTOL=1e-9
KEYS=['raw_MSE','raw_L1','raw_SSIM']

def prepare():
    OUT.mkdir();a=json.loads((V23/'analysis.json').read_text());assert a['all_verified']
    files=[Path(__file__),ROOT/'tgs_local_image_loss_v24.cu',ROOT/'tgs_local_image_loss_common_v24.py',ROOT/'TGS_LOCAL_IMAGE_LOSS_PROTOCOL_V24.md',
        ROOT/'tgs_image_metrics_v11.py',ROOT/'tgs_split_color_common_v14.py',V23/'analysis.json',V23/'results.json',V23/'evidence_manifest.json',V23/'loss_locality_results.json',ROOT/'TGS_IMAGE_LOSS_LOCALITY_NOTE_V23.md']
    old=json.loads((V23/'plan.json').read_text());files.extend(ROOT/n for n in old['inputs'] if n.endswith(('/query_patch.npz','/gt_rgb.npy','/baseline.npz','/baseline_0/rgb.npy')))
    files.extend(V23/r['scene']/f"proposal_{r['event']}_rgb_patch.npz" for r in a['rows'] if r['view']=='old_0')
    files=sorted(set(files));write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},real_cases=32,views=4,
        map_tolerance=MAP_TOL,score_atol=SCORE_ATOL,score_rtol=SCORE_RTOL,new_native_forwards=0,training=False,already_observed_outcomes=True))
    print('LOCAL_LOSS_PLAN_FROZEN',flush=True)

def frozen():
    p=json.loads((OUT/'plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    assert (p['map_tolerance'],p['score_atol'],p['score_rtol'])==(MAP_TOL,SCORE_ATOL,SCORE_RTOL);return p

def build():
    frozen();folder=OUT/'build';folder.mkdir();os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'];m=load(name='tgs_local_image_loss_v24',sources=[str(ROOT/'tgs_local_image_loss_v24.cu')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false'],build_directory=str(folder),verbose=True)
    write_new(OUT/'build_receipt.json',dict(path=m.__file__,sha256=sha(Path(m.__file__)),source_sha256=sha(ROOT/'tgs_local_image_loss_v24.cu'),
        flags=['-O3','--fmad=false','sm_120'],torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),training=False))
    print('LOCAL_LOSS_BUILD_DONE',flush=True)

def mapped(a,host,e):
    start,end=host['prefix'][3][e:e+2];b=host['boxes'][e];height,width=int(b[7]-b[5]),int(b[6]-b[4])
    return a[int(start):int(end)].reshape(3,height,width,2).transpose(1,2,0,3)

def score_gate(value,expected):
    value=np.asarray(value,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64);errors=np.abs(value-expected);limits=SCORE_ATOL+SCORE_RTOL*np.abs(expected)
    assert np.isfinite(value).all() and np.all(errors<=limits),(value.tolist(),expected.tolist(),errors.tolist(),limits.tolist())
    return errors.tolist()

def check_batch(m,base,gt,rois,patches,expected,hashes,label):
    H,W=gt.shape[:2];K=len(rois);assert base.shape==(3,H,W) and base.dtype==np.float32 and gt.dtype==np.uint8
    host=layout(rois,W,H);dev=device_layout(host);b=torch.from_numpy(np.ascontiguousarray(base)).cuda();g=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda();p=packed_patches(patches)
    local=evaluate(m,b,g,p,dev,True);normal=evaluate(m,b,g,p,dev,False);torch.cuda.synchronize()
    assert not any(local[1].cpu().tolist()) and not any(normal[1].cpu().tolist()) and normal[2].numel()==0
    assert exact(local[0].cpu().numpy(),normal[0].cpu().numpy());ls=local[0].cpu().numpy();lm=local[2].cpu().numpy()
    trials=[]
    for e,(r,patch) in enumerate(zip(rois,patches)):
        x0,y0,x1,y1=r;trial=base.copy();trial[:,y0:y1,x0:x1]=patch;assert ahash(trial)==hashes[e];trials.append(trial)
    fh=layout([[0,0,W,H]]*K,W,H);fd=device_layout(fh);fp=packed_patches(trials);full=evaluate(m,b,g,fp,fd,True);torch.cuda.synchronize()
    assert not any(full[1].cpu().tolist());fs=full[0].cpu().numpy();fm=full[2].cpu().numpy();baseline=base.transpose(1,2,0).astype(np.float64);G=gt.astype(np.float64)/255.;cpu_base=ssim_map(baseline,G);rows=[]
    for e,trial in enumerate(trials):
        r=rois[e];bb=host['boxes'][e];ax0,ay0,ax1,ay1=map(int,bb[4:8]);localmap=mapped(lm,host,e);fullmap=mapped(fm,fh,e)
        assert exact(localmap,fullmap[ay0:ay1,ax0:ax1])
        outside=np.ones((H,W),dtype=bool);outside[ay0:ay1,ax0:ax1]=False;assert exact(fullmap[:,:,:,0][outside],fullmap[:,:,:,1][outside])
        cpu_trial=ssim_map(trial.transpose(1,2,0).astype(np.float64),G)
        cpu=np.stack([cpu_base,cpu_trial],axis=-1);map_error=float(np.max(np.abs(fullmap-cpu),initial=0));assert np.isfinite(fullmap).all() and map_error<=MAP_TOL,(label,e,map_error)
        error=score_gate(ls[e],expected[e]);full_error=score_gate(fs[e],expected[e]);local_full_error=score_gate(ls[e],fs[e])
        blend=np.array([.8*ls[e,1]-.2*ls[e,2],.8*ls[e,0]-.2*ls[e,2]])
        exp=np.asarray(expected[e]);reference_blend=np.array([.8*exp[1]-.2*exp[2],.8*exp[0]-.2*exp[2]]);blend_error=score_gate(blend,reference_blend)
        allowance=SCORE_ATOL+SCORE_RTOL*np.abs(reference_blend)
        row=dict(event=e,ROI=r,score=ls[e].tolist(),expected=list(map(float,expected[e])),full_gpu_score=fs[e].tolist(),
            absolute_error=error,full_absolute_error=full_error,local_full_scalar_error=local_full_error,max_CPU_map_error=map_error,
            local_full_maps_literal=True,full_map_change_outside_A_literal_zero=True,normal_debug_scalars_literal=True,
            blends=blend.tolist(),reference_blends=reference_blend.tolist(),blend_absolute_error=blend_error,
            ambiguous_blend_signs=(np.abs(reference_blend)<=allowance).tolist(),base_hash=ahash(base),trial_hash=hashes[e],
            local_map_hash=ahash(localmap),full_map_hash=ahash(fullmap))
        rows.append(row);print(json.dumps(dict(label=label,event=e,max_score_error=max(error),max_CPU_map_error=map_error,map_literal=True)),flush=True)
    # Only compact scalars/maps hashes are retained; full debug maps are reproducible.
    del dev,b,g,p,local,normal,fd,fp,full;gc.collect();torch.cuda.empty_cache()
    return dict(rows=rows,layout_boxes=host['boxes'].tolist(),layout_totals=list(map(int,host['totals'])),
        temporary_vertical_bytes=8*8*int(host['totals'][1]),temporary_partial_bytes=8*3*int(host['totals'][4]),timing=False)

def fixtures():
    frozen();m=backend();H,W=13,19;base=np.linspace(-.1,1.2,3*H*W,dtype=np.float32).reshape(3,H,W);gt=np.arange(H*W*3,dtype=np.uint16).reshape(H,W,3).astype(np.uint8)
    rois=[[0,0,0,0],[0,0,1,1],[W-1,H-1,W,H],[0,0,W,H],[0,H-1,W,H],[0,0,1,H],[2,3,7,9],[2,3,7,9]];patches=[];expected=[];hashes=[];G=gt.astype(np.float64)/255.;B=base.transpose(1,2,0).astype(np.float64);bs=ssim_map(B,G)
    for e,r in enumerate(rois):
        x0,y0,x1,y1=r;patch=base[:,y0:y1,x0:x1].copy()
        if e not in [0,5]:patch+=np.float32(.017)
        trial=base.copy();trial[:,y0:y1,x0:x1]=patch;T=trial.transpose(1,2,0).astype(np.float64);d=T-G;b=B-G
        expected.append([float(np.mean(d*d-b*b)),float(np.mean(np.abs(d)-np.abs(b))),float(np.mean(ssim_map(T,G)-bs))]);hashes.append(ahash(trial));patches.append(patch)
    out=check_batch(m,base,gt,rois,patches,expected,hashes,'synthetic');assert out['rows'][0]['score']==[0,0,0] and out['rows'][5]['score']==[0,0,0]
    assert out['rows'][6]['score']==out['rows'][7]['score'];write_new(OUT/'fixtures.json',dict(all_passed=True,software_only=True,batch=out))

def validate():
    frozen();assert json.loads((OUT/'fixtures.json').read_text())['all_passed'];m=backend();a=json.loads((V23/'analysis.json').read_text());rows=[];batches=[]
    for scene in ['flowers','room']:
        for view in ['fresh','old_0']:
            refs=[r for r in a['rows'] if r['scene']==scene and r['view']==view];assert len(refs)==8
            base=np.load(V20/scene/'baseline.npz')['rgb'] if view=='fresh' else np.load(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy')
            gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy' if view=='fresh' else ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy')
            rois=[];patches=[]
            for en in refs:
                file=V20/scene/f"parent_{en['parent_index']}"/en['appearance']/'query_patch.npz' if view=='fresh' else V23/scene/f"proposal_{en['event']}_rgb_patch.npz"
                data=np.load(file);rois.append(data['bbox'].tolist());patches.append(data['rgb'].copy())
            out=check_batch(m,base,gt,rois,patches,[[r['changes'][k] for k in KEYS] for r in refs],[r['rgb_sha256'] for r in refs],scene+'/'+view)
            for row,en in zip(out['rows'],refs):row.update(scene=scene,view=view,parent_id=en['parent_id'],appearance=en['appearance']);rows.append(row)
            write_new(OUT/f'{scene}_{view}.json',out);batches.append(dict(scene=scene,view=view,layout_totals=out['layout_totals']))
    assert len(rows)==32
    for n,h in json.loads((OUT/'plan.json').read_text())['inputs'].items():assert sha(ROOT/n)==h,n
    write_new(OUT/'results.json',dict(all_passed=True,rows=rows,batches=batches,metrics=KEYS,new_native_forwards=0,training=False,
        renderer_integration=False,timing=False,source_sha256=sha(Path(__file__)),model_quality_gain=False))
    print('ALL32_REAL_LOCAL_IMAGE_LOSSES_PASSED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','build','fixtures','validate']);args=parser.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
