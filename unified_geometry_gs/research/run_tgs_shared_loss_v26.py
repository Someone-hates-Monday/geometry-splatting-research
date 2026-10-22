"""Validate shared fields against unshared v24 bits on the same32 real images."""
import argparse,gc,json,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new
from tgs_image_metrics_v11 import ssim_map
from tgs_local_image_loss_common_v24 import backend as old_backend
from run_tgs_local_image_loss_v24 import check_batch
from run_tgs_local_split_v19 import same_bits
from tgs_shared_image_loss_common_v26 import OUT,backend
V23=ROOT/'experiments/tgs_view_transfer_v23';V24=ROOT/'experiments/tgs_local_image_loss_v24';V25=ROOT/'experiments/tgs_packed_query_v25';V20=ROOT/'experiments/tgs_local_stress_v20'

def prepare():
    OUT.mkdir();assert json.loads((V25/'results.json').read_text())['all_passed']
    files=[Path(__file__),ROOT/'tgs_shared_image_loss_v26.cu',ROOT/'tgs_shared_image_loss_common_v26.py',ROOT/'TGS_SHARED_LOSS_PROTOCOL_V26.md',
        ROOT/'run_tgs_local_image_loss_v24.py',ROOT/'run_tgs_local_split_v19.py',ROOT/'tgs_packed_query_common_v25.py',ROOT/'tgs_local_image_loss_common_v24.py',
        V24/'build_receipt.json',V24/'results.json',V24/'evidence_manifest.json',V25/'plan.json',V25/'runtime_dependency_addendum.json',V25/'evidence_manifest.json']
    files.extend(ROOT/n for n in json.loads((V24/'plan.json').read_text())['inputs']);files.append(Path(json.loads((V24/'build_receipt.json').read_text())['path']))
    files=sorted(set(files));write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},real_cases=32,map_tolerance=1e-10,score_atol=1e-13,score_rtol=1e-9,
        expected_cached_calls_per_batch=3,expected_cache_preps_per_batch=1,new_native_forwards=0,training=False,all_outcomes_previously_observed=True))
    print('SHARED_LOSS_PLAN_FROZEN',flush=True)

def frozen():
    p=json.loads((OUT/'plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    return p

def build():
    frozen();folder=OUT/'build';folder.mkdir();os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'];m=load(name='tgs_shared_image_loss_v26',sources=[str(ROOT/'tgs_shared_image_loss_v26.cu')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false'],build_directory=str(folder),verbose=True)
    write_new(OUT/'build_receipt.json',dict(path=m.__file__,sha256=sha(Path(m.__file__)),source_sha256=sha(ROOT/'tgs_shared_image_loss_v26.cu'),
        flags=['-O3','--fmad=false','sm_120'],torch=torch.__version__,cuda=torch.version.cuda,training=False));print('SHARED_LOSS_BUILD_DONE',flush=True)

class Adapter:
    def __init__(self,new,old):self.new=new;self.old=old;self.shared=None;self.calls=[]
    def evaluate(self,base,gt,patch,boxes,ro,vo,vb,ao,ab,weights,totals,debug):
        if self.shared is None:
            prepared=self.new.prepare(base,gt,weights);assert not any(prepared[1].cpu().tolist());self.shared=prepared[0];self.base=base;self.gt=gt;self.weights=weights
        assert self.base.data_ptr()==base.data_ptr() and self.gt.data_ptr()==gt.data_ptr() and same_bits(weights,self.weights)
        new=self.new.evaluate(base,gt,patch,self.shared,boxes,ro,vo,vb,ao,ab,weights,totals,debug)
        old=self.old.evaluate(base,gt,patch,boxes,ro,vo,vb,ao,ab,weights,totals,debug);torch.cuda.synchronize()
        assert not any(new[1].cpu().tolist()) and not any(old[1].cpu().tolist())
        assert same_bits(new[0],old[0]) and same_bits(new[2],old[2])
        self.calls.append(dict(debug=debug,scores_literal_unshared=True,maps_literal_unshared=True))
        return new
    def annotate(self,out):
        assert len(self.calls)==3
        out.update(calls=self.calls,cache_bytes=self.shared.numel()*self.shared.element_size(),cache_preps=1,
            unshared_vertical_bytes=out['temporary_vertical_bytes'],temporary_vertical_bytes=3*8*out['layout_totals'][1])
        return out

@torch.no_grad()
def fixtures():
    frozen();m=backend();old=old_backend();H,W=13,19;base=np.linspace(-.1,1.2,3*H*W,dtype=np.float32).reshape(3,H,W);gt=np.arange(H*W*3,dtype=np.uint16).reshape(H,W,3).astype(np.uint8)
    rois=[[0,0,0,0],[0,0,1,1],[W-1,H-1,W,H],[0,0,W,H],[0,H-1,W,H],[0,0,1,H],[2,3,7,9],[2,3,7,9]];patches=[];expected=[];hashes=[];G=gt.astype(np.float64)/255.;B=base.transpose(1,2,0).astype(np.float64);bs=ssim_map(B,G)
    for e,r in enumerate(rois):
        x0,y0,x1,y1=r;patch=base[:,y0:y1,x0:x1].copy()
        if e not in [0,5]:patch+=np.float32(.017)
        trial=base.copy();trial[:,y0:y1,x0:x1]=patch;T=trial.transpose(1,2,0).astype(np.float64);d=T-G;b=B-G
        expected.append([float(np.mean(d*d-b*b)),float(np.mean(np.abs(d)-np.abs(b))),float(np.mean(ssim_map(T,G)-bs))]);hashes.append(ahash(trial));patches.append(patch)
    adapter=Adapter(m,old);out=adapter.annotate(check_batch(adapter,base,gt,rois,patches,expected,hashes,'synthetic_shared'))
    assert out['rows'][0]['score']==[0,0,0] and out['rows'][5]['score']==[0,0,0] and out['rows'][6]['score']==out['rows'][7]['score']
    write_new(OUT/'fixtures.json',dict(all_passed=True,software_only=True,batch=out))

@torch.no_grad()
def validate():
    frozen();assert json.loads((OUT/'fixtures.json').read_text())['all_passed'];m=backend();old=old_backend();a=json.loads((V23/'analysis.json').read_text());rows=[];batches=[]
    for scene in ['flowers','room']:
        for view in ['fresh','old_0']:
            refs=[r for r in a['rows'] if r['scene']==scene and r['view']==view]
            base=np.load(V20/scene/'baseline.npz')['rgb'] if view=='fresh' else np.load(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy')
            gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy' if view=='fresh' else ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy');rois=[];patches=[]
            for en in refs:
                path=V20/scene/f"parent_{en['parent_index']}"/en['appearance']/'query_patch.npz' if view=='fresh' else V23/scene/f"proposal_{en['event']}_rgb_patch.npz"
                data=np.load(path);rois.append(data['bbox'].tolist());patches.append(data['rgb'].copy())
            adapter=Adapter(m,old);out=adapter.annotate(check_batch(adapter,base,gt,rois,patches,[[r['changes'][k] for k in ['raw_MSE','raw_L1','raw_SSIM']] for r in refs],[r['rgb_sha256'] for r in refs],scene+'/'+view))
            for row,en in zip(out['rows'],refs):row.update(scene=scene,view=view,parent_id=en['parent_id'],appearance=en['appearance'],cached_unshared_scores_and_maps_literal=True);rows.append(row)
            write_new(OUT/f'{scene}_{view}.json',out);batches.append(dict(scene=scene,view=view,cache_bytes=out['cache_bytes'],cache_preps=1,cached_GPU_calls=3))
            del adapter;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==32;frozen();write_new(OUT/'results.json',dict(all_passed=True,rows=rows,batches=batches,new_native_forwards=0,training=False,
        cache_invalidated_by_caller=True,source_sha256=sha(Path(__file__)),timing=False));print('ALL32_SHARED_CACHE_LOSSES_LITERAL',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','build','fixtures','validate']);args=parser.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
