"""Frozen fixed-model ray-order reference; no training or production edits."""
import argparse,gc,json,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tgs_split_color_common_v14 import ROOT,HELPER,sha,ahash,write_new,module,pack,colors,CrossWorkload,exact
from tgs_sh_isolation_common_v12 import capture
OUT=ROOT/'experiments/tgs_ray_order_v16'
SOURCE=ROOT/'tgs_ray_order_v16.cu'
OLD=ROOT/'experiments/tgs_split_color_v14'

def align(n):return (n+127)//128*128

def prepare():
    OUT.mkdir(exist_ok=True)
    old=json.loads((OLD/'plan.json').read_text());native=old['modules']['native']
    files=[SOURCE,Path(__file__),ROOT/'TGS_RAY_ORDER_PROTOCOL_V16.md',HELPER,OLD/'plan.json',
        ROOT/'tgs_split_color_common_v14.py',ROOT/'tgs_cross_scene_common_v10.py',
        ROOT/'run_tgs_real_events.py',ROOT/'tgs_sh_isolation_common_v12.py',
        ROOT/'experiments/tgs_cross_scene_v10/input_audit.json']
    for scene in ['flowers','room']:
        files.extend([ROOT/e['path'] for e in old['models'] if e['scene']==scene])
        files.extend([OLD/scene/'baseline_0'/n for n in ['receipt.json','camera.npz','rgb.npy','aux.npy','contributors.npy']])
        files.extend([ROOT/'sources/tgs_gt_identity_v11'/scene/n for n in ['original.JPG','published_gt.png']])
        files.extend([ROOT/'experiments/tgs_sh_isolation_v12'/scene/'old_posthoc/main/archived_native'/n
            for n in ['rgb.npy','aux.npy','radii.npy','maxblend.npy','contributors.npy']])
    assert sha(Path(native['path']))==native['sha256']
    write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},native=native,
        scenes=['flowers','room'],forward_cap=4,workspace_cap_bytes=8*1024**3,training=False))
    print('PLAN_FROZEN',flush=True)

def observer(build=False):
    if not build:
        r=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(r['path']))==r['sha256'];return module(r['path'])
    folder=OUT/'build';folder.mkdir()
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    m=load(name='tgs_ray_order_v16',sources=[str(SOURCE)],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(folder),verbose=True)
    write_new(OUT/'build_receipt.json',dict(path=m.__file__,sha256=sha(Path(m.__file__)),source_sha256=sha(SOURCE),torch=torch.__version__,cuda=torch.version.cuda))
    return m

def frozen():
    p=json.loads((OUT/'plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    assert sha(Path(p['native']['path']))==p['native']['sha256']
    return p

def collect(m,args,W,H,tx,ty,cap):
    c,s=m.counts(*args,W,H,tx,ty);torch.cuda.synchronize()
    offsets=torch.cat([torch.zeros(1,device='cuda',dtype=torch.int64),c.cumsum(0,dtype=torch.int64)])
    N=int(offsets[-1].item());meta=dict(hits=N,max_hits=int(c.max().item()),workspace_bytes=32*N,stats=s.cpu().tolist())
    assert 32*N<=cap,meta
    hits=m.fill(*args,W,H,tx,ty,c,offsets,s);torch.cuda.synchronize()
    return c,s,offsets,hits,meta

def sort_verify(m,args,W,H,c,s,offsets,hits):
    m.sort(c,offsets,*hits[:5]);seen=m.verify(*args[:7],W,H,c,offsets,*hits,s);torch.cuda.synchronize()
    assert bool(torch.all(seen==1));assert not any(s.cpu().tolist()),s.cpu().tolist()

def fixture_args(depths,opacity=.5):
    n=len(depths);f=lambda x:torch.tensor(x,device='cuda',dtype=torch.float32)
    no=torch.zeros((n,6),device='cuda');off=torch.full((n,3),-1.,device='cuda')
    ph=torch.full((n,2),-1.,device='cuda');sig=torch.ones(n,device='cuda');co=torch.zeros((n,4),device='cuda');co[:,3]=opacity
    ranges=torch.tensor([[0,n]],device='cuda',dtype=torch.int32);ids=torch.arange(n,device='cuda',dtype=torch.int32)
    planes=torch.tensor([[0.,0.,1.,z] for z in depths],device='cuda',dtype=torch.float64).reshape(n,4)
    return [no,off,ph,sig,co,ranges,ids,planes]

def fixtures():
    frozen();m=observer(build=True);rows=[]
    for label,depths in [('reverse',[2.,1.]),('ties',[1.,1.]),('empty',[]),('single',[1.]),('uncapped',list(range(257,0,-1)))]:
        args=fixture_args(depths);c,s,off,hits,meta=collect(m,args,1,1,1.,1.,2**30)
        col=torch.zeros((len(depths),3),device='cuda');col[:,0]=1
        if len(depths)==2:col[:]=torch.tensor([[1.,0.,0.],[0.,1.,0.]],device='cuda')
        before=m.replay(c,off,hits[0],hits[1],hits[3],col)[0].cpu().numpy()
        sort_verify(m,args,1,1,c,s,off,hits)
        expected=sorted(range(len(depths)),key=lambda i:(depths[i],i));assert hits[0].cpu().tolist()==expected
        after=m.replay(c,off,hits[0],hits[1],hits[3],col)[0].cpu().numpy()
        if label=='reverse':assert np.array_equal(before[:,0],[.5,.25,0.]) and np.array_equal(after[:,0],[.25,.5,0.])
        assert meta['hits']==len(depths)
        rows.append(dict(label=label,**meta,sorted_ids=expected,passed=True))
    args=fixture_args([-1.]);c,s,off,hits,meta=collect(m,args,1,1,1.,1.,2**30);assert s[1].item()==1
    rows.append(dict(label='invalid_depth_flag',**meta,passed=True))
    write_new(OUT/'fixtures.json',dict(passed=True,rows=rows,software_fixtures_not_physical_proofs=True))
    print('FIXTURES_PASSED',flush=True)

def call(native,state,cam,col):
    empty=torch.empty(0,device='cuda')
    return native.rasterize_triangles(torch.zeros(3,device='cuda'),state['points'],state['sigma'],state['counts'],state['offsets'],col,
        state['opacity'],state['zero'].clone(),state['zero'].clone(),cam['view'],cam['projection'],len(state['points']),
        cam['tx'],cam['ty'],cam['h'],cam['w'],empty,3,cam['center'],False,False)

def cache(res,state,cam):
    n=len(state['points']);W,H=cam['w'],cam['h'];hw=W*H;raw=res[4];pos=0;fields={}
    for name,size in [('depth',4),('clamped',3),('radii',4),('anchor',8),('conic',16),('phi',8),('rgb',12)]:
        pos=align(pos);fields[name]=pos;pos+=size*n
    maxoff=raw.numel()-128-8*n;minoff=maxoff-align(8*n);pw=minoff-align(12*n);normals=pw-align(24*n);offsets=normals-align(12*n)
    def get(at,width):return raw.narrow(0,at,4*n*width).view(torch.float32).reshape(n,width)
    xyz=state['points'].double()@cam['view'][:3,:3].double()+cam['view'][3,:3].double()
    normal=torch.linalg.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);d=(normal*xyz[:,0]).sum(1)
    plane=torch.cat([normal,d[:,None]],1).contiguous()
    ranges=res[6].narrow(0,align(align(12*hw)+8*hw),8*((W+15)//16)*((H+15)//16)).view(torch.int32)
    ids=res[5].narrow(0,0,4*int(res[0])).view(torch.int32)
    return [get(normals,6),get(offsets,3),get(fields['phi'],2),state['sigma'].flatten(),get(fields['conic'],4),ranges,ids,plane]

@torch.no_grad()
def real():
    plan=frozen();assert json.loads((OUT/'fixtures.json').read_text())['passed'];m=observer();native=module(plan['native']['path']);helper=module(HELPER);rows=[];forwards=0
    for scene in plan['scenes']:
        folder=OUT/scene;folder.mkdir();w=CrossWorkload(scene);before=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh);cam=w.cameras[0];W,H=cam['w'],cam['h']
        p=state['points'].mean(1);q=p[:,2:3].repeat(1,3);direction=q-cam['center'][None];direction/=direction.norm(dim=1,keepdim=True)
        cols={'physical':colors(helper,state,cam),'legacy':(helper.eval_sh(3,state['sh'].transpose(1,2),direction)+.5).clamp_min_(0).contiguous()}
        refs={};native_arrays={}
        for appearance,col in cols.items():
            res=call(native,state,cam,col);torch.cuda.synchronize();forwards+=1;arrays=capture(res,W,H);native_arrays[appearance]=arrays
            if appearance=='physical':
                expected=json.loads((OLD/scene/'baseline_0/receipt.json').read_text())['arrays']
                for k,a in arrays.items():assert ahash(a)==expected[k]['sha256'],(scene,k)
                args=cache(res,state,cam);c,s,offsets,hits,meta=collect(m,args,W,H,cam['tx'],cam['ty'],plan['workspace_cap_bytes'])
            else:
                prior=ROOT/'experiments/tgs_sh_isolation_v12'/scene/'old_posthoc/main/archived_native'
                assert np.max(np.abs(arrays['rgb']-np.load(prior/'rgb.npy')))<=2e-4
                for k in ['aux','radii','maxblend','contributors']:assert exact(arrays[k],np.load(prior/(k+'.npy'))),(scene,k)
            replay=m.replay(c,offsets,hits[0],hits[1],hits[3],col);torch.cuda.synchronize()
            got=[t.cpu().numpy() for t in replay];checks=dict(rgb=exact(got[0].reshape(3,H,W),arrays['rgb']),
                alpha=exact(got[1].reshape(H,W),arrays['aux'][1]),paths=exact(got[2].reshape(2,H,W).view(np.uint32),arrays['contributors']))
            write_new(folder/(appearance+'_center_gate.json'),dict(checks=checks,max_rgb=float(np.max(np.abs(got[0].reshape(3,H,W)-arrays['rgb'])))))
            assert all(checks.values()),(scene,appearance,checks)
            np.save(folder/(appearance+'_native.npy'),arrays['rgb']);refs[appearance]=dict(center_gate=checks)
            del res,arrays,replay,got
        assert not any(s.cpu().tolist()),s.cpu().tolist()
        sort_verify(m,args,W,H,c,s,offsets,hits)
        for appearance,col in cols.items():
            rgb,alpha,paths=m.replay(c,offsets,hits[0],hits[1],hits[3],col);torch.cuda.synchronize()
            np.save(folder/(appearance+'_ray.npy'),rgb.cpu().numpy().reshape(3,H,W))
        # Save frozen sparse sorted packets for an independent order audit.
        pixels=sorted(set([0,W//2+(H//2)*W,int(c.argmax().item()),W*H-1]))
        for pix in pixels:
            start=int(offsets[pix].item());end=int(offsets[pix+1].item())
            np.savez_compressed(folder/f'packet_{pix}.npz',pixel=np.array([pix%W,pix//W]),ids=hits[0][start:end].cpu().numpy(),alpha=hits[1][start:end].cpu().numpy(),
                z=hits[2][start:end].cpu().numpy(),ordinal=hits[3][start:end].cpu().numpy(),rank=hits[4][start:end].cpu().numpy(),
                original=hits[5][start:end].cpu().numpy(),planes=args[-1][hits[0][start:end].long()].cpu().numpy(),
                ray=np.array([((2*(pix%W)+1)/W-1)*cam['tx'],((2*(pix//W)+1)/H-1)*cam['ty'],1.]))
        assert w.input_hashes()==before
        write_new(folder/'receipt.json',dict(**meta,stats_final=s.cpu().tolist(),appearances=refs,input_unchanged=True))
        rows.append(dict(scene=scene,**meta,correctness_passed=True));print(json.dumps(rows[-1]),flush=True)
        del w,state,args,c,s,offsets,hits,cols,native_arrays,p,q,direction,rgb,alpha,paths;gc.collect();torch.cuda.empty_cache()
    assert forwards==4
    write_new(OUT/'results.json',dict(rows=rows,full_forwards=forwards,all_correctness_gates=True,training=False,quality_not_yet_scored=True))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','fixtures','real']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
