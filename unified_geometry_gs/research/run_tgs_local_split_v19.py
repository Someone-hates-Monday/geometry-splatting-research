"""Complete fixed-input cached RGB/alpha/path subdivision reference. No training."""
import argparse,gc,json,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,module,pack,split,CrossWorkload,exact
from tgs_sh_isolation_common_v12 import capture
from trace_tgs_window_scene_v15 import call
OUT=ROOT/'experiments/tgs_local_split_v19';OLD=ROOT/'experiments/tgs_split_color_v14';SOURCE=ROOT/'tgs_local_split_v19.cu'

def align(n):return (n+127)//128*128

def cache(res,state,cam):
    n=len(state['points']);hw=cam['w']*cam['h'];tiles=((cam['w']+15)//16)*((cam['h']+15)//16);raw=res[4];pos=0;fields={}
    for name,size in [('depth',4),('clamped',3),('radii',4),('anchor',8),('conic',16),('phi',8),('rgb',12)]:
        pos=align(pos);fields[name]=pos;pos+=size*n
    hi=raw.numel()-128-8*n;lo=hi-align(8*n);pw=lo-align(12*n);normal=pw-align(24*n);offset=normal-align(12*n)
    def f(at,width):return raw.narrow(0,at,4*n*width).view(torch.float32).reshape(n,width)
    def rect(at):return raw.narrow(0,at,8*n).view(torch.int32).reshape(n,2)
    geometry=[f(fields['depth'],1),f(normal,6),f(offset,3),f(fields['phi'],2),f(fields['conic'],4),state['sigma'].reshape(n,1),res[3],rect(lo),rect(hi)]
    ranges=res[6].narrow(0,align(align(12*hw)+8*hw),8*tiles).view(torch.int32)
    ids=res[5].narrow(0,0,4*int(res[0])).view(torch.int32)
    paths=res[6].narrow(0,align(12*hw),8*hw).view(torch.int32).reshape(2,cam['h'],cam['w'])
    return geometry,ranges,ids,paths

def same_bits(a,b):return a.shape==b.shape and bool(torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32)))

def cache_gate(base,child,reference,parent):
    n=len(base[6]);visible=base[6]>0;assert same_bits(torch.cat([base[6][:parent],base[6][parent+1:]]),reference[6][:-4])
    for field,(b,c,r) in enumerate(zip(base,child,reference)):
        if field==6:continue
        mask=visible.clone();mask[parent]=False;old_ids=torch.nonzero(mask).flatten();new_ids=old_ids-(old_ids>parent).long()
        assert same_bits(b.index_select(0,old_ids),r.index_select(0,new_ids)),('external cache',field)
        active=child[6]>0
        assert same_bits(c[active],r[-4:][active]),('child cache',field)
    assert same_bits(child[6],reference[6][-4:])

def prepare():
    OUT.mkdir(exist_ok=True);old=json.loads((OLD/'plan.json').read_text());results=json.loads((OLD/'results.json').read_text())
    paths=[SOURCE,Path(__file__),ROOT/'TGS_LOCAL_SPLIT_PROTOCOL_V19.md',OLD/'plan.json',OLD/'results.json',OLD/'evidence_manifest.json',
        ROOT/'tgs_split_color_common_v14.py',ROOT/'tgs_cross_scene_common_v10.py',ROOT/'run_tgs_real_events.py',
        ROOT/'tgs_sh_isolation_common_v12.py',ROOT/'trace_tgs_window_scene_v15.py',ROOT/'sources/tgs_native_diagnostic_v1/native/cuda_rasterizer/forward.cu',
        ROOT/'sources/tgs_native_diagnostic_v1/native/cuda_rasterizer/rasterizer_impl.cu']
    paths.extend(ROOT/m['path'] for m in old['models'])
    for scene in ['flowers','room']:
        for ci in [0,1]:paths.extend(OLD/scene/f'baseline_{ci}'/n for n in ['receipt.json','colors.npy','camera.npz'])
        for event in range(8):
            paths.append(OLD/scene/f'event_{event}/input.npz')
            paths.extend(OLD/scene/f'event_{event}'/f'camera_{ci}'/'receipt.json' for ci in [0,1])
    native=old['modules']['native'];assert sha(Path(native['path']))==native['sha256']
    write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},native=native,cases=results['cases'],
        full_forward_cap=68,training=False,timing_allowed=False))
    print('PLAN_FROZEN',flush=True)

def frozen():
    p=json.loads((OUT/'plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    assert sha(Path(p['native']['path']))==p['native']['sha256']
    return p

def load_local(build=False):
    if not build:
        r=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(r['path']))==r['sha256'];return module(r['path'])
    directory=OUT/'build';directory.mkdir()
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    m=load(name='tgs_local_split_v19',sources=[str(SOURCE)],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(directory),verbose=True)
    write_new(OUT/'build_receipt.json',dict(path=m.__file__,sha256=sha(Path(m.__file__)),source_sha256=sha(SOURCE),torch=torch.__version__,cuda=torch.version.cuda))
    return m

def fake(n,depth=None,visible=True,opacity=.5):
    depth=list(range(1,n+1)) if depth is None else depth
    f=lambda x:torch.tensor(x,device='cuda',dtype=torch.float32)
    return [f(depth).reshape(n,1),torch.zeros((n,6),device='cuda'),torch.full((n,3),-1.,device='cuda'),
        torch.full((n,2),-1.,device='cuda'),torch.cat([torch.zeros((n,3),device='cuda'),torch.full((n,1),opacity,device='cuda')],1),
        torch.ones((n,1),device='cuda'),torch.full((n,),int(visible),device='cuda',dtype=torch.int32),
        torch.zeros((n,2),device='cuda',dtype=torch.int32),torch.ones((n,2),device='cuda',dtype=torch.int32)]

@torch.no_grad()
def fixtures():
    frozen();m=load_local(build=True);rows=[]
    for name in ['stable_ties','parent_absent','empty_after_removal','unaffected_copy','all257']:
        n=258 if name=='all257' else 3;parent=0 if name in ['empty_after_removal','all257'] else 1
        b=fake(n,[1.,2.,2.] if n==3 else None,visible=name!='unaffected_copy',opacity=1/128 if n==258 else .5)
        c=fake(4,[2.,2.,2.,2.],visible=name in ['stable_ties','parent_absent'])
        old=[i for i in range(n) if name!='parent_absent' or i!=parent];children=list(range(4)) if c[6][0].item() else []
        if name=='empty_after_removal':old=[parent]
        if name=='unaffected_copy':old=[]
        merged=sorted([(float(b[0][i].item()),0,i) for i in old if i!=parent]+[(2.,1,i) for i in children])
        target=[(i if i<parent else i-1) if kind==0 else n-1+i for _,kind,i in merged]
        t=lambda x:torch.tensor(x,device='cuda',dtype=torch.int32)
        br=t([0,len(old)]);bl=t(old);cr=t([0,len(children)]);cl=t(children);rr=t([0,len(target)]);rl=t(target)
        bc=torch.ones((n,3),device='cuda');cc=torch.ones((4,3),device='cuda');rgb=torch.full((3,1,1),.25,device='cuda');alpha=torch.full((1,1),.125,device='cuda');paths=t([0,-1]).reshape(2,1,1)
        rgb2,a2,p2,mask=m.query(b,br,bl,bc,c,cr,cl,cc,parent,rgb,alpha,paths,1,1)
        s=m.audit(b,br,bl,c,cr,cl,rr,rl,parent,n,1);torch.cuda.synchronize();assert not any(s[:4].cpu().tolist())
        if name=='unaffected_copy':assert same_bits(rgb,rgb2) and same_bits(alpha,a2) and same_bits(paths,p2)
        elif name=='empty_after_removal':assert torch.all(rgb2==0) and a2.item()==0 and p2[:,0,0].cpu().tolist()==[0,-1]
        else:
            T=np.float32(1);C=np.float32(0);last=0;median=-1
            for j,(_,kind,i) in enumerate(merged):
                al=np.float32(float((b if kind==0 else c)[4][i,3].item()));test=np.float32(T*np.float32(1-al))
                if test<np.float32(1e-4):break
                if T>.5:median=j+1
                C=np.float32(C+np.float32(al*T));T=test;last=j+1
            assert rgb2[:,0,0].cpu().tolist()==[float(C)]*3 and a2.item()==float(np.float32(1-T))
            assert p2[:,0,0].cpu().tolist()==[last,median]
            if name=='all257':assert last==257 and s[4].item()==257
        rows.append(dict(case=name,passed=True,merged_records=len(target)))
    write_new(OUT/'fixtures.json',dict(all_passed=True,rows=rows,software_fixtures=True));print('FIXTURES_PASSED',flush=True)

@torch.no_grad()
def validate():
    plan=frozen();assert json.loads((OUT/'fixtures.json').read_text())['all_passed'];m=load_local();native=module(plan['native']['path']);rows=[];forwards=0
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh)
        for ci,cam in enumerate(w.cameras):
            col=torch.from_numpy(np.load(OLD/scene/f'baseline_{ci}/colors.npy')).cuda();base=call(native,state,cam,col);torch.cuda.synchronize();forwards+=1
            arrays=capture(base,cam['w'],cam['h']);expected=json.loads((OLD/scene/f'baseline_{ci}/receipt.json').read_text())['arrays']
            for k,a in arrays.items():assert ahash(a)==expected[k]['sha256'],(scene,ci,k)
            del arrays
            bg,br,bl,bpaths=cache(base,state,cam);oldalpha=base[2][1].contiguous();W,H=cam['w'],cam['h'];tiles=((W+15)//16)*((H+15)//16)
            for en in [r for r in plan['cases'] if r['scene']==scene and r['camera']==ci]:
                event=en['event'];index=en['parent_id'];changed=split(state,index);child=pack(changed['points'][-4:],changed['sigma'][-4:],changed['opacity'][-4:],changed['sh'][-4:]);cc=col[index:index+1].repeat(4,1).contiguous()
                small=call(native,child,cam,cc);torch.cuda.synchronize();forwards+=1;cg,cr,cl,_=cache(small,child,cam)
                local=m.query(bg,br,bl,col,cg,cr,cl,cc,index,base[1],oldalpha,bpaths,W,H);torch.cuda.synchronize()
                fullcol=torch.cat([col[:index],col[index+1:],cc]).contiguous();full=call(native,changed,cam,fullcol);torch.cuda.synchronize();forwards+=1
                full_arrays=capture(full,W,H)
                for k,a in full_arrays.items():assert ahash(a)==en['hold_hashes'][k]['sha256'],(scene,ci,event,k)
                fg,fr,fl,fpaths=cache(full,changed,cam);cache_gate(bg,cg,fg,index)
                audit=m.audit(bg,br,bl,cg,cr,cl,fr,fl,index,len(state['points']),tiles);torch.cuda.synchronize()
                stats=audit.cpu().tolist();assert not any(stats[:4]),(scene,ci,event,stats)
                checks=dict(rgb=same_bits(local[0],full[1]),alpha=same_bits(local[1],full[2][1]),paths=same_bits(local[2],fpaths))
                mask=local[3].reshape((H+15)//16,(W+15)//16).repeat_interleave(16,0).repeat_interleave(16,1)[:H,:W].bool()
                checks['outside_rgb']=same_bits(local[0][:,~mask],base[1][:,~mask]);checks['outside_alpha']=same_bits(local[1][~mask],oldalpha[~mask]);checks['outside_paths']=same_bits(local[2][:,~mask],bpaths[:,~mask])
                folder=OUT/f'{scene}_c{ci}_e{event}';folder.mkdir();maxerr=float((local[0]-full[1]).abs().max().item())
                out=dict(scene=scene,camera=ci,event=event,parent_id=index,checks=checks,max_RGB_error=maxerr,
                    all_defined_cache_fields_literal=True,all_tile_lists_literal=True,tile_audit=stats,affected_tiles=int(local[3].sum().item()),
                    affected_pixels=int(mask.sum().item()),pixels=W*H,query_RGB_sha256=ahash(local[0].cpu().numpy()),native_RGB_sha256=en['hold_hashes']['rgb']['sha256'],
                    child_visible=int((small[3]>0).sum().item()),native_forwards=forwards)
                write_new(folder/'receipt.json',out)
                if not all(checks.values()):
                    np.savez_compressed(folder/'mismatch.npz',local_rgb=local[0].cpu().numpy(),native_rgb=full_arrays['rgb'],mask=mask.cpu().numpy());raise AssertionError(out)
                rows.append(out);print(json.dumps(dict(scene=scene,camera=ci,event=event,literal=True,affected_tiles=out['affected_tiles'],native_forwards=forwards)),flush=True)
                del changed,child,cc,small,cg,cr,cl,local,fullcol,full,full_arrays,fg,fr,fl,fpaths,audit,mask
            del base,bg,br,bl,bpaths,oldalpha,col
        assert initial==w.input_hashes()
        del state,w;gc.collect();torch.cuda.empty_cache()
    assert forwards==68 and len(rows)==32
    write_new(OUT/'results.json',dict(rows=rows,full_forwards=forwards,all_gates_passed=True,training=False,timing_not_yet_performed=True));print('ALL32_LITERAL',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','fixtures','validate']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
