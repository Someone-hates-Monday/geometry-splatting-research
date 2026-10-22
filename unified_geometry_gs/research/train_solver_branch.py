"""Run real train.py with an isolated solver and auditable common-state resume."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from research.checkpoint_state import (restore_bound,verify_binding,supplemental,
    apply_supplemental,state_digest)
from research.median_adjquad_backend import load_backend,activate


class WarmupComplete(Exception):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--solver',choices=('control','adjquad4'),required=True)
    parser.add_argument('--branch-sidecar',type=Path)
    parser.add_argument('--warmup-stop',type=int)
    opts,train_args=parser.parse_known_args()
    if opts.branch_sidecar and opts.warmup_stop:
        raise ValueError('Warmup and resume are different stages.')
    if sys.version_info[:2]!=(3,12):
        raise RuntimeError('Frame injection audited only for CPython 3.12.')
    model=Path(train_args[train_args.index('-m')+1])
    if model.exists():
        raise ValueError('New model directory required.')
    backend=load_backend(opts.solver) # Compile before restoring any RNG state.
    from scene import GaussianModel
    import gaussian_renderer
    import warp_patch_ncc
    original_restore=GaussianModel.restore
    original_save=torch.save
    original_render=gaussian_renderer.render
    original_ncc=warp_patch_ncc.warp_patch_ncc
    GaussianModel.restore=restore_bound
    started=time.perf_counter()
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'solver':opts.solver,
        'mode':'warmup' if opts.warmup_stop else 'resume','completed':False,
        'render_calls':0,'depth_render_calls':0,'ncc_calls':0,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'train.py',Path(__file__),ROOT/'research/checkpoint_state.py',
            ROOT/'research/median_adjquad_backend.py',ROOT/'research/median_adjquad_accumulate.inc',
            ROOT/'research/median_adjquad_refine.inc')},
        'limitations':['Shared initial state does not enforce deterministic CUDA atomics or identical later trajectories.',
            'No M1 router or other stateful experimental training extensions supported.',
            'Training-view reports are not novel-view image quality.']}
    record['backend_sha256']=hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest()
    side=None
    if opts.branch_sidecar:
        side=torch.load(opts.branch_sidecar,map_location='cuda',weights_only=False)
        checkpoint=Path(train_args[train_args.index('--start_checkpoint')+1])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=side['checkpoint_sha256']:
            raise ValueError('Common checkpoint changed.')
        record['common_checkpoint_sha256']=side['checkpoint_sha256']
        record['sidecar_sha256']=hashlib.sha256(opts.branch_sidecar.read_bytes()).hexdigest()
    lines=(ROOT/'train.py').read_text().splitlines()
    anchor=[i+1 for i,s in enumerate(lines) if s=='    first_iter += 1']
    if len(anchor)!=1:
        raise ValueError('Resume anchor changed.')
    applied=False
    def trace(frame,event,arg):
        nonlocal applied
        if (side is not None and not applied and event=='line' and frame.f_code.co_name=='training'
            and Path(frame.f_code.co_filename).resolve()==ROOT/'train.py' and frame.f_lineno==anchor[0]):
            local=frame.f_locals
            if local['first_iter']!=side['iteration'] or local.get('m1_router') is not None:
                raise ValueError('Unsupported resume state.')
            gs,scene=local['gaussians'],local['scene']
            stack=apply_supplemental(gs,scene,side['supplemental'])
            local['viewpoint_stack']=stack
            local['trainCameras']=scene.getTrainCameras().copy()
            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame),ctypes.c_int(0))
            verify_binding(gs)
            actual=state_digest((gs.capture(),supplemental(gs,scene,stack)))
            if actual!=side['initial_state_sha256']:
                raise ValueError(f'Common-state digest mismatch: {actual}')
            record['initial_state_sha256']=actual
            record['resume_state_identical']=True
            record['resumed_iteration']=local['first_iter']
            record['initial_gaussians']=len(gs._xyz)
            applied=True
            sys.settrace(None)
            print(f'[branch] exact common-state digest {actual}',flush=True)
        return trace
    def save(obj,path,*a,**kw):
        original_save(obj,path,*a,**kw)
        if opts.warmup_stop and Path(path).name==f'chkpnt{opts.warmup_stop}.pth':
            frame=inspect.currentframe().f_back
            if frame.f_code.co_name!='training':
                raise ValueError('Checkpoint called outside expected training frame.')
            loc=frame.f_locals
            gs,scene=loc['gaussians'],loc['scene']
            if loc.get('m1_router') is not None:
                raise ValueError('M1 state unsupported.')
            supp=supplemental(gs,scene,loc['viewpoint_stack'])
            digest=state_digest((obj[0],supp))
            state={'iteration':obj[1],'supplemental':supp,'initial_state_sha256':digest,
                'checkpoint_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            target=Path(path).with_suffix('.runtime.pth')
            original_save(state,target)
            record['initial_state_sha256']=digest
            record['common_checkpoint_sha256']=state['checkpoint_sha256']
            record['runtime_sidecar']=str(target)
            record['initial_gaussians']=len(gs._xyz)
            raise WarmupComplete()
    def render(*a,**kw):
        record['render_calls']+=1
        record['depth_render_calls']+=int(kw.get('require_depth',False))
        if 'first_camera' not in record:
            record['first_camera']=a[0].image_name
        if kw.get('require_depth',False) and 'first_depth_camera' not in record:
            record['first_depth_camera']=a[0].image_name
        return original_render(*a,**kw)
    def ncc(*a,**kw):
        result=original_ncc(*a,**kw)
        record['ncc_calls']+=1
        return result
    torch.save=save
    gaussian_renderer.render=render
    warp_patch_ncc.warp_patch_ncc=ncc
    torch.cuda.reset_peak_memory_stats()
    try:
        if side is not None:
            sys.settrace(trace)
        sys.argv=[str(ROOT/'train.py'),*train_args]
        with activate(backend):
            try:
                runpy.run_path(str(ROOT/'train.py'),run_name='__main__')
            except WarmupComplete:
                if not opts.warmup_stop:
                    raise
        if side is not None and not applied:
            raise ValueError('Resume hook was not exercised.')
        if side is not None and (record['depth_render_calls']==0 or record['ncc_calls']==0):
            raise ValueError('Resume did not exercise full depth/NCC path.')
        record['completed']=True
    except BaseException as error:
        record['error']=repr(error)
        raise
    finally:
        sys.settrace(None)
        torch.save=original_save
        GaussianModel.restore=original_restore
        gaussian_renderer.render=original_render
        warp_patch_ncc.warp_patch_ncc=original_ncc
        record['wall_seconds']=time.perf_counter()-started
        record['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated()
        model.mkdir(parents=True,exist_ok=True)
        (model/'solver_training_integration.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
