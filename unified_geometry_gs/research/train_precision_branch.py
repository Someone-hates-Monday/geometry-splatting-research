"""Isolated mixed-precision world-backward branch, optionally timing real optimizer steps.

Reuses the verified complete-state restoration without editing its frozen source.
The original wrapper's solver=control denotes five root subdivisions; this outer
record explicitly states which local world-backward precision and query grouping were loaded instead.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class TimingComplete(Exception): pass


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--world-backend',choices=('control','mixed1'),required=True)
    parser.add_argument('--timing-steps',type=int,default=0)
    parser.add_argument('--timing-warmup',type=int,default=50)
    opts,train_args=parser.parse_known_args()
    if opts.timing_steps and opts.timing_steps<=opts.timing_warmup:
        raise ValueError('Timing needs steps after warmup.')
    if '--solver' in train_args: raise ValueError('Outer wrapper selects the five-pass root solver.')
    from scene import GaussianModel
    from research import train_solver_branch as branch
    from research.mixed_preprocess_backend import load_backend
    model=Path(train_args[train_args.index('-m')+1])
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'world_backend':opts.world_backend,
        'root_subdivisions':5,'timing_steps':opts.timing_steps,'timing_warmup':opts.timing_warmup,
        'status':'running','source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),ROOT/'research/mixed_preprocess_backend.py',ROOT/'research/precise_preprocess_backend.py',ROOT/'research/train_solver_branch.py')},
        'limitations':['Short real-training timing includes RGB/NCC/geometry, backward, densification/filter and Adam when scheduled.',
            'Per-step interval begins at update_learning_rate and ends after Adam.step; excludes later zero_grad/checkpoint handling and GUI polling.',
            'Synchronized measured block wall includes the inter-step Python work; GPU event sums may include device idle time.',
            'Same initial state does not force later floating-point trajectories to be identical.',
            'TimingComplete in the inner integration is intentional, not failed quality training.']}
    original_load=branch.load_backend
    original_lr=GaussianModel.update_learning_rate
    original_step=torch.optim.Adam.step
    events=[];iterations=[];gaussian_counts=[];seen=0;current=None;block_start=None
    def load(variant):
        if variant!='control':raise ValueError('Only original five-pass roots supported.')
        return load_backend(opts.world_backend)
    def update(self,iteration):
        nonlocal current,block_start
        if opts.timing_steps and seen>=opts.timing_warmup:
            if block_start is None:
                torch.cuda.synchronize();block_start=time.perf_counter()
            current=torch.cuda.Event(enable_timing=True);current.record()
            iterations.append(iteration);gaussian_counts.append(len(self._xyz))
        return original_lr(self,iteration)
    def step(self,*a,**kw):
        nonlocal seen
        result=original_step(self,*a,**kw)
        seen+=1
        if opts.timing_steps:
            if current is not None:
                end=torch.cuda.Event(enable_timing=True);end.record();events.append((current,end))
            if seen>=opts.timing_steps:
                torch.cuda.synchronize()
                state['measured_block_wall_seconds']=time.perf_counter()-block_start
                state['iteration_gpu_ms']=[a.elapsed_time(b) for a,b in events]
                state['measured_iterations']=iterations
                state['gaussian_counts_at_iteration_start']=gaussian_counts
                state['optimizer_steps_observed']=seen
                raise TimingComplete()
        return result
    branch.load_backend=load
    GaussianModel.update_learning_rate=update
    torch.optim.Adam.step=step
    started=time.perf_counter()
    try:
        sys.argv=[str(ROOT/'research/train_solver_branch.py'),'--solver','control',*train_args]
        try:branch.main()
        except TimingComplete:
            state['intentional_stop_observed']=True
        inner=json.loads((model/'solver_training_integration.json').read_text())
        if not inner.get('resume_state_identical'):raise ValueError('Common state was not verified.')
        state['initial_state_sha256']=inner['initial_state_sha256']
        state['first_camera']=inner['first_camera']
        state['backend_sha256']=inner['backend_sha256']
        state['common_checkpoint_sha256']=inner['common_checkpoint_sha256']
        state['sidecar_sha256']=inner['sidecar_sha256']
        state['peak_torch_allocated_bytes']=inner['peak_torch_allocated_bytes']
        state['depth_render_calls']=inner['depth_render_calls'];state['ncc_calls']=inner['ncc_calls']
        if opts.timing_steps:
            if seen!=opts.timing_steps or len(events)!=opts.timing_steps-opts.timing_warmup or not state.get('intentional_stop_observed'):
                raise ValueError('Timing boundary mismatch.')
        elif not inner['completed']:raise ValueError('Full branch did not complete.')
        state['status']='complete'
    except BaseException as e:
        state['status']='failed';state['error']=repr(e);raise
    finally:
        branch.load_backend=original_load
        GaussianModel.update_learning_rate=original_lr
        torch.optim.Adam.step=original_step
        state['wrapper_wall_seconds']=time.perf_counter()-started
        model.mkdir(parents=True,exist_ok=True)
        (model/'precision_training_integration.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
