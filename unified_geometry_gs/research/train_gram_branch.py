"""Use frozen complete-state restore with an isolated, audited world VJP backend."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class DiagnosticComplete(Exception):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--world-variant',choices=('gram_factor64',),required=True)
    parser.add_argument('--diagnostic-steps',type=int,default=0)
    opts,train_args=parser.parse_known_args()
    if '--solver' in train_args:raise ValueError('Outer wrapper fixes original five-pass root solver.')
    from research import train_solver_branch as branch
    from research.gram_factor_preprocess_backend import load_backend,kernel_identity
    model=Path(train_args[train_args.index('-m')+1])
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
        'world_variant':opts.world_variant,'root_subdivisions':5,'sample_batch_size':2,
        'diagnostic_steps_requested':opts.diagnostic_steps,'steps_observed':0,'nonfinite_gradient_groups':0,
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/'research/gram_factor_preprocess_backend.py',ROOT/'research/gram_forward_probe.py',
             ROOT/'research/factor_preprocess_backend.py',ROOT/'research/factor_preprocess_probe.py',
             ROOT/'research/stable_inverse_backend.py',ROOT/'research/world_preprocess_probe.py',
             ROOT/'research/precise_preprocess_backend.py',ROOT/'research/train_solver_branch.py',ROOT/'research/checkpoint_state.py')},
        'limits':['Gram forward uses a cross-product determinant; backward is the audited factor64; contributor accumulation remains float32 and roots remain five-pass.',
                  'Fixed complete starting state does not enforce deterministic CUDA training.',
                  'DiagnosticComplete is an intentional smoke stop; a smoke run is not full quality evidence.']}
    oldload=branch.load_backend;oldstep=torch.optim.Adam.step;start=time.perf_counter()
    def load(v):
        if v!='control':raise ValueError('Unexpected root-solver mode.')
        module=load_backend(opts.world_variant)
        record['kernel_identity']=kernel_identity(module,opts.world_variant)
        return module
    def step(self,*a,**kw):
        frame=inspect.currentframe().f_back
        # torch.optim.Optimizer wraps step with a profiling/hook frame.
        # Locate the verified train.py frame rather than assuming one caller.
        for _ in range(8):
            if frame is None:break
            if frame.f_code.co_name=='training' and Path(frame.f_code.co_filename).resolve()==ROOT/'train.py':break
            frame=frame.f_back
        if frame is None or frame.f_code.co_name!='training' or Path(frame.f_code.co_filename).resolve()!=ROOT/'train.py':
            raise ValueError('Verified optimizer training frame not found.')
        loc=frame.f_locals;loss=loc['loss']
        if not bool(torch.isfinite(loss).all()):raise FloatingPointError('Nonfinite training loss.')
        if record['steps_observed']==0:
            record['first_iteration']=loc['iteration']
            record['first_loss_scalars']={k:float(v.detach().reshape(-1)[0]) for k,v in loc.items()
                if isinstance(v,torch.Tensor) and v.numel()==1 and ('loss' in k.lower() or k.startswith('Ll1'))}
        # Finite checks are smoke diagnostics only: scanning all fields and
        # synchronizing each update would distort full-training timing.
        if opts.diagnostic_steps:
            for group in self.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        record['nonfinite_gradient_groups']+=1
                        raise FloatingPointError('Nonfinite parameter gradient: '+str(group.get('name')))
        value=oldstep(self,*a,**kw);record['steps_observed']+=1
        if opts.diagnostic_steps and record['steps_observed']>=opts.diagnostic_steps:raise DiagnosticComplete()
        return value
    branch.load_backend=load;torch.optim.Adam.step=step
    try:
        sys.argv=[str(ROOT/'research/train_solver_branch.py'),'--solver','control',*train_args]
        try:branch.main()
        except DiagnosticComplete:record['intentional_stop_observed']=True
        inner=json.loads((model/'solver_training_integration.json').read_text())
        if not inner.get('resume_state_identical'):raise ValueError('Common starting state not verified.')
        for key in ('initial_state_sha256','first_camera','backend_sha256','common_checkpoint_sha256',
                    'sidecar_sha256','peak_torch_allocated_bytes','depth_render_calls','ncc_calls','initial_gaussians'):
            record[key]=inner[key]
        if opts.diagnostic_steps:
            if record['steps_observed']!=opts.diagnostic_steps or not record.get('intentional_stop_observed'):
                raise ValueError('Smoke stop boundary mismatch.')
        elif not inner['completed']:raise ValueError('Full training did not complete.')
        record['status']='complete'
    except BaseException as e:record['status']='failed';record['error']=repr(e);raise
    finally:
        branch.load_backend=oldload;torch.optim.Adam.step=oldstep
        record['wrapper_wall_seconds']=time.perf_counter()-start
        model.mkdir(parents=True,exist_ok=True)
        (model/'gram_training_integration.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
