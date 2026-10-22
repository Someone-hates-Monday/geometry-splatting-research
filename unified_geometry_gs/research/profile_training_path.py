"""Profile 10 active real training iterations after a verified common-state resume.

This intentionally stops a fresh diagnostic branch after 15 optimizer steps.
It is neither a new quality run nor an end-to-end speed measurement.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch


class ProfileComplete(Exception):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    pair=args.pair.resolve()
    base=json.loads((pair/'manifest.json').read_text())
    if base['status']!='complete' or not base['branch_state_verified']:
        raise ValueError('Verified complete base experiment required.')
    for name,sha in base['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'Profiling source differs from full pilot: {name}')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    command=next(s['argv'] for s in base['steps'] if s['stage']=='train_control').copy()
    command[command.index('-m')+1]=str(out/'diagnostic_model')
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
        'mode':'intentional short profiling branch','argv':command,
        'wait_steps':3,'warmup_steps':2,'active_steps':10,'optimizer_steps_to_stop':15,
        'base_initial_state_sha256':base['initial_state_sha256'],
        'source_sha256':{'research/profile_training_path.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **base['source_sha256']},
        'limitations':['Profiles an early regularized suffix at iterations 7000 through 7014, not all training phases.',
            'Profiling and record_function instrumentation add overhead; device activity sums are not the critical path or full wall clock.',
            'Original train_solver_branch intentionally records completed=False/ProfileComplete; this is the requested profiling stop, not a failed quality experiment.',
            'Kernel categories are descriptive name-based groups, not causal speedup estimates.']}
    def write():
        (out/'manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    write()
    import gaussian_renderer
    from utils.loss_utils import PatchMatch
    from research.train_solver_branch import main as train_main
    original_step=torch.optim.Adam.step
    original_backward=torch.Tensor.backward
    original_render=gaussian_renderer.render
    original_patchmatch=PatchMatch.__call__
    completed_steps=0
    profiler=None
    def render(*a,**kw):
        with torch.profiler.record_function('research.render'):
            return original_render(*a,**kw)
    def patchmatch(self,*a,**kw):
        with torch.profiler.record_function('research.PatchMatch_geo_NCC'):
            return original_patchmatch(self,*a,**kw)
    def backward(self,*a,**kw):
        with torch.profiler.record_function('research.loss_backward'):
            return original_backward(self,*a,**kw)
    def step(self,*a,**kw):
        nonlocal completed_steps
        with torch.profiler.record_function('research.optimizer_step'):
            result=original_step(self,*a,**kw)
        completed_steps+=1
        profiler.step()
        if completed_steps>=15:raise ProfileComplete()
        return result
    torch.optim.Adam.step=step
    torch.Tensor.backward=backward
    gaussian_renderer.render=render
    PatchMatch.__call__=patchmatch
    started=time.perf_counter()
    try:
        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=3,warmup=2,active=10,repeat=1),
            on_trace_ready=lambda p:p.export_chrome_trace(str(out/'trace.json')),
            record_shapes=False,profile_memory=False,with_stack=False)
        sys.argv=command[1:]
        with profiler:
            try:train_main()
            except ProfileComplete:state['intentional_stop_observed']=True
        if completed_steps!=15 or not state.get('intentional_stop_observed'):
            raise ValueError('Profiling did not stop at its intended boundary.')
        integration=json.loads((out/'diagnostic_model/solver_training_integration.json').read_text())
        if integration['initial_state_sha256']!=base['initial_state_sha256'] or integration['first_camera']!=base['first_camera']:
            raise ValueError('Profiling branch common state mismatch.')
        state['resume_state_verified']=True
        cuda=[]
        for event in profiler.events():
            if str(event.device_type).endswith('CUDA'):
                duration=float(event.device_time_total)
                cuda.append({'name':event.name,'duration_us':duration})
        if not cuda or not any('forward_mode_differentiation' in r['name'] for r in cuda):
            raise ValueError('CUDA NCC events missing; do not infer GPU costs from CPU timings.')
        groups={};kernels={}
        for row in cuda:
            name=row['name'];duration=row['duration_us']
            if 'forward_mode_differentiation' in name:category='NCC patch forward-mode Jacobians'
            elif 'renderCUDA' in name:category='Gaussian render forward/backward'
            elif 'preprocessCUDA' in name:category='Gaussian preprocess forward/backward'
            elif 'ssim' in name.lower():category='SSIM image loss'
            elif 'adam' in name.lower():category='Adam kernels'
            else:category='Other CUDA activities'
            groups[category]=groups.get(category,0.)+duration
            item=kernels.setdefault(name,{'calls':0,'duration_us':0.})
            item['calls']+=1;item['duration_us']+=duration
        total=sum(r['duration_us'] for r in cuda)
        ranges=[]
        for event in profiler.key_averages():
            if event.key.startswith('research.'):
                ranges.append({'name':event.key,'calls':event.count,'self_cpu_time_us':event.self_cpu_time_total,
                    'inclusive_cpu_time_us':event.cpu_time_total,'inclusive_device_time_us':event.device_time_total})
        report={'cuda_activity_count':len(cuda),'summed_device_activity_us':total,
            'groups':[{'name':k,'duration_us':v,'fraction_of_summed_device_activity':v/total} for k,v in sorted(groups.items(),key=lambda kv:-kv[1])],
            'top_kernels':[{'name':name,**values} for name,values in sorted(kernels.items(),key=lambda kv:-kv[1]['duration_us'])[:30]],
            'instrumented_ranges':ranges,'limitations':state['limitations']}
        (out/'profile_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (out/'profile_table.txt').write_text(profiler.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
        state['status']='complete';state['optimizer_steps_observed']=completed_steps
        print(json.dumps({'cuda_activities':len(cuda),'groups':report['groups']},indent=2))
    except BaseException as error:
        state['status']='failed';state['error']=repr(error);raise
    finally:
        torch.optim.Adam.step=original_step
        torch.Tensor.backward=original_backward
        gaussian_renderer.render=original_render
        PatchMatch.__call__=original_patchmatch
        state['wall_seconds']=time.perf_counter()-started;write()


if __name__=='__main__':main()
