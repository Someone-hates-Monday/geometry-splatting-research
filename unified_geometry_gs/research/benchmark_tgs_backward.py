"""Bounded operator-latency benchmark on one complete published checkpoint.

No optimizer, training, target images or quality evaluation. Fixed dense RGB
upstream VJP, debug=False, balanced orders; instrumented builds are excluded.
Native .item() calls, allocation and host gaps remain inside API timing.
"""
import gc
import hashlib
import importlib.util
import itertools
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from audit_tgs_real_checkpoint import ROOT,load_input
from run_tgs_real_events import camera_tensors
from tgs_mixed_reference import load_module

OUT=ROOT/'experiments/tgs_backward_cost_v1'
VARIANTS=['control','geometry','piecewise']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def telemetry():
    binary='/usr/lib/wsl/lib/nvidia-smi'
    fields='name,driver_version,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,clocks.current.graphics,clocks.current.memory'
    r=subprocess.run([binary,'--query-gpu='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    return dict(returncode=r.returncode,fields=fields,stdout=r.stdout.strip(),stderr=r.stderr.strip())


def modules():
    out={'control':load_module('control'),'geometry':load_module('fixed')}
    receipt=json.loads((ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/build_receipt.json').read_text())
    path=Path(receipt['module_file']);assert sha(path)==receipt['module_sha256']
    spec=importlib.util.spec_from_file_location(path.stem,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    out['piecewise']=mod
    return out


class Workload:
    def __init__(self):
        model,_,_,_,selection=load_input()
        self.points=model['triangles_points'].detach().cuda().contiguous()
        self.sigma=(.01+torch.exp(model['sigma'].detach().cuda())).contiguous()
        self.opacity=torch.sigmoid(model['opacity'].detach().cuda()).contiguous()
        self.sh=torch.cat([model['features_dc'].detach(),model['features_rest'].detach()],1).cuda().contiguous()
        self.n=len(self.points)
        self.counts=torch.full((self.n,),3,dtype=torch.int32,device='cuda')
        self.offsets=torch.arange(self.n,dtype=torch.int32,device='cuda')*3
        self.scaling=torch.zeros(self.n,device='cuda');self.density=torch.zeros_like(self.scaling)
        assert self.scaling.data_ptr()!=self.density.data_ptr()
        self.empty=torch.empty(0,device='cuda');self.bg=torch.zeros(3,device='cuda')
        audit=json.loads((ROOT/'experiments/tgs_real_event_v1/input_audit.json').read_text())
        w,h=audit['actual_resolution'];self.camera_metadata=audit['selected_cameras'][2:4]
        assert all(c['split']=='train' for c in self.camera_metadata)
        self.cameras=[camera_tensors(c,w,h) for c in self.camera_metadata]
        yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
        spatial=1+.25*torch.sin(xx/17)+.2*torch.cos(yy/23)
        self.grad=(torch.tensor([1.,-.3,.5],device='cuda')[:,None,None]*spatial[None]/(w*h)).contiguous()
        self.grad_others=torch.zeros((7,h,w),device='cuda')
        self.checkpoint=ROOT/'sources/tgs_real_checkpoint_v1'/selection['scene']/'point_cloud'/selection['iteration']/'point_cloud_state_dict.pt'
        self.metadata=dict(triangles=self.n,vertices=3*self.n,resolution=[w,h],checkpoint_sha256=sha(self.checkpoint),
            sh_degree=3,mask_semantics='public loader inference, effective mask=1; no training mask history',
            upstream='dense fixed RGB [1,-.3,.5]*(1+.25sin(x/17)+.2cos(y/23))/(W*H), others zero',
            upstream_sha256=hashlib.sha256(self.grad.cpu().numpy().tobytes()).hexdigest(),
            sigma_sha256=hashlib.sha256(self.sigma.cpu().numpy().tobytes()).hexdigest(),
            opacity_sha256=hashlib.sha256(self.opacity.cpu().numpy().tobytes()).hexdigest())

    def forward(self,mod,ci):
        c=self.cameras[ci]
        return mod.rasterize_triangles(self.bg,self.points,self.sigma,self.counts,self.offsets,
            self.empty,self.opacity,self.scaling,self.density,c['view'],c['projection'],self.n,
            c['tx'],c['ty'],c['h'],c['w'],self.sh,3,c['center'],False,False)

    def backward(self,mod,res,ci):
        c=self.cameras[ci]
        return mod.rasterize_triangles_backward(self.bg,self.points,self.sigma,self.counts,self.offsets,
            res[3],self.empty,c['view'],c['projection'],self.n,c['tx'],c['ty'],self.grad,self.grad_others,
            self.sh,3,c['center'],res[4],res[0],res[5],res[6],False)


def measured(work,mod,ci):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    resident=torch.cuda.memory_allocated()
    e0,e1,e2,e3=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    t0=time.perf_counter();e0.record();res=work.forward(mod,ci);e1.record();e1.synchronize();t1=time.perf_counter()
    t2=time.perf_counter();e2.record();g=work.backward(mod,res,ci);e3.record();e3.synchronize();t3=time.perf_counter()
    row=dict(forward_event_ms=e0.elapsed_time(e1),backward_event_ms=e2.elapsed_time(e3),
             forward_wall_ms=(t1-t0)*1000,backward_wall_ms=(t3-t2)*1000,pair_wall_ms=(t3-t0)*1000,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(),resident_bytes=resident,
             incremental_peak_bytes=torch.cuda.max_memory_allocated()-resident,
             peak_reserved_bytes=torch.cuda.max_memory_reserved(),tile_instances=int(res[0]))
    del res,g
    return row


def main():
    assert not OUT.exists(),'Explicitly version another benchmark; preserve this run'
    OUT.mkdir(parents=True)
    mods=modules();work=Workload()
    permutations=list(itertools.permutations(VARIANTS))
    rng=np.random.default_rng(20261020)
    order=[]
    for repeat in range(3):
        blocks=[dict(camera=ci,variants=list(p)) for ci in range(2) for p in permutations]
        rng.shuffle(blocks);order.extend(blocks)
    plan=dict(workload=work.metadata,cameras=work.camera_metadata,variants=VARIANTS,
              modules={k:dict(path=m.__file__,sha256=sha(Path(m.__file__))) for k,m in mods.items()},
              warmup_pairs_per_camera_variant=3,measured_pairs_per_camera_variant=18,balanced_blocks=order,
              debug=False,training=False,instrumentation=False,bootstrap_seed=5326,
              timing_scope='native extension API with output/scratch allocations and .item host syncs; CUDA events include host-induced GPU gaps, not pure kernel time',
              exclusions=['model/data loading','optimizer','loss evaluation','autograd Python wrapper','densification','CPU copies/metrics outside each timed pair'],
              script_sha256=sha(Path(__file__)),telemetry_start=telemetry())
    (OUT/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(stage='frozen_before_timing',triangles=work.n,blocks=len(order))),flush=True)
    guards=[]
    for ci in range(2):
        base=work.forward(mods['control'],ci);g0=work.backward(mods['control'],base,ci);torch.cuda.synchronize()
        assert all(torch.isfinite(g).all() for g in g0)
        baseline_geometry=g0[0].detach().clone();rgb=base[1].detach().clone();aux=base[2].detach().clone();radii=base[3].detach().clone()
        del base,g0
        for variant in ['geometry','piecewise']:
            res=work.forward(mods[variant],ci);g=work.backward(mods[variant],res,ci);torch.cuda.synchronize()
            assert all(torch.isfinite(x).all() for x in g)
            assert torch.equal(res[1],rgb) and torch.equal(res[2],aux) and torch.equal(res[3],radii)
            a=baseline_geometry.double().ravel();b=g[0].double().ravel()
            guards.append(dict(camera=ci,variant=variant,forward_rgb_aux_radii_identical=True,
                geometry_relative_to_new=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b)),
                geometry_cosine=float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)))))
            del res,g,a,b
        del baseline_geometry,rgb,aux,radii
    (OUT/'functional_guards.json').write_text(json.dumps(guards,indent=2)+'\n')
    for ci in range(2):
        for variant in VARIANTS:
            for _ in range(3):measured(work,mods[variant],ci)
    torch.cuda.synchronize();gc.collect()
    rows=[]
    started=time.time()
    for block_id,block in enumerate(order):
        for position,variant in enumerate(block['variants']):
            row=measured(work,mods[variant],block['camera'])
            row.update(block=block_id,camera=block['camera'],variant=variant,position=position)
            rows.append(row)
        (OUT/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        if block_id%6==5:print(json.dumps(dict(stage='timing',completed_blocks=block_id+1,total_blocks=len(order))),flush=True)
    result=dict(protocol_sha256=sha(OUT/'protocol.json'),rows=rows,guards=guards,
                measured_seconds=time.time()-started,telemetry_end=telemetry(),training=False,
                torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name())
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(stage='timing_complete',measured_pairs=len(rows),seconds=result['measured_seconds'])),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():(OUT/'execution_failure.txt').write_text(traceback.format_exc())
        raise
