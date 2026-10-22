"""Balanced native API timings; correctness audit must already have passed."""
import gc,hashlib,importlib.util,itertools,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_phase_common import ROOT,OUT,sha,workload
from benchmark_tgs_backward import measured,telemetry
TIMED=['piecewise','guard','oracle']

def load(path,digest):
    assert sha(path)==digest;s=importlib.util.spec_from_file_location(path.stem,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def tensor_hashes(work):
    names=['points','sigma','opacity','sh','grad','grad_others'];result={k:hashlib.sha256(getattr(work,k).cpu().numpy().tobytes()).hexdigest() for k in names}
    return result
def main():
    folder=OUT/'cost';assert not folder.exists();audit=json.loads((OUT/'precision_audit.json').read_text());assert audit['all_gates'] and audit['performance_measurement_allowed'];folder.mkdir()
    mods={};receipts={}
    for v in ['guard','oracle']:
        r=json.loads((OUT/v/'build_receipt.json').read_text());assert not r['instrumented'];p=Path(r['module_file']);mods[v]=load(p,r['module_sha256']);receipts[v]=r
    r=json.loads((ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/build_receipt.json').read_text());p=Path(r['module_file']);assert r['module_sha256']=='f06b1a1ccf5c11f99cebda0b0b5338ca273186ba316aa74fa619c4fa45ccd852';mods['piecewise']=load(p,r['module_sha256'])
    work=workload();rng=np.random.default_rng(202610206);perms=list(itertools.permutations(TIMED));order=[]
    for repeat in range(3):
        blocks=[dict(camera=ci,variants=list(p),repeat=repeat) for ci in range(4) for p in perms];rng.shuffle(blocks);order.extend(blocks)
    hashes=tensor_hashes(work)
    plan=dict(protocol_sha256=sha(ROOT/'TGS_PHASE_COST_PROTOCOL_V5.md'),precision_audit_sha256=sha(OUT/'precision_audit.json'),workload=work.metadata,cameras=work.camera_metadata,
        modules={v:dict(module_file=m.__file__,module_sha256=sha(Path(m.__file__)),instrumented=False) for v,m in mods.items()},input_tensor_hashes=hashes,balanced_blocks=order,warmup_pairs_per_camera_variant=3,measured_pairs_per_camera_variant=18,script_sha256=sha(Path(__file__)),debug=False,training=False,telemetry_start=telemetry(),bootstrap_seed=202610207,
        primary='paired guard/oracle forward+backward wall ratio',scope='Native API including allocations/.item synchronizations and host-induced CUDA-event gaps, not pure kernel or training latency',historical_baseline='piecewise has a different numerical forward, retained only as cost context; no output equivalence or quality claim')
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(stage='plan_frozen_before_warmup',blocks=len(order),measured_pairs=216)),flush=True)
    guards=[]
    for ci in range(4):
        for v,m in mods.items():
            r=work.forward(m,ci);g=work.backward(m,r,ci);torch.cuda.synchronize();finite=all(torch.isfinite(x).all() for x in g);assert finite
            if v!='piecewise':
                with np.load(OUT/f'real_{ci}_{v}.npz') as z:
                    assert np.array_equal(r[1].cpu().numpy(),z['rgb']) and np.array_equal(r[2].cpu().numpy(),z['aux']) and np.array_equal(r[3].cpu().numpy(),z['radii']) and np.array_equal(r[9].cpu().numpy(),z['maxblend'])
            else:
                old=ROOT/f'experiments/tgs_real_event_v1/baseline_{[2,3,0,1][ci]}.npz'
                if old.exists():
                    with np.load(old) as z:assert np.array_equal(r[1].cpu().numpy(),z['fixed_rgb'])
            guards.append(dict(camera=ci,variant=v,all_gradients_finite=True,forward_matches_validation_or_history=True));del r,g
    (folder/'functional_guards.json').write_text(json.dumps(guards,indent=2)+'\n')
    for ci in range(4):
        for v in TIMED:
            for _ in range(3):measured(work,mods[v],ci)
    gc.collect();torch.cuda.synchronize();rows=[];device_samples=[];started=time.time()
    for block_id,block in enumerate(order):
        for position,v in enumerate(block['variants']):
            row=measured(work,mods[v],block['camera']);row.update(block=block_id,camera=block['camera'],variant=v,position=position,repeat=block['repeat']);rows.append(row)
        (folder/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        if block_id%12==11:
            device_samples.append(dict(after_block=block_id,telemetry=telemetry()));print(json.dumps(dict(completed_blocks=block_id+1,total=len(order))),flush=True)
    final_hashes=tensor_hashes(work);assert final_hashes==hashes
    result=dict(plan_sha256=sha(folder/'plan.json'),rows=rows,guards=guards,input_hashes_after=final_hashes,device_samples=device_samples,telemetry_end=telemetry(),measured_seconds=time.time()-started,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),training=False)
    (folder/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(stage='complete',measured_pairs=len(rows),seconds=result['measured_seconds'],input_tensors_unchanged=True)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        folder=OUT/'cost'
        if folder.exists():(folder/'execution_failure.txt').write_text(traceback.format_exc())
        raise
