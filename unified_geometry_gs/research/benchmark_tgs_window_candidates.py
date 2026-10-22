"""Balanced operator calls only after individual precision decisions; keep all outcomes."""
import hashlib,importlib.util,itertools,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_window_candidates_common import ROOT,OUT,sha,modules,workload,state
from benchmark_tgs_backward import measured,telemetry
def thash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def main():
    cost=OUT/'cost';assert not cost.exists();precision=json.loads((OUT/'precision_audit.json').read_text());eligible=[v for v in ['coarse','standard'] if precision['performance_measurement_allowed'][v]];assert eligible,'Neither candidate passes precision; no timing allowed';cost.mkdir()
    mods0=modules();mods={v:mods0[v] for v in ['oracle']+eligible};rp=ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/build_receipt.json';r=json.loads(rp.read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256']=='f06b1a1ccf5c11f99cebda0b0b5338ca273186ba316aa74fa619c4fa45ccd852';s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);mods['piecewise']=m
    work=workload();variants=list(mods);rng=np.random.default_rng(202610211);orders=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]] if len(variants)==4 else list(itertools.permutations(range(3)));blocks=[]
    for repeat in range(3):
        batch=[dict(camera=ci,repeat=repeat,order_index=oi,variants=[variants[i] for i in order]) for ci in range(len(work.cameras)) for oi,order in enumerate(orders)];rng.shuffle(batch);blocks.extend(batch)
    fields=['points','sigma','opacity','sh','counts','offsets','grad','grad_others'];before={k:thash(getattr(work,k)) for k in fields};plan=dict(protocol_sha256=sha(ROOT/'TGS_WINDOW_COST_PROTOCOL_V9.md'),script_sha256=sha(Path(__file__)),precision_audit_sha256=sha(OUT/'precision_audit.json'),variants=variants,eligible=eligible,workload=work.metadata,cameras=work.camera_metadata,blocks=blocks,shuffle_seed=202610211,bootstrap_seed=202610212,warmups=3,model_before=before,modules={v:dict(path=q.__file__,sha256=sha(Path(q.__file__))) for v,q in mods.items()},historical_numerically_different='piecewise',telemetry_start=telemetry(),training=False,timing_scope='native API forward+back with allocation/.item host synchronization; CUDA events include launch gaps')
    (cost/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');guards=[]
    for ci in range(len(work.cameras)):
        for v in ['oracle']+eligible:
            f=dict(n=work.n,w=work.cameras[ci]['w'],h=work.cameras[ci]['h']);rr=work.forward(mods[v],ci);torch.cuda.synchronize();flags,ids,pref=state(rr,f,v)
            with np.load(OUT/f'real_{ci}_{v}.npz') as z:
                equal={key:bool(np.array_equal(arr,z[key])) for key,arr in [('rgb',rr[1].cpu().numpy()),('aux',rr[2].cpu().numpy()),('radii',rr[3].cpu().numpy()),('maxblend',rr[9].cpu().numpy()),('contributors',ids),('flags',flags)]};assert all(equal.values())
            guards.append(dict(camera=ci,variant=v,equal=equal));del rr
    (cost/'preflight_guards.json').write_text(json.dumps(guards,indent=2)+'\n')
    for ci in range(len(work.cameras)):
        for v in variants:
            for _ in range(3):measured(work,mods[v],ci)
    rows=[];tele=[]
    print(json.dumps(dict(stage='timing_start',blocks=len(blocks),calls=len(blocks)*len(variants))),flush=True)
    for bi,block in enumerate(blocks):
        bt=telemetry()
        for position,v in enumerate(block['variants']):
            row=measured(work,mods[v],block['camera']);row.update(block=bi,camera=block['camera'],variant=v,position=position);rows.append(row)
        tele.append(dict(block=bi,before=bt,after=telemetry()));(cost/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');(cost/'telemetry.json').write_text(json.dumps(tele,indent=2)+'\n')
        if bi%12==11:print(json.dumps(dict(completed_blocks=bi+1,total=len(blocks))),flush=True)
    after={k:thash(getattr(work,k)) for k in fields};assert after==before
    (cost/'results.json').write_text(json.dumps(dict(plan_sha256=sha(cost/'plan.json'),rows=rows,input_hashes_after=after,inputs_unchanged=True,telemetry_end=telemetry(),training=False),indent=2)+'\n');print(json.dumps(dict(calls=len(rows),inputs_unchanged=True)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        d=OUT/'cost';d.mkdir(exist_ok=True);(d/'failure.txt').write_text(traceback.format_exc());raise
