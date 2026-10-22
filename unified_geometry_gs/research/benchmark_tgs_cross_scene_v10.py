"""Only after independent four-camera precision; balanced per-scene fixed-input cost."""
import gc,importlib.util,itertools,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_cross_scene_common_v10 import ROOT,OUT,sha,modules,CrossWorkload,state
from benchmark_tgs_backward import measured,telemetry

def main():
    cost=OUT/'cost';assert not cost.exists();precision=json.loads((OUT/'precision_audit.json').read_text());assert precision['all_gates'] and precision['performance_measurement_allowed'];cost.mkdir();inp=json.loads((OUT/'input_audit.json').read_text());oldplan=json.loads((OUT/'real_plan.json').read_text());mods0=modules();mods={v:mods0[v] for v in ['oracle','coarse']};receipt=json.loads((ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/build_receipt.json').read_text());path=Path(receipt['module_file']);assert sha(path)==receipt['module_sha256']=='f06b1a1ccf5c11f99cebda0b0b5338ca273186ba316aa74fa619c4fa45ccd852';spec=importlib.util.spec_from_file_location(path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods['piecewise']=m;variants=list(mods);orders=list(itertools.permutations(variants));rng=np.random.default_rng(202610213);blocks=[];cameras=[]
    for si,s in enumerate(inp['scenes']):
        cameras.extend(dict(c,scene=s['scene'],local_camera=ci) for ci,c in enumerate(s['selected_cameras']))
        for repeat in range(3):
            batch=[dict(scene=s['scene'],camera=2*si+ci,local_camera=ci,repeat=repeat,order_index=oi,variants=list(order)) for ci in range(2) for oi,order in enumerate(orders)];rng.shuffle(batch);blocks.extend(batch)
    plan=dict(protocol_sha256=sha(ROOT/'TGS_CROSS_SCENE_COST_PROTOCOL_V10.md'),script_sha256=sha(Path(__file__)),precision_audit_sha256=sha(OUT/'precision_audit.json'),variants=variants,eligible=['coarse'],scenes=[s['scene'] for s in inp['scenes']],cameras=cameras,blocks=blocks,warmups=3,shuffle_seed=202610213,bootstrap_seed=202610214,model_before=oldplan['model_inputs'],modules={v:dict(path=m.__file__,sha256=sha(Path(m.__file__))) for v,m in mods.items()},telemetry_start=telemetry(),training=False,timing_scope='Native API allocation and host synchronization included; separate scene residency, single session.')
    (cost/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[];tele=[];guards=[];after={}
    for scene in plan['scenes']:
        work=CrossWorkload(scene);assert work.input_hashes()==plan['model_before'][scene];work.set_loss('rgb')
        for ci,cam in enumerate(work.camera_metadata):
            for v in ['oracle','coarse']:
                rr=work.forward(mods[v],ci);torch.cuda.synchronize();f=dict(n=work.n,w=work.cameras[ci]['w'],h=work.cameras[ci]['h']);flags,ids,_=state(rr,f,v);folder=OUT/scene/f'camera_{ci}'/v;equal={k:bool(np.array_equal(arr,np.load(folder/(k+'.npy'),mmap_mode='r'))) for k,arr in [('rgb',rr[1].cpu().numpy()),('aux',rr[2].cpu().numpy()),('radii',rr[3].cpu().numpy()),('maxblend',rr[9].cpu().numpy()),('flags',flags),('contributors',ids)]};assert all(equal.values());guards.append(dict(scene=scene,camera=ci,variant=v,equal=equal));del rr,flags,ids
        (cost/'preflight_guards.json').write_text(json.dumps(guards,indent=2)+'\n')
        for ci in range(2):
            for v in variants:
                for _ in range(3):measured(work,mods[v],ci)
        for bi,block in enumerate(blocks):
            if block['scene']!=scene:continue
            before=telemetry()
            for position,v in enumerate(block['variants']):
                row=measured(work,mods[v],block['local_camera']);row.update(block=bi,scene=scene,camera=block['camera'],variant=v,position=position);rows.append(row)
            tele.append(dict(block=bi,before=before,after=telemetry()));(cost/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');(cost/'telemetry.json').write_text(json.dumps(tele,indent=2)+'\n')
            if bi%12==11:print(json.dumps(dict(completed_blocks=bi+1,total=72,scene=scene)),flush=True)
        after[scene]=work.input_hashes();assert after[scene]==plan['model_before'][scene];del work;gc.collect();torch.cuda.empty_cache()
    (cost/'results.json').write_text(json.dumps(dict(plan_sha256=sha(cost/'plan.json'),rows=rows,input_hashes_after=after,inputs_unchanged=True,telemetry_end=telemetry(),training=False),indent=2)+'\n');print(json.dumps(dict(calls=len(rows),inputs_unchanged=True)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        cost=OUT/'cost';cost.mkdir(exist_ok=True);(cost/'failure.txt').write_text(traceback.format_exc());raise
