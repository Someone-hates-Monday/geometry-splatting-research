"""Twenty main and sixteen negative-control forwards on frozen old/fresh views."""
import gc,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_sh_isolation_common_v12 import ROOT,OUT,VARIANTS,sha,modules,workload,forward,capture,exact

def tensor_hash(x):return __import__('hashlib').sha256(x.cpu().numpy().tobytes()).hexdigest()
def write(folder,arrays):
    assert not folder.exists();folder.mkdir(parents=True)
    for key,value in arrays.items():np.save(folder/(key+'.npy'),value,allow_pickle=False)
def compare(a,b,keys):return {key:exact(np.load(a/(key+'.npy'),mmap_mode='r'),np.load(b/(key+'.npy'),mmap_mode='r')) for key in keys}

def main():
    target=OUT/'render_results.json';assert not target.exists();frozen=json.loads((OUT/'execution_plan.json').read_text());assert json.loads((OUT/'identity_audit.json').read_text())['all_gates']
    for name,h in frozen['scripts'].items():assert sha(ROOT/name)==h
    mods=modules();inputs={};color_hashes={}
    for scene in ['flowers','room']:
        w,rs=workload(scene);inputs[scene]=w.input_hashes();col=torch.tensor([.25,.5,.75],device='cuda').expand(w.n,3).contiguous();color_hashes[scene]=tensor_hash(col);del w,col;gc.collect();torch.cuda.empty_cache()
    p=OUT/'render_plan.json';assert not p.exists();p.write_text(json.dumps(dict(execution_plan_sha256=sha(OUT/'execution_plan.json'),model_inputs=inputs,precomputed_colors=color_hashes,modules={v:dict(path=m.__file__,sha256=sha(Path(m.__file__))) for v,m in mods.items()},training=False,timing=False),indent=2)+'\n');rows=[];comparisons=[]
    for scene in ['flowers','room']:
        w,selected=workload(scene);assert w.input_hashes()==inputs[scene];colors=torch.tensor([.25,.5,.75],device='cuda').expand(w.n,3).contiguous();assert tensor_hash(colors)==color_hashes[scene]
        for ci,entry in enumerate(selected):
            cohort=entry['cohort'];base=OUT/scene/cohort;width,height=w.cameras[ci]['w'],w.cameras[ci]['h']
            for mode in ['main','degree0','precomputed']:
                active=VARIANTS if mode=='main' else ['archived_native','sh_only']
                for v in active:
                    result=forward(w,mods[v],ci,degree=0 if mode=='degree0' else 3,colors=colors if mode=='precomputed' else None);torch.cuda.synchronize();arrays=capture(result,width,height);folder=base/mode/v;write(folder,arrays);finite=bool(np.isfinite(arrays['rgb']).all());oldchecks=None
                    if cohort=='old_posthoc' and mode=='main' and v in ['archived_native','piecewise','coarse']:
                        old=ROOT/'experiments/tgs_gt_quality_v11'/scene/v;oldchecks={key:exact(value,np.load(old/(key+'.npy'),mmap_mode='r')) for key,value in arrays.items() if (old/(key+'.npy')).exists()};assert all(oldchecks.values())
                    row=dict(scene=scene,cohort=cohort,camera=entry['camera']['img_name'],mode=mode,variant=v,tile_instances=int(result[0]),all_RGB_finite=finite,v11_old_fields_literal=oldchecks,files={key:sha(folder/(key+'.npy')) for key in arrays});rows.append(row);(OUT/'render_progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(scene=scene,cohort=cohort,mode=mode,variant=v,saved=True)),flush=True);assert finite;del result,arrays
            original=base/'main/archived_native';fixed=base/'main/sh_only';primary=compare(original,fixed,['aux','radii','maxblend','contributors']);tiles={r['variant']:r['tile_instances'] for r in rows if r['scene']==scene and r['cohort']==cohort and r['mode']=='main'};primary['tile_instances']=tiles['archived_native']==tiles['sh_only'];controls={mode:compare(base/mode/'archived_native',base/mode/'sh_only',['rgb','aux','radii','maxblend','contributors']) for mode in ['degree0','precomputed']};a=np.load(original/'rgb.npy');b=np.load(fixed/'rgb.npy');changed=not exact(a,b);q=dict(scene=scene,cohort=cohort,camera=entry['camera']['img_name'],source_only_noncolor=primary,negative_controls=controls,SH3_RGB_changed=changed,SH3_RGB_max_abs=float(np.abs(a.astype(float)-b.astype(float)).max()),isolation_gate=bool(all(primary.values()) and all(all(t.values()) for t in controls.values())));comparisons.append(q);print(json.dumps(q),flush=True)
        assert w.input_hashes()==inputs[scene] and tensor_hash(colors)==color_hashes[scene];del w,colors;gc.collect();torch.cuda.empty_cache()
    target.write_text(json.dumps(dict(rows=rows,comparisons=comparisons,all_isolation_gates=all(r['isolation_gate'] for r in comparisons),model_inputs_unchanged=True,render_plan_sha256=sha(p),script_sha256=sha(Path(__file__)),training=False,timing=False),indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except BaseException:(OUT/'render_failure.txt').write_text(traceback.format_exc());raise
