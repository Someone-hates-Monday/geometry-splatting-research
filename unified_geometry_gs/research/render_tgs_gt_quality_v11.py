"""Exactly eight forward-only renders with frozen public weights, no training."""
import gc,hashlib,importlib.util,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_cross_scene_common_v10 import ROOT,CrossWorkload,modules as prior_modules,state

OUT=ROOT/'experiments/tgs_gt_quality_v11';V10=ROOT/'experiments/tgs_cross_scene_v10';V10_SHA='9d8b5e537c4110881a6ea1696c2c1f41a93ec77cb8ab4e5eba4006c5095ee463'
VARIANTS=['oracle','coarse','piecewise','archived_native']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def binary(receipt,expected):
    q=json.loads(receipt.read_text());p=Path(q['module_file']);assert sha(p)==q['module_sha256']==expected;s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def identical(a,b):return a.shape==b.shape and a.dtype==b.dtype and bool(np.array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8)))

def main():
    target=OUT/'render_results.json';assert not target.exists();frozen=json.loads((OUT/'quality_plan.json').read_text());identity=json.loads((OUT/'identity_audit.json').read_text());assert identity['all_gates'] and identity['quality_measurement_allowed'];assert frozen['identity_audit_sha256']==sha(OUT/'identity_audit.json') and sha(V10/'evidence_manifest.json')==V10_SHA
    for name,h in frozen['scripts'].items():assert sha(ROOT/name)==h
    assert json.loads((OUT/'metric_selfcheck.json').read_text())['all_verified'];old=json.loads((V10/'real_plan.json').read_text());pm=prior_modules();mods={k:pm[k] for k in ['oracle','coarse']};mods['piecewise']=binary(ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/build_receipt.json','f06b1a1ccf5c11f99cebda0b0b5338ca273186ba316aa74fa619c4fa45ccd852');mods['archived_native']=binary(ROOT/'experiments/tgs_subdivision_v1/native_build/receipt_9d974f09b721.json','9d974f09b7217a026aaad3ebc48af16c6a9571fcd17a885711b0517b9d409da1');assert list(mods)==VARIANTS
    plan=dict(quality_plan_sha256=sha(OUT/'quality_plan.json'),modules={k:dict(path=m.__file__,sha256=sha(Path(m.__file__))) for k,m in mods.items()},model_inputs=old['model_inputs'],variants=VARIANTS,training=False,timing=False);p=OUT/'render_plan.json';assert not p.exists();p.write_text(json.dumps(plan,indent=2)+'\n');rows=[]
    for entry in identity['rows']:
        scene=entry['scene'];w=CrossWorkload(scene);assert w.camera_metadata[0]['img_name']==entry['camera'] and w.input_hashes()==plan['model_inputs'][scene];f=dict(n=w.n,w=w.cameras[0]['w'],h=w.cameras[0]['h'])
        for v,m in mods.items():
            folder=OUT/scene/v;assert not folder.exists();folder.mkdir();r=w.forward(m,0);torch.cuda.synchronize();arrays=dict(rgb=r[1].cpu().numpy(),aux=r[2].cpu().numpy(),radii=r[3].cpu().numpy(),maxblend=r[9].cpu().numpy());checks=None
            if v in ['oracle','coarse']:
                flags,ids,pref=state(r,f,v);assert not flags.any() and not pref.any();arrays.update(flags=flags,contributors=ids);checks={key:identical(value,np.load(V10/scene/'camera_0'/v/(key+'.npy'),mmap_mode='r')) for key,value in arrays.items()};assert all(checks.values())
            for key,value in arrays.items():np.save(folder/(key+'.npy'),value,allow_pickle=False)
            rgb=arrays['rgb'];assert rgb.dtype==np.float32;finite=bool(np.isfinite(rgb).all());row=dict(scene=scene,camera=entry['camera'],variant=v,shape=list(rgb.shape),all_RGB_finite=finite,raw_RGB_min=float(rgb.min()) if finite else None,raw_RGB_max=float(rgb.max()) if finite else None,raw_out_of_unit_fraction=float(((rgb<0)|(rgb>1)).mean()) if finite else None,tile_instances=int(r[0]),v10_forward_literal_bits=checks,files={key:sha(folder/(key+'.npy')) for key in arrays});rows.append(row);(OUT/'render_progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:val for k,val in row.items() if k not in ['files','v10_forward_literal_bits']}),flush=True);assert finite,'Retain nonfinite output; no selective quality scoring';del r,arrays,rgb
        assert w.input_hashes()==plan['model_inputs'][scene];del w;gc.collect();torch.cuda.empty_cache()
    target.write_text(json.dumps(dict(rows=rows,model_inputs_unchanged=True,plan_sha256=sha(p),script_sha256=sha(Path(__file__)),training=False,timing=False),indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except BaseException:(OUT/'render_failure.txt').write_text(traceback.format_exc());raise
