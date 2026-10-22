import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_phase_cost_v5'
VARIANTS=['guard_diag','oracle_diag','guard','oracle']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def modules():
    mods={}
    for v in VARIANTS:
        r=json.loads((OUT/v/'build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256']
        s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);mods[v]=m
    return mods
def workload():
    from benchmark_tgs_backward import Workload
    from run_tgs_real_events import camera_tensors
    work=Workload();p=ROOT/'experiments/tgs_real_event_v1/input_audit.json';audit=json.loads(p.read_text());w,h=audit['actual_resolution']
    work.camera_metadata=[audit['selected_cameras'][i] for i in [2,3,0,1]]
    assert [c['split'] for c in work.camera_metadata]==['train','train','test','test']
    work.cameras=[camera_tensors(c,w,h) for c in work.camera_metadata]
    work.metadata.update(camera_source_sha256=sha(p),camera_order=[c['img_name'] for c in work.camera_metadata],heldout_relative_to_v3_v4=['_DSC9269','_DSC9221'])
    return work
def freeze_inputs():
    p=OUT/'input_plan.json';assert not p.exists();old=ROOT/'experiments/tgs_compositing_guard_v4/input_plan_extended.json'
    plan=json.loads(old.read_text());plan.update(parent_plan_sha256=sha(old),protocol_sha256=sha(ROOT/'TGS_PHASE_COST_PROTOCOL_V5.md'),phase_common_sha256=sha(Path(__file__)))
    p.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(cases=len(plan['cases']),plan_sha256=sha(p))),flush=True)
if __name__=='__main__':freeze_inputs()
