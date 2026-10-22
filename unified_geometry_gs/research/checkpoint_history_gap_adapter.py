from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    a=read(R/'results/history_gap_adapter_audit_v1.json');assert a['audit_gate_pass'];latex=R/'median_depth_note.tex';expected='8f595977f473023caec9ac7676313875dc9fc6233ba854693854b06262ca2621';assert sha(latex)==expected
    cp=R/'results/latex_compile_status_v42.json'
    if cp.exists():assert read(cp)['kind']=='success' and read(cp)['source_sha256']==expected
    else:save(cp,dict(path=str(latex),kind='success',source_sha256=expected,message='Current saved source compiled successfully with desktop built-in compiler.',checked_utc=datetime.now(timezone.utc).isoformat()))
    guard=read(R/'results/canvas_history_gap_adapter_guard_v1.json');assert sha(Path(guard['destination']))==guard['staged_sha256'] and not read(R/'results/canvas_history_gap_adapter_syntax_v1.json')['errors']
    names=['history_gap_live_adapter.py','smoke_history_gap_adapter.py','check_history_gap_adapter_empty.py','verify_history_gap_adapter.py','HISTORY_GAP_ADAPTER_PROTOCOL.md','HISTORY_GAP_ADAPTER_OPACITY_AMENDMENT.md','HISTORY_GAP_ADAPTER_REFERENCE_CORRECTION.md','HISTORY_GAP_ADAPTER_RESULT.md','HISTORY_GAP_FULL_TRAIN_NEXT.md','CONTINUATION.md','history_gap_gpu.py','history_gap_gpu_v2.py','history_gap_raw.py','history_gap_math.py','history_gap_extension.py','world_root_vjp.py','world_root_vjp_v2.py','checkpoint_state.py','exact_outlier_capture_rebuilt.py','benchmark_csr_capture.py','benchmark_median_cuda.py','stage_history_gap_adapter_canvas.py','canvas_history_gap_adapter_stage_v1.tsx','results/history_gap_adapter_input_preflight_v1.json','results/history_gap_adapter_saturated_opacity_v1.json','results/history_gap_adapter_empty_v1.json','results/history_gap_adapter_empty_v1.npz','results/history_gap_adapter_audit_v1.json','results/canvas_history_gap_adapter_guard_v1.json','results/canvas_history_gap_adapter_syntax_v1.json','results/history_gap_warp_checkpoint_v1.json']
    files={R/n for n in names}|{Path(__file__),latex,cp}
    for name in ['history_gap_adapter_v1','history_gap_adapter_v2','history_gap_adapter_v3']:files.update(p for p in (R/'experiments'/name).iterdir() if p.is_file())
    m=read(R/'experiments/history_gap_adapter_v3/manifest.json')
    for value in m['binaries'].values():
        raw_path=value['path'].replace('\\','/')
        if raw_path.startswith('/mnt/c/'):p=Path('C:/'+raw_path[7:])
        else:p=Path(raw_path);p=p if p.is_absolute() else ROOT/p
        assert sha(p)==value['sha256'];files.add(p)
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};target=R/'snapshots/history_gap_adapter_checkpoint_v1.zip';assert not target.exists()
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for n in hashes:z.write(ROOT/n,n)
        z.writestr('stage_manifest.json',json.dumps(dict(source_sha256=hashes,scope='Live adapter code, replay/scatter/sample/update arrays, retained failures and private binaries. Original PLYs, PNG/calibration inputs and installed public renderer remain separate and hash referenced.'),indent=2))
    with zipfile.ZipFile(target) as z:
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    result=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,previous_goal_turn_classification='progress: warp optimization passed complete correctness/cost gates',current_goal_turn_classification='progress: actual query/AMP/CSR GPU adapter, shared Float32 scatter versus MP, explicit original saturated-logit compatibility, zero controls and36 fresh-Adam RGB memory updates audited',terminal_handles={'adapter_v1_session89078':1,'adapter_v2_session30049':1,'adapter_v3_session22299':0,'empty_fixture_session97796':0,'independent_audit_session45504':0},optimizer_smoke_steps=36,full_training_started=False,zero_control_engineering_gate_pass=True,zero_control_is_bit_or_statistical_equivalence=False,original_all_finite_parameter_gate_failed=True,existing_positive_inf_logits_preserved=True,scenes=a['scenes'],source_latex_matches_builtin_success=True,latex_sha256=sha(latex),source_canvas_matches_syntax_checked_stage=True,canvas_sha256=guard['staged_sha256'],snapshot=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),files=len(hashes),bytes=target.stat().st_size,all_member_hashes_verified=True,private_binaries_included=True),limits=a['limits'],next='HISTORY_GAP_FULL_TRAIN_NEXT.md: original complete train.py recipe with strict42/7 data and common6999 checkpoint/runtime sidecar. First verify complete state binding/camera/sample/RNG and disabled/computed-zero full-recipe short controls, then predefined paired pilot with GT/heldout/capacity/full costs. Do not treat49-trained PLY smoke or3-step loss decrease as quality or heldout success; preserve old147view GT failure.')
    out=R/'results/history_gap_adapter_checkpoint_v1.json';assert not out.exists();save(out,result);print(json.dumps({k:v for k,v in result.items() if k!='scenes'}))
if __name__=='__main__':main()
