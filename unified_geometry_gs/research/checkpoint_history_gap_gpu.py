from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    a=read(R/'results/history_gap_gpu_audit_v1.json');assert a['audit_gate_pass'] and a['new_seed13019_pass']==32
    latex=R/'median_depth_note.tex';expected='b2b18efab4b7220388b1db832a500f1cc928a9140fd7a53ef303ea286e345888';assert sha(latex)==expected
    cp=R/'results/latex_compile_status_v40.json';assert not cp.exists();save(cp,dict(path=str(latex),kind='success',source_sha256=expected,message='Current saved source compiled successfully with desktop built-in compiler.',checked_utc=datetime.now(timezone.utc).isoformat()))
    guard=read(R/'results/canvas_history_gap_gpu_guard_v1.json');assert sha(Path(guard['destination']))==guard['staged_sha256'] and not read(R/'results/canvas_history_gap_gpu_syntax_v1.json')['errors']
    names=['history_gap_gpu.py','history_gap_gpu_v2.py','history_gap_gpu.cu','benchmark_history_gap_gpu.py','benchmark_history_gap_gpu_v2.py','confirm_history_gap_gpu.py','confirm_history_gap_gpu_v2.py','recover_history_gap_gpu_confirm.py','profile_history_gap_gpu.py','verify_history_gap_gpu.py','build_history_gap_gpu_v2.py','HISTORY_GAP_GPU_PROTOCOL.md','HISTORY_GAP_GPU_CONFIRM_AMENDMENT.md','HISTORY_GAP_GPU_V2_AMENDMENT.md','HISTORY_GAP_GPU_RESULT.md','HISTORY_GAP_GPU_NEXT.md','CONTINUATION.md','history_gap_raw.py','history_gap_math.py','history_gap_extension.py','world_root_vjp.py','world_root_vjp_v2.py','exact_outlier_world_reference.py','stage_history_gap_gpu_canvas.py','canvas_history_gap_gpu_stage_v1.tsx','results/history_gap_gpu_audit_v1.json','results/history_gap_gpu_profile_v1.json','results/canvas_history_gap_gpu_guard_v1.json','results/canvas_history_gap_gpu_syntax_v1.json','results/history_gap_raw_checkpoint_v1.json']
    files={R/n for n in names}|{Path(__file__),latex,cp}
    for name in ['history_gap_gpu_v1','history_gap_gpu_v2','history_gap_gpu_confirm_v1','history_gap_gpu_confirm_v2','history_gap_gpu_confirm_v3','history_gap_gpu_confirm_v4']:
        folder=R/'experiments'/name;files.update(p for p in folder.iterdir() if p.is_file());m=read(folder/'manifest.json')
        if 'binary'in m:files.add(ROOT/m['binary']['path'])
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};target=R/'snapshots/history_gap_gpu_checkpoint_v1.zip';assert not target.exists()
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:z.write(ROOT/name,name)
        z.writestr('stage_manifest.json',json.dumps(dict(source_sha256=hashes,scope='GPU implementation, actual two .so binaries, full diagnostic arrays and retained confirmation failures. Prior raw/math/GT snapshots remain separate.'),indent=2))
    with zipfile.ZipFile(target) as z:
        for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    result=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,previous_goal_turn_classification='progress: raw chain, actual shared-ID updates and C1 loss extension proved/tested',current_goal_turn_classification='progress: GPU independent loss/seed/raw chain implemented; original real891 and corrected original32+new32 confirmed; four original synthetic failures retained and algebraically corrected; all-array/source/binary audit complete',terminal_handles={'gpu_v1_session34235':0,'gpu_v2_session14507':0,'confirm_v1_session68192':1,'confirm_v2_session27220':1,'recovery_v3':0,'confirm_v4_session66333':0,'profile':0,'independent_audit_session94607':0},real=a['real'],confirmation=a['confirmation'],new_seed13019_pass=32,special_coefficient_checks=10,invalid_input_rejected=True,source_latex_matches_builtin_success=True,latex_sha256=sha(latex),source_canvas_matches_syntax_checked_stage=True,canvas_sha256=guard['staged_sha256'],training_started=False,snapshot=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),files=len(hashes),bytes=target.stat().st_size,all_member_hashes_verified=True,actual_CUDA_binaries_included=True),limits=['Numerical Float64 signs, not interval certification.','Fixed development real rays and64 synthetic fixtures, no new real-scene training/GT/photometric proof.','79.067ms is a warm GPU geometry/loss/raw-backward batch, excludes setup/transfers/compilation and is not whole-training speed.','Original28/32 failure and reporting failures remain; original CPU/direction/147viewGT failures unchanged.'],next='Use HISTORY_GAP_GPU_NEXT.md: preserve serial source/binary, implement warp-per-ray loss/seed kernel in a new file; compare all existing MP gates and complete GPU costs in same process. Then direct-GPU fixed-sample/shared-ID training adapter with zero-weight identical-state control before paired training/GT/heldout/capacity claims.')
    out=R/'results/history_gap_gpu_checkpoint_v1.json';assert not out.exists();save(out,result);print(json.dumps(result))
if __name__=='__main__':main()
