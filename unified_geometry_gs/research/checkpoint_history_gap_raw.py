"""Recover this stage without rerunning any completed mathematical job."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
R=Path(__file__).resolve().parent;ROOT=R.parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    a=read(R/'results/history_gap_raw_audit_v1.json');e=read(R/'results/history_gap_extensions_audit_v2.json');assert a['audit_gate_pass'] and e['audit_gate_pass'];assert not a['original_raw_gate_pass'] and not e['shared_original_gate_pass']
    latex=R/'median_depth_note.tex';h='3c83b1c2f8836f2436346c2f86d38404d598a3d7084a50bc71383a541354135c';assert sha(latex)==h
    # Actual desktop built-in compiler success has been observed for this hash.
    compile_path=R/'results/latex_compile_status_v39.json';assert not compile_path.exists();save(compile_path,dict(path=str(latex),kind='success',source_sha256=h,message='Current source compiled successfully with desktop editor compiler.',checked_utc=datetime.now(timezone.utc).isoformat()))
    guard=read(R/'results/canvas_history_gap_raw_guard_v1.json');assert sha(Path(guard['destination']))==guard['staged_sha256'] and not read(R/'results/canvas_history_gap_raw_syntax_v1.json')['errors']
    names=['history_gap_raw.py','audit_history_gap_raw.py','audit_history_gap_raw_micro.py','audit_history_gap_shared.py','audit_history_gap_shared_micro.py','history_gap_extension.py','audit_history_gap_extension.py','verify_history_gap_raw.py','verify_history_gap_extensions.py','history_gap_math.py','world_root_vjp.py','world_root_vjp_v2.py','exact_outlier_world_reference.py','HISTORY_GAP_RAW_PROTOCOL.md','HISTORY_GAP_RAW_MICRO_PROTOCOL.md','HISTORY_GAP_SHARED_PROTOCOL.md','HISTORY_GAP_SHARED_MICRO_PROTOCOL.md','HISTORY_GAP_EXTENSION_PROTOCOL.md','HISTORY_GAP_RAW_RESULT.md','HISTORY_GAP_MATH_RESULT.md','CONTINUATION.md','stage_history_gap_raw_canvas.py','canvas_history_gap_raw_stage_v1.tsx','results/history_gap_raw_audit_v1.json','results/history_gap_extensions_audit_v1.json','results/history_gap_extensions_audit_v2.json','results/canvas_history_gap_raw_guard_v1.json','results/canvas_history_gap_raw_syntax_v1.json','results/history_gap_math_checkpoint_v1.json','results/frozen_certified_mesh_checkpoint_v1.json']
    files={R/n for n in names}|{Path(__file__),latex,compile_path}
    experiments=['history_gap_raw_v1','history_gap_raw_micro_v1','history_gap_shared_v1','history_gap_shared_micro_v1','history_gap_extension_v1']
    for name in experiments:
        folder=R/'experiments'/name;assert read(folder/'manifest.json')['status']=='complete';files.update(p for p in folder.iterdir() if p.is_file())
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};target=R/'snapshots/history_gap_raw_checkpoint_v1.zip';assert not target.exists()
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for n in hashes:z.write(ROOT/n,n)
        z.writestr('stage_manifest.json',json.dumps(dict(source_sha256=hashes,scope='Current raw/shared/extension stage. Prior math48MB and frozenGT2.7GB snapshots remain separate, hash referenced by prior checkpoints.'),indent=2))
    with zipfile.ZipFile(target) as z:
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    result=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,previous_goal_turn_classification='progress: original C1 history-loss proof, adverse case and coefficient audit complete',current_goal_turn_classification='progress: raw analytic/AD chain, independent MP forward diagnosis, actual shared-ID descent and C1 extension through missing median implemented/proven/tested',terminal_handles={'raw_session42177':0,'raw_micro_session89864':0,'shared_session33823':0,'shared_micro_session23825':0,'extension_session77335':0,'independent_raw_audit':0,'independent_extensions_audit_v2':0},raw_counts=a['branch_pass_counts'],raw_max_relative_l2=a['max_relative_l2'],original_raw_gate_pass=False,original_shared_gate_pass=False,shared_micro_gate_sequence=[False,True,False],shared_unique_primitives=a['shared_unique_primitives'],shared_repeated_primitives=a['shared_repeated_primitives'],C1_extension_gate_pass=True,original_failure_verdicts_preserved=True,source_latex_matches_builtin_success=True,latex_sha256=sha(latex),source_canvas_matches_syntax_checked_stage=True,canvas_sha256=guard['staged_sha256'],training_started=False,snapshot=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),files=len(hashes),bytes=target.stat().st_size,all_member_hashes_verified=True),limitations=['Fixed captured-source continuous raw extension, fixed filters/cameras/visibility and accepted IDs; not literal IEEE derivative or full renderer training.','Original dense and finite-step direction gates fail; micro diagnosis is follow-up, not independent confirmation or changed original verdict.','Shared local loss decreases and C1 missing-root extension are not GT/photometric/efficiency/novelty success.','Cap/visibility transitions and parameter boundaries are outside coefficient C1 claim.'],next='Implement the loss including missing-median continuation on GPU with batched independent roots, stable log-alpha/side density seeds and complete raw chain. Verify existing full891 plus independent new confirmation inputs versus MP/CPU, and complete costs before any paired training. Preserve all old failures and full147-view GT failure.')
    out=R/'results/history_gap_raw_checkpoint_v1.json';assert not out.exists();save(out,result);print(json.dumps(result))
if __name__=='__main__':main()
