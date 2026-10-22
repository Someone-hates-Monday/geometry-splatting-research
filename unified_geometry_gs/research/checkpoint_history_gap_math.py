"""Small recoverable stage snapshot, independent of the earlier 2.7GB archive."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
R=Path(__file__).resolve().parent;ROOT=R.parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    audit=read(R/'results/history_gap_math_audit_v3.json');assert audit['gate_pass']
    latex=R/'median_depth_note.tex';expected='d344146d5f7ae0d2e6ab81488195eeda3bbe3b995a3f6ad2c3b92c6674d23bbc';assert sha(latex)==expected
    # Actual built-in compiler succeeded on this exact saved source before this
    # record was written. This script does not itself invoke a shell TeX compiler.
    compile_record=dict(path=str(latex),kind='success',message='Current source compiled successfully with desktop editor compiler.',source_sha256=expected,checked_utc=datetime.now(timezone.utc).isoformat())
    compile_out=R/'results/latex_compile_status_v38.json';assert not compile_out.exists();save(compile_out,compile_record)
    guard=read(R/'results/canvas_history_gap_guard_v1.json');assert sha(Path(guard['destination']))==guard['staged_sha256'];assert not read(R/'results/canvas_history_gap_syntax_v1.json')['errors']
    files={Path(__file__),latex,R/'CONTINUATION.md',R/'history_gap_math.py',R/'audit_history_gap_math.py',R/'verify_history_gap_math.py',R/'stage_history_gap_canvas.py',R/'install_history_gap_canvas.py',R/'canvas_history_gap_stage_v1.tsx',R/'HISTORY_GAP_REGULARIZER_CANDIDATE.md',R/'HISTORY_GAP_MATH_PROTOCOL.md',R/'HISTORY_GAP_MATH_AMENDMENT_V2.md',R/'HISTORY_GAP_MATH_AMENDMENT_V3.md',R/'HISTORY_GAP_MATH_RESULT.md',R/'FROZEN_CERTIFIED_MESH_RESULT.md',compile_out,R/'results/history_gap_math_audit_v3.json',R/'results/canvas_history_gap_guard_v1.json',R/'results/canvas_history_gap_syntax_v1.json',R/'results/frozen_certified_mesh_checkpoint_v1.json',R/'results/frozen_certified_mesh_audit_v1.json'}
    for version in [1,2,3]:
        files.update(p for p in (R/f'experiments/history_gap_math_v{version}').iterdir() if p.is_file())
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};target=R/'snapshots/history_gap_math_checkpoint_v1.zip';assert not target.exists()
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:z.write(ROOT/name,name)
        z.writestr('stage_manifest.json',json.dumps(dict(scope='Current math stage and prior GT audit reference; previous 2.7GB snapshot remains separate.',source_sha256=hashes),indent=2))
    with zipfile.ZipFile(target) as z:
        for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    result=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,current_goal_turn_classification='progress: C1 mathematical supplement, adverse counterexample, stable coefficient adjoint and fixed real-source checks completed; optimization utility and raw chain remain open',terminal_handles={'v1_session10868':1,'v2_session5867':0,'v3_session57182':0,'independent_artifact_audit':0},rays=audit['rays'],nodes=audit['nodes'],pass_counts=audit['pass_counts'],max_relative_l2=audit['max_relative_l2'],mathematical_theorem_scope='finite fixed N, independent continuous alpha/peak/rho, 0<alpha<1,rho>0,Omega>log2; L=C1, no full raw cap/visibility/IEEE assertion',adverse_vanishing_gradient_confirmed=True,training_started=False,source_latex_matches_builtin_success=sha(latex)==expected,source_canvas_matches_syntax_checked_stage=True,canvas_sha256=guard['staged_sha256'],snapshot=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),files=len(hashes),bytes=target.stat().st_size,all_member_hashes_verified=True),limits=audit['limitations'],next='Test fixed-source raw-parameter directional derivatives and constrained small-step descent coverage, retaining high-loss near-zero-gradient cases. Only then consider CUDA/paired training. Do not rerun or relabel the failed full frozen GT stage.')
    out=R/'results/history_gap_math_checkpoint_v1.json';assert not out.exists();save(out,result);print(json.dumps(result))
if __name__=='__main__':main()
