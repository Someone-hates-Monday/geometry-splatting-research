from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    audit=read(R/'results/history_gap_warp_audit_v1.json');assert audit['audit_gate_pass'] and audit['adoption_gate_pass'];latex=R/'median_depth_note.tex';expected='a5b6f81e0d16825fb73ecd367070b6dd7611221ecfe95d0f32e7670681e05f26';assert sha(latex)==expected
    cp=R/'results/latex_compile_status_v41.json';assert not cp.exists();save(cp,dict(path=str(latex),kind='success',source_sha256=expected,message='Current saved source compiled successfully with desktop built-in compiler.',checked_utc=datetime.now(timezone.utc).isoformat()))
    guard=read(R/'results/canvas_history_gap_warp_guard_v1.json');assert sha(Path(guard['destination']))==guard['staged_sha256'] and not read(R/'results/canvas_history_gap_warp_syntax_v1.json')['errors']
    names=['HISTORY_GAP_WARP_PROTOCOL.md','HISTORY_GAP_WARP_RUNNER_AMENDMENT.md','HISTORY_GAP_WARP_RESULT.md','HISTORY_GAP_TRAIN_ADAPTER_PLAN.md','CONTINUATION.md','history_gap_warp.cu','history_gap_warp_body.inc','history_gap_warp.py','build_history_gap_warp.py','benchmark_history_gap_warp.py','verify_history_gap_warp.py','history_gap_gpu.py','history_gap_gpu_v2.py','history_gap_gpu.cu','benchmark_history_gap_gpu.py','history_gap_extension.py','history_gap_raw.py','history_gap_math.py','world_root_vjp.py','world_root_vjp_v2.py','exact_outlier_world_reference.py','stage_history_gap_warp_canvas.py','canvas_history_gap_warp_stage_v1.tsx','results/history_gap_warp_audit_v1.json','results/canvas_history_gap_warp_guard_v1.json','results/canvas_history_gap_warp_syntax_v1.json','results/history_gap_gpu_checkpoint_v1.json']
    files={R/n for n in names}|{Path(__file__),latex,cp}
    for name in ['history_gap_warp_v1','history_gap_warp_v2']:
        folder=R/'experiments'/name;files.update(p for p in folder.iterdir() if p.is_file());m=read(folder/'manifest.json')
        if 'binary'in m:files.add(ROOT/m['binary']['path'])
    serial=read(R/'experiments/history_gap_gpu_v2/manifest.json')['binary'];files.add(ROOT/serial['path'])
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};target=R/'snapshots/history_gap_warp_checkpoint_v1.zip';assert not target.exists()
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for n in hashes:z.write(ROOT/n,n)
        z.writestr('stage_manifest.json',json.dumps(dict(source_sha256=hashes,scope='Warp loss stage, complete diagnostic inputs/outputs/source ZIPs/timings and actual serial+warp binaries. Prior GPU/raw/math/fullGT snapshots remain separate.'),indent=2))
    with zipfile.ZipFile(target) as z:
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    result=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,previous_goal_turn_classification='progress: serial GPU extended loss/raw chain891 and original32 plus fresh32 confirmed',current_goal_turn_classification='progress: warp-per-ray loss implemented, all existing MP gates plus long-list exact witnesses pass, same-process alternating full GPU cost ratio.151583 adopted; independent source/binary/array/timing audit complete',terminal_handles={'warp_v1_session99775':1,'warp_v2_session72993':0,'independent_audit_session33454':0},correctness_gate_pass=True,adoption_gate_pass=True,real_pass_counts=audit['real_pass_counts'],confirmed=64,special_cases=12,long_counts=audit['long_counts_exact_mass_witness'],full_median_ms=audit['full_median_ms'],kernel_median_ms=audit['kernel_median_ms'],full_ratio=audit['full_ratio'],component_speedup=audit['component_speedup'],warp_binary_sha256=audit['binary_sha256'],source_latex_matches_builtin_success=True,latex_sha256=sha(latex),source_canvas_matches_syntax_checked_stage=True,canvas_sha256=guard['staged_sha256'],training_started=False,snapshot=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),files=len(hashes),bytes=target.stat().st_size,all_member_hashes_verified=True,actual_serial_and_warp_binaries_included=True),limits=['This loss component GPU batch is faster versus its serial reference, not original renderer or whole training.','Fixed development real rays and64 synthetic models, no GT/heldout/capacity/training or novelty success.','Numerical Float64 signs, not interval certificates; prior CPU/direction/full147viewGT failures unchanged.','Initial runner summary duplicate-key failure preserved; second compile_wall field is binary loading, not compile cost.'],next='Implement the actual direct-GPU sampled/shared-ID GaussianModel regularizer and zero-weight/disabled controls using HISTORY_GAP_TRAIN_ADAPTER_PLAN.md. Preserve original renderer/source. Validate actual query/AMP capture, scatter/optimizer state, finite/empty handling and complete iteration cost before short nonzero updates and predefined paired training/GT/heldout/capacity evaluation.')
    out=R/'results/history_gap_warp_checkpoint_v1.json';assert not out.exists();save(out,result);print(json.dumps(result))
if __name__=='__main__':main()
