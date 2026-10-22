"""Selected immutable stage snapshot; excludes all live pilot model/JSONL files."""
from pathlib import Path
import hashlib,json,time,zipfile
R=Path(__file__).resolve().parent;ROOT=R.parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    dest=R/'snapshots/history_gap_full_smoke_checkpoint_v1.zip';assert not dest.exists()
    audit=read(R/'results/history_gap_full_smoke_audit_v1.json');assert audit['audit_gate_pass']
    launch=R/'results/history_gap_pilot_launch_snapshot_v1.json';assert not launch.exists()
    live=read(R/'experiments/history_gap_pilot_v1/manifest.json')
    launch.write_text(json.dumps(dict(epoch=time.time(),snapshot_only=True,pilot_manifest=live),indent=2)+'\n',encoding='utf-8')
    files=set((R/'experiments/history_gap_full_smoke_v1').rglob('*'))
    names=['CONTINUATION.md','HISTORY_GAP_FULL_TRAIN_RESULT.md','HISTORY_GAP_FULL_TRAIN_NEXT.md',
        'HISTORY_GAP_FULL_TRAIN_PROTOCOL.md','HISTORY_GAP_PILOT_PROTOCOL.md','TRAINING_STALL_STATUS.md',
        'preflight_history_gap_full_train.py','train_history_gap_branch.py','run_history_gap_full_smoke.py',
        'verify_history_gap_full_smoke.py','train_history_gap_pilot.py','run_history_gap_pilot.py',
        'median_depth_note.tex','stage_history_gap_full_smoke_canvas.py','check_history_gap_full_smoke_canvas.cjs',
        'canvas_history_gap_full_smoke_stage_v1.tsx',Path(__file__).name,
        'results/history_gap_full_train_preflight_v1.json','results/history_gap_full_smoke_audit_v1.json',
        'results/history_gap_pilot_launch_snapshot_v1.json','results/latex_compile_status_v43.json',
        'results/canvas_history_gap_full_smoke_guard_v1.json','results/canvas_history_gap_full_smoke_syntax_v1.json',
        'experiments/history_gap_pilot_v1/sources.zip']
    files.update(R/name for name in names);files={p for p in files if p.is_file()}
    reference=read(R/'results/history_gap_adapter_checkpoint_v1.json')
    meta=dict(epoch=time.time(),status='complete',scope='Completed full-recipe24-update diagnostic and frozen pilot launch sources; full pilot still active.',
        previous_stage=reference,files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},
        active_exec_session=7120,live_pilot_directory=str(R/'experiments/history_gap_pilot_v1'),
        excluded='Live pilot model, checkpoint and growing JSONL/manifest are excluded; launch JSON is only a historical snapshot.')
    with zipfile.ZipFile(dest,'x') as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix in ['.zip','.npz'] else zipfile.ZIP_DEFLATED)
        z.writestr('snapshot_manifest.json',json.dumps(meta,indent=2)+'\n')
    with zipfile.ZipFile(dest) as z:
        for name,h in meta['files'].items():
            with z.open(name) as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==h,name
    meta.update(snapshot=str(dest),snapshot_sha256=sha(dest),snapshot_bytes=dest.stat().st_size,members_verified=len(meta['files']))
    out=R/'results/history_gap_full_smoke_checkpoint_v1.json';assert not out.exists()
    out.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ['status','snapshot','snapshot_sha256','snapshot_bytes','members_verified','active_exec_session']}),flush=True)
if __name__=='__main__':main()
