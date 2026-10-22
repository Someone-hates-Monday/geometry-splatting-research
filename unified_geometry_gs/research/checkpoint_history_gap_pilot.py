"""Archive completed stage evidence; large models and datasets remain hash references."""
from pathlib import Path
import hashlib,json,time,zipfile
R=Path(__file__).resolve().parent;ROOT=R.parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    target=R/'snapshots/history_gap_pilot_checkpoint_v1.zip';assert not target.exists()
    pilot=R/'experiments/history_gap_pilot_v1';post=R/'experiments/history_gap_posthoc_v1'
    pm=read(pilot/'manifest.json');pa=read(R/'results/history_gap_pilot_audit_v1.json')
    assert pm['status']=='complete' and not pm['gate_pass'] and pa['audit_gate_pass'] and not pa['pilot_gate_pass']
    assert read(post/'manifest.json')['status']=='complete' and read(R/'results/history_gap_posthoc_audit_v1.json')['audit_gate_pass']
    allowed={'.json','.jsonl','.zip','.npz','.log','.txt','.py'}
    files={p for folder in [pilot,post] for p in folder.rglob('*') if p.is_file() and p.suffix in allowed}
    files.update(ROOT/name for name in pm['source_sha256'])
    names=['CONTINUATION.md','HISTORY_GAP_PILOT_RESULT.md','HISTORY_GAP_PILOT_PROTOCOL.md','HISTORY_GAP_FULL_TRAIN_NEXT.md',
        'HISTORY_GAP_POSTHOC_PROTOCOL.md','HISTORY_GAP_RELATED_WORK_2026_10_16.md','TGS_SUBDIVISION_AUDIT_NEXT.md',
        'TRAINING_STALL_STATUS.md','median_depth_note.tex','verify_history_gap_pilot.py','verify_history_gap_pilot_initial_f64_audit.py',
        'inspect_history_gap_pilot_mechanism.py','verify_history_gap_posthoc.py','stage_history_gap_pilot_canvas.py',
        'check_history_gap_pilot_canvas.cjs','canvas_history_gap_pilot_stage_v1.tsx',Path(__file__).name,
        'results/history_gap_pilot_audit_v1.json','results/history_gap_pilot_partial_audit_v1.json',
        'results/history_gap_posthoc_audit_v1.json','results/history_gap_related_work_search_v1.json',
        'results/latex_compile_status_v44.json','results/canvas_history_gap_pilot_guard_v1.json','results/canvas_history_gap_pilot_syntax_v1.json']
    files.update(R/name for name in names)
    binaries=read(pilot/'history_gap/history_gap_training_integration.json')['binaries']
    for b in [binaries['query'],binaries['csr']]:
        name=b.get('binary_relative_path')
        path=ROOT/name if name else Path('C:/'+b['binary_path'][7:])
        assert sha(path)==b['binary_sha256'];files.add(path)
    path=ROOT/binaries['loss']['path'];assert sha(path)==binaries['loss']['sha256'];files.add(path)
    refs={name:h for name,h in pm['artifacts_sha256'].items() if Path(name).suffix in ['.pth','.ply']}
    meta=dict(status='complete',epoch=time.time(),client_date='2026-10-16',pilot_gate_pass=False,
        previous_stage=read(R/'results/history_gap_full_smoke_checkpoint_v1.json')['snapshot_sha256'],
        model_and_mesh_hash_references=refs,input_hash_references=pm['input_sha256'],
        files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},
        scope='Completed negative30k pair, all scalar/source/gradient/posthoc evidence and private binaries; no active job.',
        excluded='Large final checkpoints/PLY/mesh and original datasets remain at recorded paths with hashes; not a complete data/model backup.')
    with zipfile.ZipFile(target,'x') as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix in ['.zip','.npz','.so'] else zipfile.ZIP_DEFLATED)
        z.writestr('snapshot_manifest.json',json.dumps(meta,indent=2)+'\n')
    with zipfile.ZipFile(target) as z:
        for name,h in meta['files'].items():
            with z.open(name) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,name
    meta.update(snapshot=str(target),snapshot_sha256=sha(target),snapshot_bytes=target.stat().st_size,members_verified=len(meta['files']))
    dest=R/'results/history_gap_pilot_checkpoint_v1.json';assert not dest.exists()
    dest.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ['status','pilot_gate_pass','snapshot','snapshot_sha256','snapshot_bytes','members_verified']}),flush=True)
if __name__=='__main__':main()
