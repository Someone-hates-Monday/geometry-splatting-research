"""Seal bounded v2 evidence without modifying any prior sealed evidence."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_stable_window_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=OUT/'evidence_manifest.json';assert not target.exists()
    prior=json.loads((ROOT/'experiments/tgs_radius_contraction_v1/evidence_manifest.json').read_text())
    for e in prior['files']:
        p=ROOT/e['path'];assert p.stat().st_size==e['size'] and sha(p)==e['sha256']
    decision=json.loads((OUT/'promotion_decision.json').read_text())
    assert decision['protocol_gates_passed'] and not decision['expanded_full_image_gate_passed'] and not decision['performance_measurement_allowed']
    names=['attribute_tgs_forward_precision.py','prepare_tgs_stable_inputs.py','tgs_stable_window_backend.py',
        'probe_tgs_stable_window.py','audit_tgs_stable_window.py','summarize_tgs_stable_window.py',
        'analyze_tgs_stable_fd.py','diagnose_tgs_stable_image_changes.py','diagnose_tgs_small_sigma.py',
        'audit_tgs_small_sigma.py','seal_tgs_stable_window.py','TGS_STABLE_WINDOW_PROTOCOL_V2.md',
        'TGS_STABLE_WINDOW_RESULT_2026_10_20.md','TGS_SMALL_SIGMA_STABILITY_NOTE.md']
    files=[ROOT/name for name in names]
    files.extend(p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.csv','.cu','.so'] and '__pycache__' not in p.parts)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=entries,file_count=len(entries),bytes=sum(e['size'] for e in entries),prior_manifest_files_verified=len(prior['files']),
        exclusions='Mutable CONTINUATION.md, compiler object/cache/ninja files and this self-referential manifest are excluded; CUDA modules and generated CUDA sources are included.',promotion_decision=decision,training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    for e in entries:assert sha(ROOT/e['path'])==e['sha256']
    print(json.dumps(dict(file_count=len(entries),bytes=result['bytes'],prior_manifest_files_verified=len(prior['files']),manifest_sha256=sha(target),performance_measurement_allowed=False,training=False)),flush=True)


if __name__=='__main__':main()
