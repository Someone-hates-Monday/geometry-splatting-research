"""Create a hash inventory after the independent audit; no rerender or training."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/tgs_backward_v1'


def main():
    summary = json.loads((OUT / 'summary_audit.json').read_text())
    assert summary['passed'] and not summary['training_performed']
    target = OUT / 'evidence_manifest.json'
    assert not target.exists(), 'Preserve existing inventories; explicitly version a later seal'
    files = [p for p in OUT.rglob('*') if p.is_file() and (
        p.suffix in {'.json', '.npz', '.md', '.py', '.cu', '.so'} or p.name == 'build.ninja')]
    files += [ROOT / name for name in ['probe_tgs_backward.py', 'tgs_backward_fixed_backend.py',
              'audit_tgs_backward.py', 'seal_tgs_backward_evidence.py',
              'TGS_BACKWARD_DERIVATION.md', 'TGS_BACKWARD_RESULT_2026_10_20.md']]
    files += [ROOT / 'sources/tgs_native_diagnostic_v1/manifest.json',
              ROOT / 'experiments/tgs_subdivision_v1/sh_direction_v1/fixed_build/tgs_diag_6d61f4c_shfix_v1.so']
    entries = [dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))]
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope='Hash inventory of completed fixed-input VJP diagnosis; not a scholarly novelty certificate',
                  files=entries, excluded='Intermediate object files and mutable CONTINUATION.md; 452 archived files verified by audit',
                  training_performed=False)
    target.write_text(json.dumps(result, indent=2) + '\n')
    for entry in json.loads(target.read_text())['files']:
        p = ROOT / entry['path']
        assert p.stat().st_size == entry['size']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256']
    print(json.dumps(dict(files_verified=len(entries), total_bytes=sum(e['size'] for e in entries),
                          inventory=str(target), passed=True)))


if __name__ == '__main__':
    main()
