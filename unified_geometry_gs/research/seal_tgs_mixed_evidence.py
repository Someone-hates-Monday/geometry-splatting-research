"""Seal completed mixed/real VJP evidence after both independent audits."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_mixed_backward_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for relative in ['experiments/tgs_mixed_backward_v1/summary_audit.json',
                     'experiments/tgs_real_backward_v2/summary_audit.json',
                     'experiments/tgs_real_backward_v2/checkpoint_tensor_audit.json']:
        assert json.loads((ROOT/relative).read_text())['passed']
    old=ROOT/'experiments/tgs_backward_v1/evidence_manifest.json'
    for e in json.loads(old.read_text())['files']:assert sha(ROOT/e['path'])==e['sha256']
    source=ROOT/'sources/tgs_native_diagnostic_v1'
    source_entries=json.loads((source/'manifest.json').read_text())['files']
    for e in source_entries:assert sha(source/e['path'])==e['sha256']
    target=OUT/'evidence_manifest.json'
    assert not target.exists(),'Explicitly version any later seal; preserve completed evidence'
    paths=[old,source/'manifest.json',ROOT/'sources/tgs_real_checkpoint_v1/manifest.json',
           ROOT/'sources/tgs_real_checkpoint_v1/source_manifest.json']
    for relative in ['experiments/tgs_mixed_backward_v1','experiments/tgs_real_backward_v1','experiments/tgs_real_backward_v2']:
        paths += [p for p in (ROOT/relative).rglob('*') if p.is_file() and p.suffix in {'.json','.npz','.py','.txt'}]
    paths += [ROOT/name for name in ['tgs_mixed_reference.py','tgs_mixed_numpy.py','probe_tgs_mixed_backward.py',
              'probe_tgs_real_backward.py','audit_tgs_mixed_backward.py','audit_tgs_real_backward.py',
              'seal_tgs_mixed_evidence.py','TGS_MIXED_BACKWARD_DERIVATION.md','TGS_MIXED_REAL_BACKWARD_RESULT_2026_10_20.md']]
    entries=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))]
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries,
                prior_inventory_files_verified=40,source_archive_files_verified=len(source_entries),
                scope='Completed mixed RGB/SH and real-subset VJP checks; not a full-scene training or novelty certificate',
                exclusions='Large checkpoint and prior native binaries are identified by hashed manifests/audit receipts; mutable CONTINUATION.md excluded',
                training_performed=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    for e in json.loads(target.read_text())['files']:
        assert sha(ROOT/e['path'])==e['sha256'] and (ROOT/e['path']).stat().st_size==e['size']
    print(json.dumps(dict(passed=True,files_verified=len(entries),source_files_verified=len(source_entries),
                          prior_sealed_files_unchanged=40,total_bytes=sum(e['size'] for e in entries))))


if __name__=='__main__':main()
