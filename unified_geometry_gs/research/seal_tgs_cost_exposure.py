"""Seal measured API cost, actual exposure counts, and an untested next protocol."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_backward_cost_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for relative in ['experiments/tgs_backward_cost_v1/summary_audit.json','experiments/tgs_branch_exposure_v1/summary_audit.json']:
        assert json.loads((ROOT/relative).read_text())['evidence_audit_passed']
    paths=[]
    for directory in ['experiments/tgs_backward_cost_v1','experiments/tgs_branch_exposure_v1']:
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and (
            p.suffix in {'.json','.npz','.cu','.h','.cpp','.so'} or p.name=='build.ninja')]
    paths += [ROOT/name for name in ['benchmark_tgs_backward.py','audit_tgs_backward_cost.py','tgs_branch_counter_backend.py',
        'measure_tgs_branch_exposure.py','audit_tgs_branch_exposure.py','seal_tgs_cost_exposure.py',
        'TGS_COST_EXPOSURE_RESULT_2026_10_20.md','TGS_RADIUS_CONTRACTION_PROTOCOL_V1.md']]
    checked=0
    for directory in ['tgs_backward_v1','tgs_mixed_backward_v1','tgs_branch_aux_v1']:
        old=ROOT/'experiments'/directory/'evidence_manifest.json';paths.append(old)
        for e in json.loads(old.read_text())['files']:
            assert sha(ROOT/e['path'])==e['sha256'];checked+=1
    source=ROOT/'sources/tgs_native_diagnostic_v1'
    paths.append(source/'manifest.json')
    source_entries=json.loads((source/'manifest.json').read_text())['files']
    for e in source_entries:assert sha(source/e['path'])==e['sha256']
    files=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))]
    target=OUT/'evidence_manifest.json';assert not target.exists()
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
                prior_sealed_file_checks=checked,source_archive_files_checked=len(source_entries),
                scope='Measured API cost and backward contributor-pair exposure; next contraction candidate remains untested',
                training=False,exclusions='Intermediate object files and mutable CONTINUATION.md; native/checkpoint origins covered by chained manifests')
    target.write_text(json.dumps(result,indent=2)+'\n')
    for e in json.loads(target.read_text())['files']:
        assert sha(ROOT/e['path'])==e['sha256'] and (ROOT/e['path']).stat().st_size==e['size']
    print(json.dumps(dict(passed=True,files_verified=len(files),prior_sealed_file_checks=checked,
                          source_files_checked=len(source_entries),total_bytes=sum(e['size'] for e in files))))


if __name__=='__main__':main()
