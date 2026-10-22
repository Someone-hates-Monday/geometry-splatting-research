"""Seal failed-v1 outcomes, exact coverage certificate and causal diagnostics."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_radius_contraction_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    a=json.loads((OUT/'summary_audit.json').read_text());b=json.loads((OUT/'tile_coverage/summary_audit.json').read_text())
    assert a['evidence_audit_passed'] and b['evidence_audit_passed']
    assert not a['performance_measurement_allowed']
    paths=[p for p in OUT.rglob('*') if p.is_file() and (p.suffix in {'.json','.npz','.cu','.so'} or p.name=='build.ninja')]
    paths += [ROOT/name for name in ['verify_tgs_radius_contraction.py','tgs_contraction_backend.py',
        'prepare_tgs_contraction_inputs.py','probe_tgs_radius_contraction.py','audit_tgs_radius_contraction.py',
        'diagnose_tgs_tile_coverage.py','tgs_tile_upper_backend.py','diagnose_tgs_stress_coverage.py',
        'measure_tgs_tile_upper_real.py','audit_tgs_tile_coverage.py','seal_tgs_contraction_evidence.py',
        'TGS_RADIUS_CONTRACTION_RESULT_2026_10_20.md','TGS_TILE_COVERAGE_NOTE.md','TGS_RADIUS_CONTRACTION_PROTOCOL_V1.md']]
    prior_checks=0
    for directory in ['tgs_backward_v1','tgs_mixed_backward_v1','tgs_branch_aux_v1','tgs_backward_cost_v1']:
        p=ROOT/'experiments'/directory/'evidence_manifest.json';paths.append(p)
        for e in json.loads(p.read_text())['files']:
            assert sha(ROOT/e['path'])==e['sha256'];prior_checks+=1
    source=ROOT/'sources/tgs_native_diagnostic_v1';paths.append(source/'manifest.json')
    entries=json.loads((source/'manifest.json').read_text())['files']
    for e in entries:assert sha(source/e['path'])==e['sha256']
    files=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))]
    target=OUT/'evidence_manifest.json';assert not target.exists()
    receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
        prior_sealed_file_checks=prior_checks,source_archive_files_checked=len(entries),
        scope='contraction v1 failed advancement; coverage causal diagnostics do not overwrite failed outcomes',
        candidate_performance_measured=False,training=False,image_quality_gain_established=False,
        exclusions='Intermediate objects and mutable CONTINUATION.md; checkpoint origin covered by chained manifests')
    target.write_text(json.dumps(receipt,indent=2)+'\n')
    for e in json.loads(target.read_text())['files']:
        assert sha(ROOT/e['path'])==e['sha256'] and (ROOT/e['path']).stat().st_size==e['size']
    print(json.dumps(dict(passed=True,files_verified=len(files),prior_checks=prior_checks,source_files=len(entries),
                          total_bytes=sum(e['size'] for e in files))))


if __name__=='__main__':main()
