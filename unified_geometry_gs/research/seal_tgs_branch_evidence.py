"""Hash completed branch/auxiliary diagnosis and its real prevalence limits."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_branch_aux_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    audit=json.loads((OUT/'summary_audit.json').read_text())
    assert audit['evidence_audit_passed'] and audit['all_piecewise_gates']
    assert json.loads((OUT/'prior_regression.json').read_text())['all_gates']
    profile=json.loads((OUT/'real_branch_profile.json').read_text())
    assert profile['numpy_and_torch_floor_classification_equal']
    assert profile['script_sha256']==sha(ROOT/'profile_tgs_real_branches.py')
    for i,row in zip([2,3],profile['per_train_camera']):
        assert row['baseline_sha256']==sha(ROOT/f'experiments/tgs_real_event_v1/baseline_{i}.npz')
    source=ROOT/'sources/tgs_native_diagnostic_v1'
    source_entries=json.loads((source/'manifest.json').read_text())['files']
    for e in source_entries:assert sha(source/e['path'])==e['sha256']
    static=ROOT/'sources/tgs_subdivision_static_v1'
    entries=json.loads((static/'manifest.json').read_text())['files']
    main_paths=['train.py','arguments/__init__.py','triangle_renderer/__init__.py']
    primary=[]
    for p in main_paths:
        e=next(e for e in entries if e['repo']=='triangle-splatting' and e['path']==p)
        path=static/'triangle-splatting'/p;assert sha(path)==e['sha256'];primary.append(path)
    target=OUT/'evidence_manifest.json';assert not target.exists()
    paths=[p for p in OUT.rglob('*') if p.is_file() and (p.suffix in {'.json','.npz','.py','.cu','.so'} or p.name=='build.ninja')]
    paths += [ROOT/name for name in ['tgs_aux_reference.py','tgs_aux_numpy.py','probe_tgs_branch_aux.py',
        'tgs_piecewise_backend.py','regress_tgs_piecewise.py','profile_tgs_real_branches.py',
        'audit_tgs_branch_aux.py','seal_tgs_branch_evidence.py','TGS_BRANCH_AUX_DERIVATION.md','TGS_BRANCH_AUX_RESULT_2026_10_20.md']]
    paths += primary+[source/'manifest.json',static/'manifest.json']
    for relative in ['experiments/tgs_backward_v1/evidence_manifest.json','experiments/tgs_mixed_backward_v1/evidence_manifest.json']:
        old=ROOT/relative;paths.append(old)
        for e in json.loads(old.read_text())['files']:assert sha(ROOT/e['path'])==e['sha256']
    files=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))]
    receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
        source_files_verified=len(source_entries),training=False,
        scope='Piecewise derivative and auxiliary-map diagnostics; old failure cases and actual-scope limits retained',
        exclusions='Intermediate objects and mutable CONTINUATION.md; large checkpoint remains covered by previously verified provenance')
    target.write_text(json.dumps(receipt,indent=2)+'\n')
    for e in json.loads(target.read_text())['files']:
        assert sha(ROOT/e['path'])==e['sha256'] and (ROOT/e['path']).stat().st_size==e['size']
    print(json.dumps(dict(passed=True,files_verified=len(files),source_files_verified=len(source_entries),
                          total_bytes=sum(e['size'] for e in files))))


if __name__=='__main__':main()
