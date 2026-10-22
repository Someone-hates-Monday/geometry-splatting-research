"""Append the trace auditor/RN helper omitted by the primary filename glob; no old file edits."""
import datetime,json
from pathlib import Path
from seal_tgs_window_scene_v15 import ROOT,OUT,sha,write_new,deps

def main():
    primary=OUT/'evidence_manifest.json';expected='a1354ac425e2a01c888d778b2b5a0481a1d518095b8d9f69f507953e11864e72'
    assert sha(primary)==expected
    manifest=json.loads(primary.read_text());present={e['path'] for e in manifest['files']}
    auditor=ROOT/'audit_tgs_window_trace_v15.py';helper=ROOT/'audit_tgs_compensated_depth_trace.py'
    audit=json.loads((OUT/'trace/independent_audit.json').read_text())
    assert sha(auditor)==audit['script_sha256'] and sha(helper)==audit['RN_helper_sha256']
    required=deps([auditor,Path(__file__)])
    extra=sorted(p for p in required if p.relative_to(ROOT).as_posix() not in present)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in extra]
    target=OUT/'dependency_manifest_addendum.json'
    write_new(target,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        primary_manifest_sha256=expected,reason='Primary filename glob included window_scene scripts but omitted window_trace auditor and its RN helper.',
        files=entries,all_required_trace_dependencies_covered=True,primary_and_outputs_unchanged=True,training=False))
    for en in entries:assert sha(ROOT/en['path'])==en['sha256']
    assert sha(primary)==expected
    print(json.dumps(dict(additional_files=len(entries),bytes=sum(e['size'] for e in entries),addendum_sha256=sha(target))),flush=True)

if __name__=='__main__':main()
