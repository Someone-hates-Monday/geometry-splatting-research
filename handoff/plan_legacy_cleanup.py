"""Prepare a bounded, hashed pruning plan. This script never deletes files."""
import hashlib,json,os,re
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'handoff'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def record(p):
    assert p.is_file() and not p.is_symlink()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=sha(p))
assert not (OUT/'cleanup_plan.json').exists(), 'Do not overwrite a historical cleanup plan.'
candidates=[];preserved=[]
for tau in ['0.3','0.5','1.2']:
    folder=ROOT/f'unified_geometry_gs/output_compare/ggs_tau_{tau}/scan105'
    for name in ['chkpnt30000.pth','point_cloud/iteration_30000/point_cloud.ply','point_cloud/iteration_7000/point_cloud.ply','vis/results.json','cfg_args','cameras.json','input.ply','multi_view.json','recon.ply','recon_post.ply','recon_aligned.ply']:
        p=folder/name;assert p.exists(),str(p);preserved.append(record(p))
    for step in range(10000,30000,1000):
        candidates.append(record(folder/f'chkpnt{step}.pth'))
    print(f'hashed tau={tau}',flush=True)
# Search source/notes/receipts for literal checkpoint paths or content hashes.
# This cannot prove the absence of dynamic external consumers; inspect scripts too.
refs=[];family_refs=[];errors=[];scanned=0
extensions={'.md','.tex','.py','.cu','.h','.cpp','.json','.sh','.txt','.yaml','.yml','.csv','.env'}
for folder,dirs,files in os.walk(ROOT,followlinks=False):
    dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','node_modules','handoff'} and not (Path(folder)/d).is_symlink()]
    for name in files:
        p=Path(folder)/name
        if p.suffix.lower() not in extensions:continue
        try:
            if p.stat().st_size>64*1024*1024:
                errors.append(dict(path=p.relative_to(ROOT).as_posix(),reason='text larger than64MiB'));continue
            s=p.read_text(encoding='utf8',errors='replace').replace('\\\\','/').replace('\\','/');scanned+=1
        except OSError as e:
            errors.append(dict(path=str(p),reason=type(e).__name__));continue
        if 'ggs_tau_' in s:family_refs.append(p.relative_to(ROOT).as_posix())
        for item in candidates:
            short='/'.join(item['path'].split('/')[-3:])
            if short in s or item['sha256'] in s:refs.append(dict(source=p.relative_to(ROOT).as_posix(),candidate=item['path']))
assert not refs,refs
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),workspace=str(ROOT),scope='Only the60 intermediate 10000..29000 step checkpoints of historical tau0.3/0.5/1.2 scan105 runs. No directory, data, PLY, final30000 or research artifact deletion.',reason='Completed tau sweep uses final point clouds and vis/results.json; intermediate optimizer states are not needed for the retained final evaluation. Exact intermediate resume capability will be lost. scripts/cleanup_outputs.sh previously prescribed broader pruning; this plan is narrower.',candidates=candidates,candidate_count=len(candidates),candidate_bytes=sum(x['bytes'] for x in candidates),preserved_essential=preserved,reference_audit=dict(text_files_scanned=scanned,literal_candidate_hits=refs,family_reference_files=family_refs,errors=errors,limitations='Literal-path/hash scan only, not a proof about dynamic or external consumers. run_research_p3/p4 generate checkpoints; judge_g1 reads vis/results.json. No recovery copy is created for pruned intermediate optimizer states.'))
(OUT/'cleanup_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(count=plan['candidate_count'],bytes=plan['candidate_bytes'],GiB=plan['candidate_bytes']/2**30,text_files_scanned=scanned,literal_hits=refs,errors=errors)),flush=True)
