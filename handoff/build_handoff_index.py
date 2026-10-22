"""Read-only project inventory and handoff link checks; never runs research jobs."""
import argparse,csv,hashlib,json,os,re,shutil
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'handoff';RESEARCH=ROOT/'unified_geometry_gs/research'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def kind(p):
    n=p.name.upper()
    if p.suffix.lower() in {'.py','.cu','.h','.cpp'}:return 'implementation'
    if p.suffix.lower()=='.tex':return 'latex'
    if any(t in n for t in ['PROTOCOL','PLAN','NEXT']):return 'protocol_or_proposal'
    if 'RESULT' in n:return 'result_report'
    return 'audit_theory_or_record'
def index():
    records=[]
    paths=list(RESEARCH.glob('*'))+list((ROOT/'unified_geometry_gs/docs').glob('*'))
    for p in sorted(paths):
        if p.is_file() and p.suffix.lower() in {'.md','.tex','.py','.cu','.h','.cpp'}:
            records.append(dict(path=p.relative_to(ROOT).as_posix(),kind=kind(p),bytes=p.stat().st_size,sha256=digest(p)))
    manifests=[]
    for parent in [RESEARCH/'experiments',RESEARCH/'results']:
        for folder,dirs,files in os.walk(parent,followlinks=False):
            dirs[:]=[d for d in dirs if d not in {'.git','build','__pycache__','node_modules','fixed_src','archive','source'}]
            for name in files:
                p=Path(folder)/name
                if name in {'evidence_manifest.json','manifest.json','input_plan.json','execution_plan.json'} or (Path(folder)==RESEARCH/'results' and name.endswith('.json')):
                    size=p.stat().st_size;manifests.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=size,sha256=digest(p) if size<=16*1024*1024 else None))
    write(OUT/'research_records.json',dict(snapshot_date='2026-10-22',scope='top-level research code/notes and subproject docs, selected experiment manifests and top-level results JSON; not a claim of all artifact hashes verified',records=records,manifests=sorted(manifests,key=lambda r:r['path'])))
    lines=['# 自动研究目录索引','', '2026-10-22。完整元数据见 [research_records.json](research_records.json)。这是目录/哈希索引，不代表全部历史结论重新审计。','', '先读 [交接入口](../HANDOFF.md) 与 [研究脉络](RESEARCH_MAP.md)，不要按本索引文件名顺序重跑实验。','']
    for category in ['result_report','audit_theory_or_record','protocol_or_proposal','latex','implementation']:
        lines.extend(['## '+category,''])
        for r in records:
            if r['kind']==category:lines.append(f"- [{Path(r['path']).name}](../{r['path']})")
        lines.append('')
    lines.extend(['## 原始证据入口','', f'索引到 {len(manifests)} 份 manifest/plan/顶层结果 JSON；路径、字节数和小文件哈希保存在 JSON。大于16MiB者只列元数据，未在本次索引读取全部内容。',''])
    (OUT/'RECORDS_INDEX.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(records=len(records),manifests=len(manifests))),flush=True)
def inventory():
    totals=defaultdict(lambda:dict(files=0,bytes=0));large=[];nested_git=[];skipped=[];errors=[];sensitive=[]
    def onerror(e):errors.append(dict(path=str(e.filename),error=type(e).__name__))
    for folder,dirs,files in os.walk(ROOT,followlinks=False,onerror=onerror):
        kept=[]
        for d in dirs:
            p=Path(folder)/d
            if d=='.git':nested_git.append(p.relative_to(ROOT).as_posix());continue
            if d in {'node_modules','__pycache__'} or p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction()):skipped.append(p.relative_to(ROOT).as_posix());continue
            kept.append(d)
        dirs[:]=kept
        for n in files:
            p=Path(folder)/n
            if p.is_symlink():continue
            try:size=p.stat().st_size
            except OSError as e:errors.append(dict(path=str(p),error=type(e).__name__));continue
            rel=p.relative_to(ROOT).as_posix();top=rel.split('/')[0] if '/' in rel else '(root files)';totals[top]['files']+=1;totals[top]['bytes']+=size
            if size>=50*1024*1024:large.append(dict(path=rel,bytes=size,MiB=round(size/2**20,3)))
            if n=='.env' or n.endswith(('.pem','.key')) or n.lower() in {'credentials.json','token.json','tokens.json'}:sensitive.append(rel)
    large.sort(key=lambda x:-x['bytes'])
    with (OUT/'large_file_candidates.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','MiB']);w.writeheader();w.writerows(large)
    summary=dict(snapshot_date='2026-10-22',metadata_only=True,top_level=dict(totals),large_file_threshold_MiB=50,large_files=len(large),total_scanned_files=sum(x['files'] for x in totals.values()),total_scanned_bytes=sum(x['bytes'] for x in totals.values()),nested_git_paths=nested_git,skipped_directories=skipped,filename_only_sensitive_candidates=sensitive,errors=errors,not_a_secret_or_license_audit=True,gitignore_effect_not_simulated=True)
    write(OUT/'workspace_inventory.json',summary);print(json.dumps({k:summary[k] for k in ['total_scanned_files','total_scanned_bytes','large_files','nested_git_paths','errors']}),flush=True)
def verify():
    docs=[ROOT/'HANDOFF.md',*OUT.glob('*.md')];bad=[];checked=0
    for p in docs:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf8')):
            target=target.strip('<>');u=urlsplit(target)
            if u.scheme or target.startswith('#'):continue
            checked+=1;dest=(p.parent/unquote(u.path)).resolve()
            if not dest.exists():bad.append(dict(document=p.relative_to(ROOT).as_posix(),target=target))
    write(OUT/'link_check.json',dict(checked_local_links=checked,missing=bad));print(json.dumps(dict(checked_local_links=checked,missing=bad)),flush=True)
    if bad:raise SystemExit(1)
def receipts():
    chosen=[];omitted=[]
    for name in ['tgs_batch_score_v22','tgs_shared_loss_v26','tgs_joint_edit_v27','tgs_policy_transfer_v28','tgs_joint_policy_v29']:
        folder=RESEARCH/'experiments'/name
        chosen.extend(folder.glob('*.json'));chosen.extend(folder.glob('*failure*.txt'))
        if (folder/'timing').is_dir():chosen.extend((folder/'timing').glob('*.json'));chosen.extend((folder/'timing').glob('*failure*.txt'))
    for name in ['tgs_shared_loss_v26','tgs_policy_transfer_v28','tgs_joint_policy_v29']:
        folder=RESEARCH/'experiments'/name
        chosen.extend(folder.glob('*.png'));chosen.extend(folder.glob('*.pdf'))
    chosen.extend((ROOT/'unified_geometry_gs/output_compare/metrics').glob('*.json'))
    for tau in ['0.3','0.5','1.2']:
        folder=ROOT/f'unified_geometry_gs/output_compare/ggs_tau_{tau}/scan105'
        chosen.extend([folder/'vis/results.json',folder/'cfg_args'])
    for name in ['evidence_audit.json','factor_quality55_report_v1.json','factor_holdout55_quality_report_v1.json','history_gap_pilot_audit_v1.json','training_stall_diagnosis_v1.json','thread_stream_diagnosis_v2.json']:
        p=RESEARCH/'results'/name
        if p.exists():chosen.append(p)
        else:omitted.append(dict(path=p.relative_to(ROOT).as_posix(),reason='not present at this path; consult research index'))
    entries=[]
    for p in sorted(set(chosen)):
        rel=p.relative_to(ROOT);size=p.stat().st_size
        if size>2*1024*1024:omitted.append(dict(path=rel.as_posix(),reason='larger than curated2MiB receipt limit'));continue
        dst=OUT/'evidence_receipts'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert digest(dst)==digest(p)
        else:shutil.copy2(p,dst)
        assert digest(dst)==digest(p)
        entries.append(dict(original=rel.as_posix(),copy=dst.relative_to(ROOT).as_posix(),bytes=size,sha256=digest(dst)))
    write(OUT/'copied_receipts.json',dict(snapshot_date='2026-10-22',entries=entries,omitted=omitted,raw_images_checkpoints_and_binaries_not_copied=True))
    print(json.dumps(dict(copied=len(entries),bytes=sum(e['bytes'] for e in entries),omitted=omitted)),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['index','inventory','verify','receipts']);args=a.parse_args();globals()[args.action]()
