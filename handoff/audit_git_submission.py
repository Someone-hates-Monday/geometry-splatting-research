"""Audit candidate/staged Git contents without printing credential values."""
import argparse,hashlib,json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT='handoff/git_submission_audit.json'
TEXT={'.py','.cu','.cpp','.h','.hpp','.inl','.md','.tex','.txt','.sh','.ps1','.json','.jsonl','.yaml','.yml','.toml','.ini','.cfg','.env','.html','.js','.ts','.tsx','.css','.ipynb','.cmake','.cmd'}
PATTERNS={
    'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'github_token':rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    'github_fine_grained':rb'\bgithub_pat_[A-Za-z0-9_]{50,}\b',
    'aws_access_key':rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'openai_key':rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b',
    'slack_token':rb'\bxox[baprs]-[A-Za-z0-9-]{24,}\b',
    'credential_assignment':rb'''(?i)(?:api[_-]?key|access[_-]?token|password|secret[_-]?key)\s*[:=]\s*['"][^'"\r\n]{12,}['"]''',
}
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--staged',action='store_true');a=parser.parse_args()
    staged=[x.decode('utf8') for x in git('ls-files','--cached','-z').split(b'\0') if x]
    paths=staged if a.staged else sorted(set(staged)|{x.decode('utf8') for x in git('ls-files','--others','--exclude-standard','-z').split(b'\0') if x})
    entries=[];findings=[];oversized=[];mismatched=[];forbidden=[];mode_issues=[];staged_blobs={}
    object_format=git('rev-parse','--show-object-format').decode().strip()
    if a.staged:
        for row in git('ls-files','--stage','-z').split(b'\0'):
            if not row:continue
            meta,name=row.split(b'\t',1);mode,oid,stage=meta.split();path=name.decode('utf8');staged_blobs[path]=oid.decode()
            if mode in {b'120000',b'160000'} or stage!=b'0':mode_issues.append(path)
    for path in paths:
        if path==REPORT:continue # This report cannot contain its own content hash.
        p=ROOT/path;data=p.read_bytes()
        if a.staged and hashlib.new(object_format,b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=staged_blobs[path]:mismatched.append(path)
        n=len(data);entries.append(dict(path=path,bytes=n,sha256=sha(data)))
        if n>=50*1024*1024:oversized.append(dict(path=path,bytes=n))
        if path!='unified_geometry_gs/dtu_eval/eval.py' and re.match(r'^(?:dtu/|unified_geometry_gs/(?:dtu_eval|output[^/]*|research/(?:experiments|sources|snapshots|results))/|tools/guizang-ppt-skill/)',path):forbidden.append(path)
        if p.suffix.lower() in {'.pth','.pt','.ply','.so','.pyd','.exe','.dll','.key','.pem'} or p.name in {'.env','credentials.json','token.json'}:forbidden.append(path)
        if (p.suffix.lower() in TEXT or p.name.startswith('.git')) and b'\0' not in data[:8192]:
            for label,pattern in PATTERNS.items():
                for match in re.finditer(pattern,data):findings.append(dict(path=path,line=data.count(b'\n',0,match.start())+1,pattern=label))
    records=json.loads((ROOT/'handoff/research_records.json').read_text(encoding='utf8'))['records']
    receipts=json.loads((ROOT/'handoff/copied_receipts.json').read_text(encoding='utf8'))['entries']
    seen=set(paths);missing_records=[r['path'] for r in records if r['path'] not in seen];missing_receipts=[r['copy'] for r in receipts if r['copy'] not in seen]
    invalid_receipts=[r['copy'] for r in receipts if sha((ROOT/r['copy']).read_bytes())!=r['sha256']]
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),mode='staged' if a.staged else 'candidate',file_count=len(paths),audited_file_count=len(entries),audited_bytes=sum(e['bytes'] for e in entries),largest=max(entries,key=lambda x:x['bytes']),files_at_least50MiB=oversized,excluded_asset_leaks=sorted(set(forbidden)),credential_pattern_findings=findings,staged_worktree_mismatches=mismatched,symlinks_or_gitlinks=mode_issues,missing_research_records=missing_records,missing_receipt_copies=missing_receipts,invalid_receipt_hashes=invalid_receipts,research_records=len(records),receipt_copies=len(receipts),report_self_excluded_from_hashes=True,limitations='Pattern scan is not a complete secret/security/license audit. Large external research assets are excluded intentionally; this source package alone is not full historical reproduction.',entries=entries)
    (ROOT/REPORT).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='entries'},ensure_ascii=True,indent=2))
    if any([oversized,forbidden,findings,mismatched,mode_issues,missing_records,missing_receipts,invalid_receipts]):raise SystemExit(1)
if __name__=='__main__':main()
