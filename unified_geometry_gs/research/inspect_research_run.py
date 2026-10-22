"""Read-only WSL process/artifact snapshot; never restart or kill a job."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import time
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tail(p,n=2500):
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-n));return f.read().decode('utf-8',errors='replace').replace('\r','\n').splitlines()[-3:]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path);args=p.parse_args()
    run=args.run.resolve();manifest=run/'manifest.json';state=json.loads(manifest.read_text());processes=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if not argv or not any('python' in x for x in argv[:1]):continue
            matched=False
            for option in ('--out','-m','--model'):
                if option in argv:
                    target=Path(argv[argv.index(option)+1])
                    if not target.is_absolute():target=(proc/'cwd').resolve()/target
                    target=target.resolve()
                    if target==run or run in target.parents:matched=True
            if not matched:continue
            info={k:v.strip() for k,v in (x.split(':',1) for x in (proc/'status').read_text().splitlines() if ':' in x)}
            processes.append({'pid':int(proc.name),'ppid':int(info['PPid']),'state':info['State'],'argv':argv})
        except (FileNotFoundError,PermissionError,ProcessLookupError,IndexError):continue
    logs={}
    for f in run.glob('*.log'):
        logs[f.name]={'bytes':f.stat().st_size,'seconds_since_write':time.time()-f.stat().st_mtime,'tail':tail(f)}
    outputs={}
    for f in run.glob('*/*training_integration.json'):
        x=json.loads(f.read_text());outputs[str(f.relative_to(run))]={'status':x.get('status'),'completed':x.get('completed'),
            'steps_observed':x.get('steps_observed'),'sha256':sha(f)}
    alive=[x for x in processes if not x['state'].startswith('Z')]
    stopped=[x for x in alive if x['state'].startswith(('T','t'))]
    label='terminal_manifest' if state['status'] in ('complete','failed','superseded_before_candidate') else ('stopped_process_requires_review' if stopped else ('live_process' if alive else 'running_manifest_without_live_process'))
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'manifest_status':state['status'],
        'current_stage':state.get('current_stage'),'manifest_age_seconds':time.time()-manifest.stat().st_mtime,
        'assessment':label,'processes':processes,'logs':logs,'integration_outputs':outputs,
        'limits':['Read-only snapshot, not a proof that a live process is making progress.',
            'Training stage manifest changes at stage boundaries; use log write times and integration artifacts together.',
            'No restart, kill, wall-time or quality inference is performed.']}
    if args.out:
        if args.out.exists():raise ValueError('Output path must be fresh.')
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
