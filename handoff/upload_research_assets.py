"""Resumable private GitHub Release transfer. Never removes original assets."""
import concurrent.futures,hashlib,json,os,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.asset-transfer';OUT.mkdir(exist_ok=True)
REPO='Someone-hates-Monday/geometry-splatting-research';TAG='research-assets-2026-11-15'
LIMIT=1500*1024**2
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def gh(*args):return subprocess.check_output(['gh',*args],stderr=subprocess.STDOUT).decode('utf8',errors='replace')
def save(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
def remote():
    r=json.loads(gh('api',f'repos/{REPO}/releases/tags/{TAG}'))
    assets=json.loads(gh('api',f'repos/{REPO}/releases/{r["id"]}/assets?per_page=100','--paginate','--slurp'))
    return {x['name']:x for page in assets for x in page}
def upload(p):
    size=p.stat().st_size;digest=sha(p);name=p.name
    for attempt in range(4):
        try:
            existing=remote().get(name)
            if not existing:
                gh('release','upload',TAG,str(p),'--repo',REPO)
                existing=remote().get(name)
            if existing and existing.get('state')!='uploaded':
                gh('release','upload',TAG,str(p),'--repo',REPO,'--clobber');existing=remote().get(name)
            assert existing and existing['size']==size,(name,'remote size mismatch')
            assert existing.get('digest')=='sha256:'+digest,(name,'remote SHA256 mismatch',existing.get('digest'))
            receipt=dict(name=name,bytes=size,sha256=digest,url=existing['browser_download_url'],asset_id=existing['id'])
            save(OUT/(name+'.receipt.json'),receipt)
            print(json.dumps(dict(uploaded=name,bytes=size),ensure_ascii=True),flush=True)
            p.unlink() # Only generated transfer archive, never source files.
            return receipt
        except Exception as e:
            print(json.dumps(dict(retry=name,attempt=attempt+1,error=str(e)[:600])),flush=True)
            if attempt==3:raise
            time.sleep(5)
def main():
    gh('repo','view',REPO,'--json','visibility')
    try:gh('release','view',TAG,'--repo',REPO)
    except subprocess.CalledProcessError:
        gh('release','create',TAG,'--repo',REPO,'--target','main','--title','Research data, trained models and raw evidence (2026-11-15)','--notes','Private research asset transfer. See ASSET_MANIFEST.json for paths, hashes, volumes and skipped links. Download all listed volumes before restoration. Original source assets remain unchanged.','--latest=false')
    planfile=OUT/'plan.json'
    if planfile.exists():plan=json.loads(planfile.read_text(encoding='utf8'))
    else:
        groups=['unified_geometry_gs/research/results','unified_geometry_gs/output_compare','dtu','unified_geometry_gs/dtu_eval','unified_geometry_gs/research/experiments','unified_geometry_gs/research/sources','unified_geometry_gs/research/snapshots']
        files=[];skipped=[]
        for base in groups:
            def err(e):skipped.append(dict(path=str(e.filename),reason=str(e)))
            for folder,dirs,names in os.walk(ROOT/base,followlinks=False,onerror=err):
                dirs[:]=sorted(d for d in dirs if d not in {'.git','__pycache__','node_modules'} and not (Path(folder)/d).is_symlink())
                for n in sorted(names):
                    p=Path(folder)/n
                    try:
                        if p.is_symlink():skipped.append(dict(path=p.relative_to(ROOT).as_posix(),reason='symlink',target=os.readlink(p)));continue
                        if p.suffix.lower() in {'.pem','.key'} or n in {'.env','credentials.json','token.json'}:skipped.append(dict(path=p.relative_to(ROOT).as_posix(),reason='credential filename'));continue
                        s=p.stat();files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
                    except OSError as e:skipped.append(dict(path=str(p),reason=str(e)))
        batches=[];batch=[];total=0
        for f in files:
            if f['bytes']>LIMIT:
                if batch:batches.append(batch);batch=[];total=0
                for start in range(0,f['bytes'],LIMIT):batches.append([dict(**f,offset=start,length=min(LIMIT,f['bytes']-start))])
            else:
                if total+f['bytes']>LIMIT and batch:batches.append(batch);batch=[];total=0
                batch.append(f);total+=f['bytes']
        if batch:batches.append(batch)
        plan=dict(repo=REPO,tag=TAG,files=len(files),bytes=sum(f['bytes'] for f in files),skipped=skipped,batches=batches)
        save(planfile,plan)
    print(json.dumps(dict(files=plan['files'],bytes=plan['bytes'],volumes=len(plan['batches']),skipped=len(plan['skipped']))),flush=True)
    futures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for i,batch in enumerate(plan['batches']):
            name=f'research-assets-{i:04d}.zip';p=OUT/name;catalog=OUT/(name+'.files.json')
            if (OUT/(name+'.receipt.json')).exists():continue
            if not p.exists():
                entries=[];temp=p.with_suffix('.partial')
                with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
                    for f in batch:
                        src=ROOT/f['path'];s=src.stat();assert s.st_size==f['bytes'] and s.st_mtime_ns==f['mtime_ns'],f['path']
                        archive=f['path']+(f'.asset-part-{f["offset"]:012d}' if 'offset' in f else '')
                        h=hashlib.sha256();remaining=f.get('length',f['bytes'])
                        with src.open('rb') as inp,z.open(archive,'w',force_zip64=True) as dst:
                            inp.seek(f.get('offset',0))
                            while remaining:
                                b=inp.read(min(8*1024**2,remaining));assert b,'unexpected EOF';h.update(b);dst.write(b);remaining-=len(b)
                        entries.append(dict(**f,archive_path=archive,sha256=h.hexdigest()))
                save(catalog,entries);temp.replace(p)
            futures.append(pool.submit(upload,p))
            while len([f for f in futures if not f.done()])>=3:
                concurrent.futures.wait([f for f in futures if not f.done()],timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in futures:
                    if f.done():f.result()
        for f in futures:f.result()
    receipts=[];entries=[]
    for i in range(len(plan['batches'])):
        name=f'research-assets-{i:04d}.zip';receipts.append(json.loads((OUT/(name+'.receipt.json')).read_text(encoding='utf8')))
        entries.extend(json.loads((OUT/(name+'.files.json')).read_text(encoding='utf8')))
    manifest=dict(repo=REPO,tag=TAG,total_source_files=plan['files'],source_bytes=plan['bytes'],skipped=plan['skipped'],assets=receipts,entries=entries,restore='Download and verify all ZIP hashes; extract retaining repository-relative paths. Large source files use .asset-part-OFFSET suffixes: concatenate in numeric offset order, validating each part SHA256 first. Never overwrite existing research evidence without checking hashes.')
    actual=remote()
    for r in receipts:
        a=actual[r['name']];assert a['size']==r['bytes'] and a.get('digest')=='sha256:'+r['sha256'],r['name']
    dest=OUT/'ASSET_MANIFEST.json';save(dest,manifest)
    gh('release','upload',TAG,str(dest),'--repo',REPO,'--clobber')
    save(OUT/'COMPLETE.json',dict(verified_assets=len(receipts),source_files=plan['files'],source_bytes=plan['bytes'],manifest_sha256=sha(dest)))
    print('TRANSFER_COMPLETE',flush=True)
if __name__=='__main__':main()
