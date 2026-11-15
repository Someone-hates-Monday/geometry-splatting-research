"""Python3 stdlib receiver. No package installation, CUDA compilation or training."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path,PurePosixPath

REPO='Someone-hates-Monday/geometry-splatting-research';TAG='research-assets-2026-11-15'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def target(root,name):
    q=PurePosixPath(name)
    if not name or q.is_absolute() or '\\' in name or ':' in name or any(x in {'..','.'} for x in name.split('/')):raise ValueError('Unsafe path: '+name)
    p=root.joinpath(*q.parts)
    if not p.resolve().is_relative_to(root.resolve()):raise ValueError('Path escapes root: '+name)
    for x in [p,*p.parents]:
        if x.is_symlink() or (hasattr(x,'is_junction') and x.is_junction()):raise ValueError('Link/junction in destination: '+str(x))
        if x==root:break
    return p
def valid(p,size,sha):return p.is_file() and p.stat().st_size==size and digest(p)==sha
def write_verified(root,name,source,size,sha):
    p=target(root,name)
    if p.exists():
        if valid(p,size,sha):return
        raise ValueError('Existing file differs; refusing overwrite: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as f:
            tmp=Path(f.name);shutil.copyfileobj(source,f,8*1024**2)
        if not valid(tmp,size,sha):raise ValueError('Size/SHA256 mismatch: '+name)
        if p.exists():raise ValueError('Destination appeared during restore: '+str(p))
        tmp.rename(p);tmp=None
    finally:
        if tmp is not None:tmp.unlink(missing_ok=True)
def restore(manifest,cache,root):
    root=root.absolute();root.mkdir(parents=True,exist_ok=True)
    entries={e['archive_path']:e for e in manifest['entries']}
    if len(entries)!=len(manifest['entries']):raise ValueError('Duplicate archive paths')
    assets=manifest['assets'];seen=set()
    if len({a['name'] for a in assets})!=len(assets):raise ValueError('Duplicate volumes')
    # Validate all archives and target paths before writing any repository data.
    for e in entries.values():target(root,e['archive_path']);target(root,e['path'])
    for a in assets:
        p=target(cache,a['name'])
        if not valid(p,a['bytes'],a['sha256']):raise ValueError('Missing or corrupt volume: '+str(p))
        with zipfile.ZipFile(p) as z:
            for info in z.infolist():
                name=info.filename
                if name not in entries or name in seen or info.is_dir() or ((info.external_attr>>16)&0o170000)==0o120000:raise ValueError('Unexpected/duplicate/link ZIP member: '+name)
                e=entries[name]
                if info.file_size!=e.get('length',e['bytes']):raise ValueError('Member length mismatch: '+name)
                seen.add(name)
    if seen!=set(entries):raise ValueError('Incomplete archive member set')
    for a in assets:
        with zipfile.ZipFile(target(cache,a['name'])) as z:
            for info in z.infolist():
                e=entries[info.filename]
                with z.open(info) as inp:write_verified(root,info.filename,inp,e.get('length',e['bytes']),e['sha256'])
        print('RESTORED '+a['name'],flush=True)
    groups={}
    for e in entries.values():
        if 'offset' in e:groups.setdefault(e['path'],[]).append(e)
    assembled=[]
    for name,parts in groups.items():
        parts.sort(key=lambda x:x['offset']);offset=0
        for e in parts:
            if e['offset']!=offset or e['bytes']!=parts[0]['bytes']:raise ValueError('Noncontiguous split file: '+name)
            offset+=e['length']
        if offset!=parts[0]['bytes']:raise ValueError('Incomplete split file: '+name)
        p=target(root,name);p.parent.mkdir(parents=True,exist_ok=True)
        # Hash concatenation first, so existing complete files can be verified safely.
        h=hashlib.sha256()
        for e in parts:
            with target(root,e['archive_path']).open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        if p.exists():
            if not valid(p,offset,h.hexdigest()):raise ValueError('Existing assembled file differs: '+name)
        else:
            tmp=None
            try:
                with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as f:
                    tmp=Path(f.name)
                    for e in parts:
                        with target(root,e['archive_path']).open('rb') as inp:shutil.copyfileobj(inp,f,8*1024**2)
                if not valid(tmp,offset,h.hexdigest()):raise ValueError('Assembled file mismatch: '+name)
                if p.exists():raise ValueError('Destination appeared: '+name)
                tmp.rename(p);tmp=None
            finally:
                if tmp is not None:tmp.unlink(missing_ok=True)
        assembled.append(dict(path=name,bytes=offset,sha256=h.hexdigest()))
    return dict(volumes=len(assets),entries=len(entries),assembled=assembled,skipped_source_paths=manifest.get('skipped',[]))
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a.add_argument('--cache',type=Path,default=Path.home()/'geometry-splatting-assets')
    a.add_argument('--repo',default=REPO);a.add_argument('--tag',default=TAG);a.add_argument('--manifest',type=Path)
    a.add_argument('--download',action='store_true');a.add_argument('--restore',action='store_true');a.add_argument('--check-env',action='store_true');a.add_argument('--wait',action='store_true')
    args=a.parse_args();report=dict(environment={},assets=None)
    if not any([args.download,args.restore,args.check_env]):a.print_help();return
    if args.check_env:
        report['environment']=dict(python=sys.version,platform=sys.platform,tools={x:shutil.which(x) for x in ['git','gh','nvidia-smi','nvcc','gcc','g++','conda']})
        # Do not import Torch or execute checkpoint payloads as part of asset deployment.
        print(json.dumps(report['environment'],indent=2))
    if args.download or args.restore:
        args.cache=args.cache.absolute();args.cache.mkdir(parents=True,exist_ok=True)
        manifest=args.manifest or target(args.cache,'ASSET_MANIFEST.json')
        if args.download:
            if not args.manifest:
                while True:
                    r=subprocess.run(['gh','release','download',args.tag,'--repo',args.repo,'--pattern','ASSET_MANIFEST.json','--dir',str(args.cache),'--clobber'],capture_output=True,text=True)
                    if r.returncode==0:break
                    print(r.stderr.strip(),file=sys.stderr)
                    if not args.wait:raise RuntimeError('Final manifest unavailable. Upload may be incomplete; check authentication and Release status. No data deployed.')
                    time.sleep(60)
            m=json.loads(manifest.read_text(encoding='utf8'))
            for item in m['assets']:
                p=target(args.cache,item['name'])
                if valid(p,item['bytes'],item['sha256']):continue
                if p.exists():raise ValueError('Corrupt cache file; move aside before retry: '+str(p))
                for attempt in range(3):
                    subprocess.run(['gh','release','download',args.tag,'--repo',args.repo,'--pattern',item['name'],'--dir',str(args.cache),'--clobber'],check=True)
                    if valid(p,item['bytes'],item['sha256']):break
                else:raise ValueError('Downloaded volume hash mismatch: '+item['name'])
        if args.restore:
            report['assets']=restore(json.loads(manifest.read_text(encoding='utf8')),args.cache,args.root)
            report['assets']['manifest_sha256']=digest(manifest)
            report['assets']['repo']=args.repo;report['assets']['tag']=args.tag
    args.cache.mkdir(parents=True,exist_ok=True)
    (args.cache/'deployment_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Deployment check complete. No training or package installation was performed.')
if __name__=='__main__':main()
