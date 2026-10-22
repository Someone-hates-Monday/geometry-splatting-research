"""Fetch pinned source only; never pip install or modify another extension."""
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
STATIC = ROOT/'sources/tgs_subdivision_static_v1/diff-triangle-rasterization'
OUT = ROOT/'sources/tgs_native_diagnostic_v1'
PIN = '6d61f4c0a571c635e37ffb740250706fa30a5541'
GLM = '5c46b9c07008ae65cb81ab79cd677ecc1934b903'


def fetch(url):
    with urlopen(Request(url,headers={'User-Agent':'TGS-local-research-diagnostic'}),timeout=45) as r:
        return r.read()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, str(path)
    else:
        path.write_bytes(data)


def main():
    records = []
    tree = json.loads((STATIC/'metadata/tree.json').read_text())
    assert tree['sha'] == 'c6d587a65400f320f303d9e1e69487ee02748217' and not tree['truncated']
    for e in tree['tree']:
        if e['type'] != 'blob':
            continue
        path = e['path']
        url = f'https://raw.githubusercontent.com/trianglesplatting/diff-triangle-rasterization/{PIN}/{path}'
        target = OUT/'native'/path
        cached = STATIC/path
        data = target.read_bytes() if target.exists() else (cached.read_bytes() if cached.exists() else fetch(url))
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        assert blob == e['sha'] and len(data) == e['size'], path
        save(target,data)
        records.append({'path':'native/'+path,'git_blob_sha1':blob,'sha256':hashlib.sha256(data).hexdigest()})
        print('verified '+path,flush=True)
    # The native gitlink pins glm. Check downloaded headers against glm's Git tree.
    glm_tree_file = OUT/'glm_tree.json'
    gt = glm_tree_file.read_bytes() if glm_tree_file.exists() else fetch(f'https://api.github.com/repos/g-truc/glm/git/trees/{GLM}?recursive=1')
    save(glm_tree_file,gt)
    glm_tree = json.loads(gt)
    assert not glm_tree['truncated']
    entries = {e['path']:e for e in glm_tree['tree'] if e['type']=='blob'}
    archive = OUT/'glm.zip'
    data = archive.read_bytes() if archive.exists() else fetch(f'https://codeload.github.com/g-truc/glm/zip/{GLM}')
    save(archive,data)
    count = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            rel = item.filename.split('/',1)[1]
            if not (rel.startswith('glm/') or rel == 'copying.txt'):
                continue
            assert '..' not in Path(rel).parts and not Path(rel).is_absolute()
            raw = z.read(item)
            blob = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
            assert blob == entries[rel]['sha'], rel
            save(OUT/'native/third_party/glm'/rel,raw)
            records.append({'path':'native/third_party/glm/'+rel,'git_blob_sha1':blob,'sha256':hashlib.sha256(raw).hexdigest()})
            count += 1
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),'native_commit':PIN,
                'glm_commit':GLM,'files':records,'glm_files':count,
                'scope':'native source plus pinned glm headers; no installation'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'complete':True,'native_files':len(records)-count,'glm_files':count}),flush=True)


if __name__ == '__main__':
    main()
