"""Inspect the public archive link/HTTP range support, with bounded reads."""
import codecs
import hashlib
import html
import json
import re
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parent/'sources/tgs_pretrained_listing_v1'


def capture(url,filename,headers=None,limit=8*1024*1024,method=None):
    path=ROOT/filename
    if path.exists():
        return path.read_bytes()
    with urlopen(Request(url,headers={'User-Agent':'TGS-research-input-audit',**(headers or {})},method=method),timeout=45) as r:
        data=r.read(limit+1)
        if len(data)>limit:
            raise RuntimeError('Bounded HTTP read exceeded; refused whole-archive download')
        record={'url':url,'resolved_url':r.url,'status':r.status,'bytes':len(data),
                'headers':dict(r.headers),'sha256':hashlib.sha256(data).hexdigest(),
                'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    path.write_bytes(data)
    (ROOT/(filename+'.receipt.json')).write_text(json.dumps(record,indent=2)+'\n')
    return data


def main():
    folder=(ROOT/'folder.html').read_text(encoding='utf8')
    m=re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'((?:\\.|[^'])*)'",folder)
    assert m
    # Decode string escaping only, never eval JavaScript.
    content=codecs.decode(m[1],'unicode_escape')
    listing=json.loads(content)
    files=[{'id':x[0],'name':x[2],'mime':x[3],'size':x[13]} for x in listing[0]]
    (ROOT/'files.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps(files),flush=True)
    assert len(files)==1 and files[0]['name']=='full_eval.zip'
    fid=files[0]['id']
    page=capture(f'https://drive.google.com/file/d/{fid}/view','archive_view.html').decode('utf8')
    matches=[]
    for p in [r'https://drive\.usercontent\.google\.com/[^"<>\s]+',r'https://drive\.google\.com/uc[^"<>\s]+',r'downloadUrl.{0,500}']:
        matches.extend(re.findall(p,page))
    print(json.dumps({'view_download_metadata':matches[:10]},ensure_ascii=True),flush=True)
    # Official public download endpoint, byte range only. Hard cap avoids a
    # server ignoring Range turning into an unbounded model download.
    url=f'https://drive.google.com/uc?export=download&id={fid}'
    payload=capture(url,'range_probe.bin',headers={'Range':'bytes=0-0'},limit=512*1024)
    receipt=json.loads((ROOT/'range_probe.bin.receipt.json').read_text())
    print(json.dumps({'probe_status':receipt['status'],'bytes':len(payload),
                      'content_type':receipt['headers'].get('Content-Type'),
                      'content_range':receipt['headers'].get('Content-Range'),
                      'resolved_url':receipt['resolved_url']}),flush=True)
    if 'html' in receipt['headers'].get('Content-Type',''):
        print(payload[:4500].decode('utf8',errors='replace'),flush=True)


if __name__=='__main__':
    main()
