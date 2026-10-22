"""Bounded standard HTTP Range reader for the official public ZIP.

Uses the public download form as served. No credentials, quota workarounds, or
whole-archive fallback. Every range is checked and cached with a byte hash.
"""
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode,urlsplit
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parent/'sources/tgs_pretrained_listing_v1'


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__();self.active=False;self.action=None;self.fields={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':
            self.active=True;self.action=a['action']
        if self.active and tag=='input' and a.get('type')=='hidden':
            self.fields[a['name']]=a['value']
    def handle_endtag(self,tag):
        if tag=='form':self.active=False


class RemoteZip(io.RawIOBase):
    def __init__(self,max_download=96*1024*1024):
        files=json.loads((ROOT/'files.json').read_text())
        assert len(files)==1 and files[0]['name']=='full_eval.zip'
        self.size=files[0]['size'];self.pos=0;self.downloaded=0;self.budget=max_download
        form=DownloadForm();form.feed((ROOT/'range_probe.bin').read_text(encoding='utf8'))
        assert form.fields['id']==files[0]['id'] and urlsplit(form.action).hostname=='drive.usercontent.google.com'
        self.url=form.action+'?'+urlencode(form.fields)
        self.cache=ROOT/'ranges';self.cache.mkdir(exist_ok=True)
        self.request_records=[]
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos if whence==1 else self.size)+offset
        assert 0<=self.pos<=self.size
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if not n:return b''
        if n>16*1024*1024:raise RuntimeError('Refused single remote read above 16 MiB')
        start,end=self.pos,self.pos+n-1
        path=self.cache/f'{start}_{end}.bin'
        receipt_path=path.with_suffix('.json')
        if path.exists() and receipt_path.exists():
            raw=path.read_bytes();receipt=json.loads(receipt_path.read_text())
            assert len(raw)==n and hashlib.sha256(raw).hexdigest()==receipt['sha256']
        else:
            if self.downloaded+n>self.budget:raise RuntimeError('Range download budget exceeded')
            with urlopen(Request(self.url,headers={'User-Agent':'TGS-research-input-audit',
                 'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}),timeout=60) as r:
                content_range=r.headers.get('Content-Range','')
                match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',content_range)
                if r.status!=206 or not match or tuple(map(int,match.groups()))!=(start,end,self.size):
                    small=r.read(65536)
                    (ROOT/'range_rejection.txt').write_text(json.dumps({'status':r.status,
                        'content_range':content_range,'body':small.decode('utf8',errors='replace')},indent=2))
                    raise RuntimeError(f'Server did not honor Range: HTTP {r.status}, {content_range}')
                raw=r.read(n+1)
                assert len(raw)==n
                receipt={'start':start,'end':end,'total':self.size,'status':r.status,
                         'content_range':content_range,'sha256':hashlib.sha256(raw).hexdigest(),
                         'bytes':n,'etag':r.headers.get('ETag'),'last_modified':r.headers.get('Last-Modified'),
                         'retrieved_utc':datetime.now(timezone.utc).isoformat()}
            self.downloaded+=n
            path.write_bytes(raw);receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        self.request_records.append(receipt)
        self.pos+=n
        return raw


def main():
    r=RemoteZip()
    with zipfile.ZipFile(r) as z:
        entries=[{'path':i.filename,'bytes':i.file_size,'compressed':i.compress_size,
                  'method':i.compress_type,'CRC':i.CRC,'offset':i.header_offset,'flags':i.flag_bits}
                 for i in z.infolist()]
    (ROOT/'zip_entries.json').write_text(json.dumps({'files':entries,'ranges':r.request_records,
                'new_download_bytes':r.downloaded},indent=2)+'\n')
    models=[x for x in entries if any(s in x['path'] for s in ['point_cloud_state_dict','hyperparameters.pt','cfg_args','cameras.json'])]
    print(json.dumps({'entries':len(entries),'new_bytes':r.downloaded,'model_metadata':models},indent=2),flush=True)


if __name__=='__main__':
    main()
