"""Immutable-identity public GCS Range reader, small bounded cache; no whole ZIP fetch."""
import hashlib,io,json,re
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request,urlopen

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class PublicZip(io.RawIOBase):
    def __init__(self,url,folder,max_new_bytes):
        assert urlsplit(url).scheme=='https' and urlsplit(url).hostname=='storage.googleapis.com';self.url=url;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.pos=0;self.downloaded=0;self.budget=max_new_bytes;head=self.folder/'head.json'
        if head.exists():self.identity=json.loads(head.read_text());assert self.identity['url']==url
        else:
            with urlopen(Request(url,method='HEAD',headers={'User-Agent':'TGS-GT-identity-audit','Accept-Encoding':'identity'}),timeout=30) as r:
                assert r.status==200 and urlsplit(r.geturl()).hostname=='storage.googleapis.com';self.identity=dict(url=url,size=int(r.headers['Content-Length']),etag=r.headers.get('ETag'),last_modified=r.headers.get('Last-Modified'),google_hash=r.headers.get('x-goog-hash'),created_utc=datetime.now(timezone.utc).isoformat())
            head.write_text(json.dumps(self.identity,indent=2)+'\n')
        self.size=self.identity['size'];assert self.size>0;self.cache=self.folder/'ranges';self.cache.mkdir(exist_ok=True);self.requests=[]
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos if whence==1 else self.size)+offset;assert 0<=self.pos<=self.size;return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if not n:return b''
        assert n<=16*1024*1024,'Refuse unbounded/whole archive reads';start,end=self.pos,self.pos+n-1;p=self.cache/f'{start}_{end}.bin';receipt=p.with_suffix('.json')
        if p.exists() and receipt.exists():
            raw=p.read_bytes();q=json.loads(receipt.read_text());assert len(raw)==n and hashlib.sha256(raw).hexdigest()==q['sha256'] and q['total']==self.size
        else:
            assert not p.exists() and self.downloaded+n<=self.budget;headers={'User-Agent':'TGS-GT-identity-audit','Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
            if self.identity['etag']:headers['If-Match']=self.identity['etag']
            with urlopen(Request(self.url,headers=headers),timeout=30) as r:
                cr=r.headers.get('Content-Range','');m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
                assert r.status==206 and m and tuple(map(int,m.groups()))==(start,end,self.size),'Server must honor bounded Range'
                assert urlsplit(r.geturl()).hostname=='storage.googleapis.com'
                if self.identity['etag']:assert r.headers.get('ETag')==self.identity['etag']
                raw=r.read(n+1);assert len(raw)==n;q=dict(start=start,end=end,total=self.size,bytes=n,status=206,content_range=cr,sha256=hashlib.sha256(raw).hexdigest(),etag=r.headers.get('ETag'),created_utc=datetime.now(timezone.utc).isoformat())
            p.write_bytes(raw);receipt.write_text(json.dumps(q,indent=2)+'\n');self.downloaded+=n
        self.requests.append(q);self.pos+=n;return raw
