"""Metadata only: official page, object identity and ZIP directories before image bytes."""
import hashlib,json,traceback,zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin,urlsplit
from urllib.request import Request,urlopen
from remote_public_zip_v11 import PublicZip,sha

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'sources/tgs_gt_identity_v11';PAGE='https://jonbarron.info/mipnerf360/'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.hrefs=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href')
            if href:self.hrefs.append(urljoin(PAGE,href))
def main():
    OUT.mkdir(exist_ok=True);target=OUT/'catalogue.json';assert not target.exists();html=OUT/'official_project.html'
    if html.exists():raw=html.read_bytes()
    else:
        with urlopen(Request(PAGE,headers={'User-Agent':'TGS-GT-identity-audit'}),timeout=30) as r:raw=r.read(1024*1024+1);assert len(raw)<=1024*1024
        html.write_bytes(raw)
    parser=Links();parser.feed(raw.decode('utf8'));urls=sorted(set(u.replace('http://','https://',1) for u in parser.hrefs if urlsplit(u).hostname=='storage.googleapis.com' and urlsplit(u).path.endswith('.zip')));assert len(urls)==2
    entries=[];downloaded=0
    for i,url in enumerate(urls):
        folder=OUT/f'dataset_{i}';r=PublicZip(url,folder,8*1024*1024-downloaded)
        with zipfile.ZipFile(r) as z:files=[dict(path=x.filename,bytes=x.file_size,compressed=x.compress_size,CRC=x.CRC,method=x.compress_type,offset=x.header_offset,flags=x.flag_bits) for x in z.infolist()]
        listing=folder/'entries.json';assert not listing.exists();listing.write_text(json.dumps(dict(files=files,url=url,head_sha256=sha(folder/'head.json')),indent=2)+'\n');downloaded+=r.downloaded;entries.append(dict(url=url,folder=folder.relative_to(ROOT).as_posix(),listing_sha256=sha(listing),size=r.size,entries=len(files),new_range_bytes=r.downloaded));print(json.dumps(entries[-1]),flush=True)
    plan=dict(official_page=PAGE,page_sha256=sha(html),archives=entries,new_directory_bytes=downloaded,protocol_sha256=sha(ROOT/'TGS_GT_IDENTITY_QUALITY_PROTOCOL_V11.md'),script_sha256=sha(Path(__file__)),reader_sha256=sha(ROOT/'remote_public_zip_v11.py'),image_payloads_fetched=False,training=False);target.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(complete=True,new_directory_bytes=downloaded,image_payloads_fetched=False)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(exist_ok=True);(OUT/'catalogue_failure.txt').write_text(traceback.format_exc());raise
