"""Read the official public pretrained-folder listing, without downloading models."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parent/'sources/tgs_pretrained_listing_v1'
URL='https://drive.google.com/drive/folders/1YrH9IVU8QWgfnIg_i0iRHq_M9rNCNPyV?usp=sharing'


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'folder.html'
    if path.exists():
        raise RuntimeError('Listing already captured; inspect saved bytes rather than overwrite.')
    with urlopen(Request(URL,headers={'User-Agent':'TGS-research-input-audit'}),timeout=45) as r:
        data=r.read(8*1024*1024+1)
        assert len(data)<=8*1024*1024
        receipt={'url':URL,'resolved_url':r.url,'status':r.status,
                 'content_type':r.headers.get('Content-Type'),'bytes':len(data),
                 'sha256':hashlib.sha256(data).hexdigest(),
                 'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                 'source':'pinned official triangle-splatting README trained MipNeRF360 link',
                 'scope':'public folder HTML only; no model files downloaded'}
    path.write_bytes(data)
    (ROOT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
