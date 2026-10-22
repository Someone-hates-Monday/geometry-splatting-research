"""Archive only primary public reports used by this bounded narrative review."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parent/'sources/tgs_topology_review_v1'
SOURCES=[('stamm_eidenbenz_pajarola_1998.pdf','https://www.ifi.uzh.ch/vmml/publications/older-puclications/CCCG98.pdf'),
         ('ets_2608_29106v1.html','https://arxiv.org/html/2608.29106v1'),
         ('meshsplatting_2512_06818v1.html','https://arxiv.org/html/2512.06818v1'),
         ('cf_splat_2610_22108v1.html','https://arxiv.org/html/2610.22108v1')]


def main():
    ROOT.mkdir(parents=True,exist_ok=True);records=[]
    for name,url in SOURCES:
        target=ROOT/name
        if target.exists():
            record=json.loads((ROOT/(name+'.receipt.json')).read_text())
            assert hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256']
            records.append(record);continue
        try:
            with urlopen(Request(url,headers={'User-Agent':'TGS-bounded-literature-audit'}),timeout=35) as r:
                data=r.read(8*1024*1024+1);assert len(data)<=8*1024*1024
                if name.endswith('.pdf'):assert data.startswith(b'%PDF')
                record={'name':name,'url':url,'resolved_url':r.url,'bytes':len(data),
                        'sha256':hashlib.sha256(data).hexdigest(),
                        'retrieved_utc':datetime.now(timezone.utc).isoformat(),'status':r.status}
            target.write_bytes(data)
            (ROOT/(name+'.receipt.json')).write_text(json.dumps(record,indent=2)+'\n');records.append(record)
            print(json.dumps({'saved':name,'bytes':len(data)}),flush=True)
        except Exception as e:
            records.append({'name':name,'url':url,'archive_failed':str(e)})
            print(json.dumps({'failed':name,'error':str(e)}),flush=True)
    (ROOT/'manifest.json').write_text(json.dumps({'scope':'selected primary reports, not exhaustive prior-art review','sources':records},indent=2)+'\n')


if __name__=='__main__':main()
