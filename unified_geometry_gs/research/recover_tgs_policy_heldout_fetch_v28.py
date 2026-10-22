"""One explicit continuation after terminal TLS failure; same frozen image members."""
import hashlib,json,traceback,zipfile
from pathlib import Path
from prepare_tgs_policy_data_v28 import ROOT,OUT,SOURCE,sha,write,cumulative
from remote_tgs_zip import RemoteZip,ROOT as LISTING
from extract_tgs_gt_identity_v11 import extract_member
class Reuse(RemoteZip):
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if not n:return b''
        assert n<=16*1024*1024
        start,end=self.pos,self.pos+n-1
        for folder in [LISTING/'ranges',ROOT/'sources/tgs_gt_identity_v11/tgs_ranges',ROOT/'sources/tgs_sh_isolation_v12/tgs_ranges']:
            p=folder/f'{start}_{end}.bin';rp=p.with_suffix('.json')
            if p.exists() and rp.exists():
                q=json.loads(rp.read_text());raw=p.read_bytes();assert len(raw)==n and hashlib.sha256(raw).hexdigest()==q['sha256'] and q['total']==self.size
                self.borrowed.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),receipt=rp.relative_to(ROOT).as_posix(),receipt_sha256=sha(rp),bytes=n));self.request_records.append(q);self.pos+=n;return raw
        return super().read(n)
def main():
    target=SOURCE/'heldout_manifest.json';assert not target.exists();failure=OUT/'fetch_heldout_failure.txt';assert failure.exists()
    assert 'SSL: UNEXPECTED_EOF_WHILE_READING' in failure.read_text();plan=json.loads((OUT/'input_plan.json').read_text());selected=[r for r in plan['rows'] if r['cohort']=='heldout'];records=[];deps=[Path(__file__),OUT/'input_plan.json',OUT/'choices.json',failure]
    for row in selected:
        p=SOURCE/row['scene']/row['view']/'original.JPG';rp=p.with_suffix('.JPG.receipt.json');q=json.loads(rp.read_text());assert sha(p)==q['sha256'] and q['archive_member']==row['original']['path'];records.append(dict(q,view=row['view']));deps.extend([p,rp])
    write(OUT/'heldout_fetch_recovery_plan.json',dict(inputs={p.relative_to(ROOT).as_posix():sha(p) for p in deps},same_four_preselected_test_views=True,replace_or_resample=False,retry_attempts=1,borrow_verified_cached_ranges=True,maximum_total_new_cached_payload=plan['new_payload_budget'],test_scoring_started=False,original_fetch_exit=1,training=False))
    r=Reuse(max_download=plan['new_payload_budget']-cumulative());r.cache=SOURCE/'tgs'/'ranges';r.cache.mkdir(parents=True,exist_ok=True);r.borrowed=[]
    with zipfile.ZipFile(r) as z:
        for row in selected:
            q=extract_member(z,row['published_gt'],SOURCE/row['scene']/row['view']/'published_gt.png',row['scene'],'published_gt');records.append(dict(q,view=row['view']))
    assert len(records)==8 and cumulative()<=plan['new_payload_budget']
    write(target,dict(files=records,input_plan_sha256=sha(OUT/'input_plan.json'),choices_sha256=sha(OUT/'choices.json'),total_range_bytes=cumulative(),borrowed_readonly_ranges=r.borrowed,recovery_plan_sha256=sha(OUT/'heldout_fetch_recovery_plan.json'),original_fetch_failed_TLS=True,training=False))
    print(json.dumps(dict(stage='HELDOUT_FIXED_MEMBERS_RECOVERED',range_bytes=cumulative(),borrowed_ranges=len(r.borrowed))),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        with (OUT/'heldout_fetch_recovery_failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
