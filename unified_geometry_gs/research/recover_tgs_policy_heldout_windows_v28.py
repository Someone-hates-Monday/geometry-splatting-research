"""Explicit Windows TLS continuation, four bounded pinned local-member ranges only."""
import hashlib,json,re,struct,traceback,zlib
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
from prepare_tgs_policy_data_v28 import ROOT,OUT,SOURCE,sha,write,cumulative
from remote_tgs_zip import RemoteZip
def main():
    assert not (SOURCE/'heldout_manifest.json').exists();failed=OUT/'heldout_fetch_recovery_failure.txt';assert failed.exists() and 'timed out' in failed.read_text()
    plan=json.loads((OUT/'input_plan.json').read_text());selected=[r for r in plan['rows'] if r['cohort']=='heldout'];deps=[Path(__file__),OUT/'input_plan.json',OUT/'choices.json',failed,OUT/'heldout_fetch_recovery_plan.json'];records=[]
    for row in selected:
        p=SOURCE/row['scene']/row['view']/'original.JPG';rp=p.with_suffix('.JPG.receipt.json');q=json.loads(rp.read_text());assert sha(p)==q['sha256'];records.append(dict(q,view=row['view']));deps.extend([p,rp])
    write(OUT/'heldout_windows_recovery_plan.json',dict(inputs={p.relative_to(ROOT).as_posix():sha(p) for p in deps},same_member_selection=True,attempts_per_member=1,timeout_seconds=30,maximum_total_range_bytes=96*1024*1024,whole_archive=False,TLS_verification_enabled=True,original_WSL_attempts_failed=True,windows_native_TLS=True,quality_outcomes_unseen=True,training=False))
    remote=RemoteZip();folder=SOURCE/'windows_gt'/'ranges';folder.mkdir(parents=True)
    for row in selected:
        en=row['published_gt'];start=en['offset'];n=30+len(en['path'].encode('utf8'))+65535+en['compressed'];end=start+n-1
        assert n<=16*1024*1024 and cumulative()+n<=plan['new_payload_budget']
        with urlopen(Request(remote.url,headers={'User-Agent':'TGS-frozen-holdout-audit','Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}),timeout=30) as response:
            cr=response.headers.get('Content-Range','');m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr);assert response.status==206 and m and tuple(map(int,m.groups()))==(start,end,remote.size)
            host=urlsplit(response.geturl()).hostname;assert host=='drive.usercontent.google.com' or host.endswith('.googleusercontent.com')
            raw=response.read(n+1);assert len(raw)==n;etag=response.headers.get('ETag')
        bp=folder/f'{start}_{end}.bin';bp.write_bytes(raw);write(bp.with_suffix('.json'),dict(start=start,end=end,total=remote.size,bytes=n,status=206,content_range=cr,sha256=sha(bp),etag=etag,created_utc=datetime.now(timezone.utc).isoformat(),windows_TLS=True))
        header=struct.unpack('<IHHHHHIIIHH',raw[:30]);sig,ver,flags,method,tm,dt,crc,csize,usize,fnlen,extra=header
        assert sig==0x04034b50 and method==en['method']==8 and not(flags&1) and flags==en['flags'];assert raw[30:30+fnlen].decode('utf8')==en['path']
        if not(flags&8):assert (crc,csize,usize)==(en['CRC'],en['compressed'],en['bytes'])
        begin=30+fnlen+extra;compressed=raw[begin:begin+en['compressed']];assert len(compressed)==en['compressed'];image=zlib.decompress(compressed,-15)
        assert len(image)==en['bytes'] and zlib.crc32(image)==en['CRC'];dest=SOURCE/row['scene']/row['view']/'published_gt.png';assert not dest.exists();dest.write_bytes(image)
        q=dict(scene=row['scene'],kind='published_gt',archive_member=en['path'],local=dest.relative_to(ROOT).as_posix(),bytes=len(image),sha256=sha(dest),CRC32=en['CRC'],zip_crc_checked=True,local_header_and_compression_verified=True,range_receipt=bp.with_suffix('.json').relative_to(ROOT).as_posix())
        write(dest.with_suffix('.png.receipt.json'),q);records.append(dict(q,view=row['view']));print(json.dumps(dict(stage='WINDOWS_PINNED_GT_RECEIVED',scene=row['scene'],view=row['view'],bytes=len(image))),flush=True)
    write(SOURCE/'heldout_manifest.json',dict(files=records,input_plan_sha256=sha(OUT/'input_plan.json'),choices_sha256=sha(OUT/'choices.json'),total_range_bytes=cumulative(),recovery_plan_sha256=sha(OUT/'heldout_windows_recovery_plan.json'),two_WSL_fetch_failures_preserved=True,windows_native_TLS=True,training=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        with (OUT/'heldout_windows_recovery_failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
