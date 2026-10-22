"""Frozen camera identities, existing calibration references, bounded new GT pairs."""
import argparse,hashlib,json,re,struct,traceback,zipfile,zlib
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
import numpy as np
from PIL import Image
from prepare_tgs_policy_data_v28 import ROOT,sha,write
from remote_public_zip_v11 import PublicZip
from remote_tgs_zip import RemoteZip,ROOT as LISTING
from extract_tgs_gt_identity_v11 import extract_member
OLD=ROOT/'experiments/tgs_policy_transfer_v28';OUT=ROOT/'experiments/tgs_joint_policy_v29';SOURCE=ROOT/'sources/tgs_joint_policy_v29'
def count():return sum(json.loads(p.read_text())['bytes'] for p in SOURCE.glob('**/ranges/*.json'))
def prepare():
    OUT.mkdir();SOURCE.mkdir();prior=json.loads((OLD/'input_plan.json').read_text());catalog=json.loads((ROOT/'sources/tgs_gt_identity_v11/catalogue.json').read_text());tgs=json.loads((LISTING/'zip_entries.json').read_text())['files'];rows=[]
    files=[Path(__file__),ROOT/'TGS_JOINT_POLICY_PROTOCOL_V29.md',OLD/'input_plan.json',OLD/'evidence_manifest.json',ROOT/'experiments/tgs_sh_isolation_v12/input_plan.json',ROOT/'experiments/tgs_cross_scene_v10/input_audit.json',LISTING/'zip_entries.json',ROOT/'sources/tgs_gt_identity_v11/catalogue.json',ROOT/'remote_public_zip_v11.py',ROOT/'remote_tgs_zip.py',ROOT/'extract_tgs_gt_identity_v11.py']
    for scene in ['flowers','room']:
        for r in [r for r in prior['rows'] if r['scene']==scene and r['cohort']=='calibration']:
            gt=OLD/scene/r['view']/'gt_rgb.npy';files.append(gt);rows.append(dict(r,GT=gt.relative_to(ROOT).as_posix(),calibration_reused=True))
        cameras_path=ROOT/'sources/tgs_cross_scene_v10'/scene/'cameras.json';files.append(cameras_path);cams=sorted(json.loads(cameras_path.read_text()),key=lambda c:c['img_name'])[::8]
        used={r['camera']['img_name'] for r in prior['rows'] if r['scene']==scene};used.update(n for r in prior['rows'] if r['scene']==scene for n in r['excluded_prior_camera_names'])
        chosen=sorted([c for c in cams if c['img_name'] not in used],key=lambda c:(hashlib.sha256(f"tgs_joint_policy_v29/{scene}/{c['img_name']}".encode()).hexdigest(),c['img_name']))[:2];assert len(chosen)==2
        for vi,c in enumerate(chosen):
            index=[x['img_name'] for x in cams].index(c['img_name']);wanted=f"{scene}/{'images_4' if scene=='flowers' else 'images_2'}/{c['img_name']}.JPG";found=[]
            for arc in catalog['archives']:
                p=ROOT/arc['folder']/'entries.json';files.append(p);assert sha(p)==arc['listing_sha256']
                for e in json.loads(p.read_text())['files']:
                    if e['path']==wanted:found.append((arc,e))
            assert len(found)==1;arc,en=found[0];gt=[e for e in tgs if e['path'].endswith(f'/{scene}/test/ours_30000/gt/{index:05d}.png')];assert len(gt)==1
            rows.append(dict(scene=scene,cohort='heldout',view=f'heldout_{vi}',camera=c,split='test',split_index=index,original=en,archive=arc,published_gt=gt[0],GT=f'experiments/tgs_joint_policy_v29/{scene}/heldout_{vi}/gt_rgb.npy',excluded_prior_camera_names=sorted(used)))
    write(OUT/'input_plan.json',dict(rows=rows,inputs={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))},new_range_budget=96*1024*1024,test_payloads_seen=False,training=False))
    print(json.dumps(dict(stage='NEW_JOINT_POLICY_VIEWS_FROZEN',views=[dict(scene=r['scene'],view=r['view'],name=r['camera']['img_name']) for r in rows])),flush=True)
def fetch():
    p=json.loads((OUT/'input_plan.json').read_text());assert not (SOURCE/'manifest.json').exists();choice=json.loads((OUT/'choices.json').read_text());assert choice['calibration_only'] and not choice['heldout_payloads_seen'];assert json.loads((OUT/'calibration_results.json').read_text())['all_passed']
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    selected=[r for r in p['rows'] if r['cohort']=='heldout'];files=[]
    for r in selected:
        arc=r['archive'];reader=PublicZip(arc['url'],SOURCE/Path(arc['folder']).name,p['new_range_budget']-count());identity=json.loads((ROOT/arc['folder']/'head.json').read_text());assert reader.identity['size']==identity['size'] and reader.identity['etag']==identity['etag']
        dest=SOURCE/r['scene']/r['view'];dest.mkdir(parents=True)
        with zipfile.ZipFile(reader) as z:q=extract_member(z,r['original'],dest/'original.JPG',r['scene'],'original');files.append(dict(q,view=r['view']))
    remote=RemoteZip();folder=SOURCE/'published_gt'/'ranges';folder.mkdir(parents=True)
    for r in selected:
        en=r['published_gt'];start=en['offset'];n=30+len(en['path'].encode())+65535+en['compressed'];end=start+n-1;assert n<=16*1024*1024 and count()+n<=p['new_range_budget']
        with urlopen(Request(remote.url,headers={'User-Agent':'TGS-frozen-joint-policy','Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}),timeout=30) as response:
            cr=response.headers.get('Content-Range','');m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr);assert response.status==206 and m and tuple(map(int,m.groups()))==(start,end,remote.size)
            host=urlsplit(response.geturl()).hostname;assert host=='drive.usercontent.google.com' or host.endswith('.googleusercontent.com');raw=response.read(n+1);assert len(raw)==n;etag=response.headers.get('ETag')
        path=folder/f'{start}_{end}.bin';path.write_bytes(raw);write(path.with_suffix('.json'),dict(start=start,end=end,total=remote.size,bytes=n,status=206,sha256=sha(path),content_range=cr,etag=etag,created_utc=datetime.now(timezone.utc).isoformat()))
        sig,ver,flags,method,tm,dt,crc,cs,us,nl,extra=struct.unpack('<IHHHHHIIIHH',raw[:30]);assert sig==0x04034b50 and method==en['method']==8 and flags==en['flags'] and not(flags&1);assert raw[30:30+nl].decode()==en['path']
        if not(flags&8):assert (crc,cs,us)==(en['CRC'],en['compressed'],en['bytes'])
        image=zlib.decompress(raw[30+nl+extra:30+nl+extra+en['compressed']],-15);assert len(image)==en['bytes'] and zlib.crc32(image)==en['CRC'];dest=SOURCE/r['scene']/r['view']/'published_gt.png';assert not dest.exists();dest.write_bytes(image)
        q=dict(scene=r['scene'],view=r['view'],kind='published_gt',local=dest.relative_to(ROOT).as_posix(),archive_member=en['path'],bytes=len(image),sha256=sha(dest),CRC32=en['CRC'],zip_crc_checked=True,local_header_verified=True);write(dest.with_suffix('.png.receipt.json'),q);files.append(q);print(json.dumps(dict(stage='NEW_GT_MEMBER_SAVED',scene=r['scene'],view=r['view'])),flush=True)
    assert count()<=p['new_range_budget'];write(SOURCE/'manifest.json',dict(files=files,total_range_bytes=count(),choices_sha256=sha(OUT/'choices.json'),input_plan_sha256=sha(OUT/'input_plan.json'),training=False))
def audit():
    p=json.loads((OUT/'input_plan.json').read_text());mf=json.loads((SOURCE/'manifest.json').read_text());assert mf['choices_sha256']==sha(OUT/'choices.json');rows=[]
    for r in [r for r in p['rows'] if r['cohort']=='heldout']:
        images={}
        for f in [f for f in mf['files'] if f['scene']==r['scene'] and f['view']==r['view']]:
            assert sha(ROOT/f['local'])==f['sha256'] and f['zip_crc_checked'];images[f['kind']]=np.array(Image.open(ROOT/f['local']).convert('RGB'),dtype=np.uint8)
        a=images['original'];assert a.shape=={'flowers':(828,1256,3),'room':(1038,1557,3)}[r['scene']] and np.array_equal(a,images['published_gt']);dest=ROOT/r['GT'];dest.parent.mkdir(parents=True);np.save(dest,a,allow_pickle=False)
        rows.append(dict(scene=r['scene'],view=r['view'],original_published_RGB_literal=True,shape=list(a.shape),GT_sha256=sha(dest)))
    write(OUT/'heldout_identity.json',dict(all_passed=True,rows=rows,manifest_sha256=sha(SOURCE/'manifest.json'),training=False));print('NEW_HELDOUT_IDENTITIES_PASSED',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fetch','audit']);a=p.parse_args()
    try:globals()[a.action]()
    except BaseException:
        if OUT.exists():
            with (OUT/(a.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
