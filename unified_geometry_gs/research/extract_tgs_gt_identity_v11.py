"""Freeze four exact image members, then CRC/hash-checked bounded public extraction."""
import hashlib,json,traceback,zipfile
from pathlib import Path
from remote_public_zip_v11 import PublicZip,sha
from remote_tgs_zip import RemoteZip,ROOT as TGS_LISTING

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'sources/tgs_gt_identity_v11';V10=ROOT/'experiments/tgs_cross_scene_v10';MODEL=ROOT/'sources/tgs_cross_scene_v10'
def cumulative():return sum(json.loads(p.read_text())['bytes'] for p in OUT.glob('dataset_*/ranges/*.json'))+sum(json.loads(p.read_text())['bytes'] for p in (OUT/'tgs_ranges').glob('*.json'))

def main():
    target=OUT/'manifest.json';assert not target.exists();catalog=json.loads((OUT/'catalogue.json').read_text());assert not catalog['image_payloads_fetched'];input_audit=json.loads((V10/'input_audit.json').read_text());all_tgs={e['path']:e for e in json.loads((TGS_LISTING/'zip_entries.json').read_text())['files']};selected=[]
    for scene in ['flowers','room']:
        sc=next(s for s in input_audit['scenes'] if s['scene']==scene);cam=next(c for c in sc['selected_cameras'] if c['split']=='test');canonical=sorted(json.loads((MODEL/scene/'cameras.json').read_text()),key=lambda c:c['img_name'])[::8];index=[c['img_name'] for c in canonical].index(cam['img_name']);assert index=={'flowers':11,'room':31}[scene];group='images_4' if scene=='flowers' else 'images_2';wanted=f"{scene}/{group}/{cam['img_name']}.JPG";found=[]
        for arc in catalog['archives']:
            listing=ROOT/arc['folder']/'entries.json';assert sha(listing)==arc['listing_sha256']
            for e in json.loads(listing.read_text())['files']:
                if e['path']==wanted:found.append((arc,e))
        assert len(found)==1;arc,entry=found[0];suffix=f'{scene}/test/ours_30000/gt/{index:05d}.png';gt=[e for p,e in all_tgs.items() if p.endswith('/'+suffix)];assert len(gt)==1;selected.append(dict(scene=scene,camera=cam,expected_GT_index=index,original_archive=arc,original=entry,published_gt=gt[0]))
    selection=dict(rows=selected,protocol_sha256=sha(ROOT/'TGS_GT_IDENTITY_QUALITY_PROTOCOL_V11.md'),control_addendum_sha256=sha(ROOT/'TGS_GT_CONTROL_ADDENDUM_V11.md'),catalogue_sha256=sha(OUT/'catalogue.json'),v10_input_audit_sha256=sha(V10/'input_audit.json'),script_sha256=sha(Path(__file__)),frozen_before_image_payloads=True,training=False)
    p=OUT/'selection.json'
    if p.exists():assert json.loads(p.read_text())==selection
    else:p.write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(dict(stage='members_frozen',images=[dict(scene=r['scene'],original=r['original']['path'],GT=r['published_gt']['path']) for r in selected])),flush=True)
    receipts=[]
    for row in selected:
        arc=row['original_archive'];reader=PublicZip(arc['url'],ROOT/arc['folder'],96*1024*1024-cumulative())
        receipts.append(extract(reader,row['original'],OUT/row['scene']/'original.JPG',row['scene'],'original'))
    tgs=RemoteZip(max_download=96*1024*1024-cumulative());tgs.cache=OUT/'tgs_ranges';tgs.cache.mkdir(exist_ok=True)
    with zipfile.ZipFile(tgs) as z:
        for row in selected:receipts.append(extract_member(z,row['published_gt'],OUT/row['scene']/'published_gt.png',row['scene'],'published_gt'))
    count=cumulative();assert count<=96*1024*1024;target.write_text(json.dumps(dict(files=receipts,total_cached_range_payload_bytes=count,selection_sha256=sha(p),script_sha256=sha(Path(__file__)),GT_identity_verified=False,training=False),indent=2)+'\n');print(json.dumps(dict(complete=True,files=len(receipts),total_cached_range_payload_bytes=count)),flush=True)

def extract(reader,entry,target,scene,kind):
    with zipfile.ZipFile(reader) as z:return extract_member(z,entry,target,scene,kind)
def extract_member(z,entry,target,scene,kind):
    receipt=target.with_suffix(target.suffix+'.receipt.json')
    if target.exists() and receipt.exists():
        q=json.loads(receipt.read_text());assert sha(target)==q['sha256'] and target.stat().st_size==q['bytes'];return q
    assert not target.exists();target.parent.mkdir(exist_ok=True);info=z.getinfo(entry['path']);assert (info.file_size,info.compress_size,info.CRC,info.compress_type,info.header_offset)==(entry['bytes'],entry['compressed'],entry['CRC'],entry['method'],entry['offset']);assert info.file_size<=16*1024*1024
    raw=z.read(info);assert len(raw)==entry['bytes'];target.write_bytes(raw);q=dict(scene=scene,kind=kind,archive_member=entry['path'],local=target.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),CRC32=info.CRC,zip_crc_checked=True);receipt.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(dict(scene=scene,kind=kind,bytes=len(raw))),flush=True);return q

if __name__=='__main__':
    try:main()
    except BaseException:(OUT/'extraction_failure.txt').write_text(traceback.format_exc());raise
