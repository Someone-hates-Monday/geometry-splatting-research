"""Extract only the lowest-transfer real checkpoint and necessary metadata.

Selection uses ZIP compressed size, before inspecting any rendering outcome.
Model payload is data only; loading happens separately with weights_only=True.
"""
import hashlib
import json
import zipfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
from urllib.request import Request,urlopen
from remote_tgs_zip import ROOT as LISTING, RemoteZip

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'sources/tgs_real_checkpoint_v1'
PIN='685ff270a8273afc3a2b094ae3148dd0b98e31e9'


def archive_sources():
    tree=json.loads((ROOT/'sources/tgs_subdivision_static_v1/triangle-splatting/metadata/tree.json').read_text())
    index={x['path']:x for x in tree['tree']}
    paths=['scene/__init__.py','scene/cameras.py','scene/dataset_readers.py','utils/camera_utils.py',
           'utils/graphics_utils.py','utils/sh_utils.py','full_eval.py','render.py']
    records=[]
    for rel in paths:
        target=OUT/'source'/rel
        if target.exists():raw=target.read_bytes()
        else:
            url=f'https://raw.githubusercontent.com/trianglesplatting/triangle-splatting/{PIN}/{rel}'
            with urlopen(Request(url,headers={'User-Agent':'TGS-research-input-audit'}),timeout=45) as r:
                raw=r.read(1024*1024)
        blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        assert blob==index[rel]['sha'] and len(raw)==index[rel]['size']
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(raw)
        records.append({'path':rel,'git_blob_sha1':blob,'sha256':hashlib.sha256(raw).hexdigest()})
    (OUT/'source_manifest.json').write_text(json.dumps({'commit':PIN,'files':records},indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    entries=json.loads((LISTING/'zip_entries.json').read_text())['files']
    candidates=sorted([e for e in entries if e['path'].endswith('/point_cloud_state_dict.pt')],
                       key=lambda e:(e['compressed'],e['path']))
    chosen=candidates[0]
    scene_path=chosen['path'].split('/point_cloud/')[0]
    iteration=chosen['path'].split('/point_cloud/')[1].split('/')[0]
    scene=scene_path.rsplit('/',1)[-1]
    wanted=[chosen['path'],chosen['path'].rsplit('/',1)[0]+'/hyperparameters.pt',
            scene_path+'/cfg_args',scene_path+'/cameras.json']
    # One existing GT image is enough to verify actual rendered dimensions;
    # it does not constitute dataset evaluation of the 7000 checkpoint.
    gt=sorted(e['path'] for e in entries if e['path'].startswith(scene_path+'/test/') and '/gt/' in e['path'] and e['path'].endswith('.png'))
    if gt:wanted.append(gt[0])
    selection={'rule':'minimum ZIP compressed size among all official state_dict checkpoints; tie by path',
               'scene':scene,'iteration':iteration,'chosen':chosen,'files':wanted,
               'selected_before_rendering':True,'created_utc':datetime.now(timezone.utc).isoformat()}
    selection_path=OUT/'selection.json'
    if selection_path.exists():
        old=json.loads(selection_path.read_text());assert old['files']==wanted and old['chosen']==chosen
    else:selection_path.write_text(json.dumps(selection,indent=2)+'\n')
    archive_sources()
    r=RemoteZip(max_download=220*1024*1024)
    records=[]
    with zipfile.ZipFile(r) as z:
        for member in wanted:
            rel=PurePosixPath(member).relative_to(PurePosixPath(scene_path))
            assert '..' not in rel.parts
            target=OUT/scene/str(rel)
            info=z.getinfo(member)
            receipt_path=target.with_suffix(target.suffix+'.receipt.json')
            if target.exists() and receipt_path.exists():
                receipt=json.loads(receipt_path.read_text())
                assert hashlib.sha256(target.read_bytes()).hexdigest()==receipt['sha256']
                records.append(receipt);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_suffix(target.suffix+'.partial')
            digest=hashlib.sha256();count=0
            with z.open(info) as src,partial.open('wb') as dst:
                while True:
                    chunk=src.read(8*1024*1024)
                    if not chunk:break
                    dst.write(chunk);digest.update(chunk);count+=len(chunk)
                    print(json.dumps({'file':str(rel),'decoded_bytes':count,'expected':info.file_size,
                                      'network_new_bytes':r.downloaded}),flush=True)
            assert count==info.file_size  # ZipExtFile also verifies CRC at EOF.
            partial.replace(target)
            receipt={'archive_member':member,'local':str(target.relative_to(ROOT)),
                     'bytes':count,'sha256':digest.hexdigest(),'CRC32':info.CRC,
                     'zip_crc_checked':True,'compressed_bytes':info.compress_size,
                     'extracted_utc':datetime.now(timezone.utc).isoformat()}
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');records.append(receipt)
    report={'files':records,'source':'official README public full_eval.zip',
            'archive_size':r.size,'new_network_bytes':r.downloaded,
            'range_records':r.request_records,'training_commit_proven':False,
            'scope':'real published model tensors and cameras; no captured split or full optimizer state'}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'complete':True,'scene':scene,'iteration':iteration,'new_network_bytes':r.downloaded}),flush=True)


if __name__=='__main__':
    main()
