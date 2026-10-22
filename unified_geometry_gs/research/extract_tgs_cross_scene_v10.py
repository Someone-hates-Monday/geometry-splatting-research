"""Extract two prespecified public checkpoints; own cache, bounded transfer, CRC checked."""
import hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
from remote_tgs_zip import ROOT as LISTING,RemoteZip

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'sources/tgs_cross_scene_v10'
GROUPS={'outdoor':['bicycle','flowers','garden','treehill'],'indoor':['room','counter','kitchen','bonsai']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True);entries=json.loads((LISTING/'zip_entries.json').read_text())['files'];chosen=[]
    for group,names in GROUPS.items():
        candidates=[e for e in entries if '/point_cloud/iteration_7000/point_cloud_state_dict.pt' in e['path'] and e['path'].split('/point_cloud/')[0].rsplit('/',1)[-1] in names]
        en=min(candidates,key=lambda e:(e['compressed'],e['path']));scene_path=en['path'].split('/point_cloud/')[0];scene=scene_path.rsplit('/',1)[-1];members=[en['path'],en['path'].rsplit('/',1)[0]+'/hyperparameters.pt',scene_path+'/cfg_args',scene_path+'/cameras.json'];gt=sorted(e['path'] for e in entries if e['path'].startswith(scene_path+'/test/') and '/gt/' in e['path'] and e['path'].endswith('.png'));assert gt;members.append(gt[0]);chosen.append(dict(group=group,scene=scene,scene_path=scene_path,iteration='iteration_7000',chosen=en,members=members))
    assert [x['scene'] for x in chosen]==['flowers','room']
    selection=dict(rule='Per group minimum compressed iteration7000 checkpoint, excluding previously used stump; tie by archive path.',groups=GROUPS,scenes=chosen,listing_sha256=sha(LISTING/'zip_entries.json'),script_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'TGS_CROSS_SCENE_PROTOCOL_V10.md'),selected_before_payload=True,created_utc=datetime.now(timezone.utc).isoformat(),training=False)
    select=OUT/'selection.json'
    if select.exists():
        old=json.loads(select.read_text());assert old['scenes']==chosen and old['script_sha256']==selection['script_sha256'] and old['protocol_sha256']==selection['protocol_sha256']
    else:select.write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(dict(stage='selected_before_payload',scenes=[dict(scene=x['scene'],compressed=x['chosen']['compressed']) for x in chosen])),flush=True)
    r=RemoteZip(max_download=650*1024*1024);r.cache=OUT/'ranges';r.cache.mkdir(exist_ok=True);records=[]
    with zipfile.ZipFile(r) as z:
        for s in chosen:
            for member in s['members']:
                rel=PurePosixPath(member).relative_to(PurePosixPath(s['scene_path']));assert '..' not in rel.parts;target=OUT/s['scene']/str(rel);receipt_path=target.with_suffix(target.suffix+'.receipt.json');info=z.getinfo(member)
                if target.exists() and receipt_path.exists():
                    q=json.loads(receipt_path.read_text());assert sha(target)==q['sha256'] and target.stat().st_size==q['bytes'];records.append(q);continue
                assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);part=target.with_suffix(target.suffix+'.partial');assert not part.exists(),'Retain partial transfer; inspect failure and choose explicit resume policy';digest=hashlib.sha256();count=0
                with z.open(info) as src,part.open('wb') as dst:
                    while True:
                        raw=src.read(8*1024*1024)
                        if not raw:break
                        dst.write(raw);digest.update(raw);count+=len(raw);print(json.dumps(dict(scene=s['scene'],file=str(rel),bytes=count,expected=info.file_size,new_network_bytes=r.downloaded)),flush=True)
                assert count==info.file_size;part.replace(target);q=dict(archive_member=member,local=target.relative_to(ROOT).as_posix(),bytes=count,sha256=digest.hexdigest(),CRC32=info.CRC,zip_crc_checked=True,compressed_bytes=info.compress_size,extracted_utc=datetime.now(timezone.utc).isoformat());receipt_path.write_text(json.dumps(q,indent=2)+'\n');records.append(q)
    report=dict(files=records,archive_size=r.size,new_network_bytes=r.downloaded,range_records=r.request_records,selection_sha256=sha(select),source='Official README public full_eval.zip',script_sha256=sha(Path(__file__)),training_commit_proven=False,training=False)
    target=OUT/'manifest.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(complete=True,scenes=[s['scene'] for s in chosen],new_network_bytes=r.downloaded)),flush=True)

if __name__=='__main__':main()
