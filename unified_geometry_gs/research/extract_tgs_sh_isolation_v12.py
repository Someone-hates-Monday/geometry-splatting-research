"""Old verified image references and two fresh, prespecified JPEG/GT pairs only."""
import json,zipfile
from pathlib import Path
from remote_public_zip_v11 import PublicZip,sha
from remote_tgs_zip import RemoteZip,ROOT as TGS_LISTING
from extract_tgs_gt_identity_v11 import extract_member
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'sources/tgs_sh_isolation_v12';EXP=ROOT/'experiments/tgs_sh_isolation_v12';OLD=ROOT/'sources/tgs_gt_identity_v11'
def cumulative():return sum(json.loads(p.read_text())['bytes'] for p in OUT.glob('dataset_*/ranges/*.json'))+sum(json.loads(p.read_text())['bytes'] for p in (OUT/'tgs_ranges').glob('*.json'))
def main():
    target=OUT/'manifest.json';assert not target.exists();plan=json.loads((EXP/'input_plan.json').read_text());catalog=json.loads((OLD/'catalogue.json').read_text());tgs_entries=json.loads((TGS_LISTING/'zip_entries.json').read_text())['files'];selected=[];records=[]
    for row in plan['rows']:
        scene,cohort=row['scene'],row['cohort'];name=row['camera']['img_name']
        if cohort=='old_posthoc':
            for kind,file in [('original','original.JPG'),('published_gt','published_gt.png')]:
                p=OLD/scene/file;q=json.loads(p.with_suffix(p.suffix+'.receipt.json').read_text());assert sha(p)==q['sha256'];records.append(dict(q,cohort=cohort))
            continue
        folder='images_4' if scene=='flowers' else 'images_2';wanted=f'{scene}/{folder}/{name}.JPG';found=[]
        for arc in catalog['archives']:
            for en in json.loads((ROOT/arc['folder']/'entries.json').read_text())['files']:
                if en['path']==wanted:found.append((arc,en))
        assert len(found)==1;arc,en=found[0];suffix=f"/{scene}/test/ours_30000/gt/{row['test_index']:05d}.png";gt=[e for e in tgs_entries if e['path'].endswith(suffix)];assert len(gt)==1;selected.append(dict(row,archive=arc,original=en,published_gt=gt[0]))
    assert len(selected)==2;p=OUT/'image_selection.json';selection=dict(rows=selected,input_plan_sha256=sha(EXP/'input_plan.json'),script_sha256=sha(Path(__file__)),fresh_payloads_seen=False,training=False)
    if p.exists():assert json.loads(p.read_text())==selection
    else:p.write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(dict(stage='fresh_members_frozen',images=[dict(scene=r['scene'],camera=r['camera']['img_name'],GT_index=r['test_index']) for r in selected])),flush=True)
    for row in selected:
        arc=row['archive'];folder=OUT/Path(arc['folder']).name;reader=PublicZip(arc['url'],folder,96*1024*1024-cumulative());old_head=json.loads((ROOT/arc['folder']/'head.json').read_text());assert reader.identity['size']==old_head['size'] and reader.identity['etag']==old_head['etag'];dest=OUT/row['scene']/'fresh';dest.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(reader) as z:q=extract_member(z,row['original'],dest/'original.JPG',row['scene'],'original');records.append(dict(q,cohort='fresh'))
    reader=RemoteZip(max_download=96*1024*1024-cumulative());reader.cache=OUT/'tgs_ranges';reader.cache.mkdir(exist_ok=True)
    with zipfile.ZipFile(reader) as z:
        for row in selected:q=extract_member(z,row['published_gt'],OUT/row['scene']/'fresh'/'published_gt.png',row['scene'],'published_gt');records.append(dict(q,cohort='fresh'))
    assert len(records)==8 and cumulative()<=96*1024*1024;target.write_text(json.dumps(dict(files=records,old_views_referenced_readonly=True,total_new_cached_Range_payload_bytes=cumulative(),selection_sha256=sha(p),script_sha256=sha(Path(__file__)),training=False),indent=2)+'\n');print(json.dumps(dict(complete=True,files=len(records),new_payload=cumulative())),flush=True)
if __name__=='__main__':main()
