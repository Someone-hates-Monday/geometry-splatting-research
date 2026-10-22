"""Metadata-first public ranges, with holdout payloads gated by frozen choices."""
import argparse, hashlib, json, traceback, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from remote_public_zip_v11 import PublicZip
from remote_tgs_zip import RemoteZip, ROOT as LISTING
from extract_tgs_gt_identity_v11 import extract_member
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_policy_transfer_v28'
SOURCE=ROOT/'sources/tgs_policy_transfer_v28'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def cumulative():return sum(json.loads(p.read_text())['bytes'] for p in SOURCE.glob('**/ranges/*.json'))
def prepare():
    OUT.mkdir();SOURCE.mkdir()
    catalog=json.loads((ROOT/'sources/tgs_gt_identity_v11/catalogue.json').read_text())
    v10=json.loads((ROOT/'experiments/tgs_cross_scene_v10/input_audit.json').read_text())
    v12=json.loads((ROOT/'experiments/tgs_sh_isolation_v12/input_plan.json').read_text())
    tgs=json.loads((LISTING/'zip_entries.json').read_text())['files'];rows=[]
    dependencies=[Path(__file__),ROOT/'TGS_POLICY_TRANSFER_PROTOCOL_V28.md',ROOT/'remote_public_zip_v11.py',ROOT/'remote_tgs_zip.py',ROOT/'extract_tgs_gt_identity_v11.py',ROOT/'sources/tgs_gt_identity_v11/catalogue.json',LISTING/'zip_entries.json',ROOT/'experiments/tgs_cross_scene_v10/input_audit.json',ROOT/'experiments/tgs_sh_isolation_v12/input_plan.json']
    for scene in ['flowers','room']:
        cp=ROOT/'sources/tgs_cross_scene_v10'/scene/'cameras.json';dependencies.append(cp)
        cams=sorted(json.loads(cp.read_text()),key=lambda c:c['img_name'])
        used={c['img_name'] for s in v10['scenes'] if s['scene']==scene for c in s['selected_cameras']}
        used.update(r['camera']['img_name'] for r in v12['rows'] if r['scene']==scene)
        for split,label in [('train','calibration'),('test','heldout')]:
            group=[c for i,c in enumerate(cams) if (i%8==0)==(split=='test')]
            eligible=[c for c in group if c['img_name'] not in used]
            chosen=sorted(eligible,key=lambda c:(hashlib.sha256(f"tgs_policy_transfer_v28/{scene}/{split}/{c['img_name']}".encode()).hexdigest(),c['img_name']))[:2]
            assert len(chosen)==2
            for vi,c in enumerate(chosen):
                wanted=f"{scene}/{'images_4' if scene=='flowers' else 'images_2'}/{c['img_name']}.JPG";found=[]
                for arc in catalog['archives']:
                    lp=ROOT/arc['folder']/'entries.json';assert sha(lp)==arc['listing_sha256'];dependencies.append(lp)
                    for en in json.loads(lp.read_text())['files']:
                        if en['path']==wanted:found.append((arc,en))
                assert len(found)==1;arc,en=found[0];gt=None
                index=[x['img_name'] for x in group].index(c['img_name'])
                if split=='test':
                    suffix=f"/{scene}/test/ours_30000/gt/{index:05d}.png";matches=[e for e in tgs if e['path'].endswith(suffix)];assert len(matches)==1;gt=matches[0]
                rows.append(dict(scene=scene,cohort=label,view=f'{label}_{vi}',camera=c,split=split,split_index=index,archive=arc,original=en,published_gt=gt,excluded_prior_camera_names=sorted(used)))
    write(OUT/'input_plan.json',dict(rows=rows,inputs={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(dependencies))},new_payload_budget=96*1024*1024,training=False,test_payloads_seen=False))
    print(json.dumps(dict(stage='CAMERAS_AND_MEMBERS_FROZEN',rows=[{k:r[k] for k in ['scene','cohort','view','split_index']}|dict(name=r['camera']['img_name']) for r in rows])),flush=True)
def fetch(cohort):
    plan=json.loads((OUT/'input_plan.json').read_text());target=SOURCE/(cohort+'_manifest.json');assert not target.exists()
    for n,h in plan['inputs'].items():assert sha(ROOT/n)==h,n
    if cohort=='heldout':
        choice=json.loads((OUT/'choices.json').read_text());assert choice['calibration_only'] and not choice['test_scores_seen'];choice_hash=sha(OUT/'choices.json')
    else:choice_hash=None
    selected=[r for r in plan['rows'] if r['cohort']==cohort];records=[]
    for row in selected:
        arc=row['archive'];folder=SOURCE/Path(arc['folder']).name
        reader=PublicZip(arc['url'],folder,plan['new_payload_budget']-cumulative());old=json.loads((ROOT/arc['folder']/'head.json').read_text())
        assert reader.identity['size']==old['size'] and reader.identity['etag']==old['etag']
        dest=SOURCE/row['scene']/row['view'];dest.mkdir(parents=True)
        with zipfile.ZipFile(reader) as z:q=extract_member(z,row['original'],dest/'original.JPG',row['scene'],'original')
        records.append(dict(q,view=row['view']))
    if cohort=='heldout':
        reader=RemoteZip(max_download=plan['new_payload_budget']-cumulative());reader.cache=SOURCE/'tgs'/'ranges';reader.cache.mkdir(parents=True)
        with zipfile.ZipFile(reader) as z:
            for row in selected:
                q=extract_member(z,row['published_gt'],SOURCE/row['scene']/row['view']/'published_gt.png',row['scene'],'published_gt');records.append(dict(q,view=row['view']))
    assert cumulative()<=plan['new_payload_budget']
    write(target,dict(files=records,input_plan_sha256=sha(OUT/'input_plan.json'),choices_sha256=choice_hash,total_range_bytes=cumulative(),training=False))
    print(json.dumps(dict(stage='FETCHED',cohort=cohort,range_bytes=cumulative())),flush=True)
def audit(cohort):
    plan=json.loads((OUT/'input_plan.json').read_text());mp=SOURCE/(cohort+'_manifest.json');manifest=json.loads(mp.read_text());rows=[]
    for row in [r for r in plan['rows'] if r['cohort']==cohort]:
        dest=SOURCE/row['scene']/row['view'];files={e['kind']:e for e in manifest['files'] if e['scene']==row['scene'] and e['view']==row['view']}
        arrays={}
        for kind,e in files.items():
            p=ROOT/e['local'];assert sha(p)==e['sha256'] and e['zip_crc_checked'];arrays[kind]=np.array(Image.open(p).convert('RGB'),dtype=np.uint8)
        a=arrays['original'];expected={'flowers':(828,1256,3),'room':(1038,1557,3)}[row['scene']];assert a.shape==expected,(row,a.shape)
        identical=bool(np.array_equal(a,arrays['published_gt'])) if cohort=='heldout' else None
        if cohort=='heldout':assert identical
        d=OUT/row['scene']/row['view'];d.mkdir(parents=True);np.save(d/'gt_rgb.npy',a,allow_pickle=False)
        rows.append(dict(scene=row['scene'],view=row['view'],camera=row['camera']['img_name'],shape=list(a.shape),decoded_RGB_sha256=hashlib.sha256(a.tobytes()).hexdigest(),original_published_exact=identical,source_JPEG_calibration_only=cohort=='calibration',resized=False))
    write(OUT/(cohort+'_identity.json'),dict(all_passed=True,rows=rows,manifest_sha256=sha(mp),input_plan_sha256=sha(OUT/'input_plan.json'),training=False))
    print(json.dumps(dict(stage='IDENTITY_PASSED',cohort=cohort,views=len(rows))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fetch_calibration','audit_calibration','fetch_heldout','audit_heldout']);args=p.parse_args()
    try:
        if args.action=='prepare':prepare()
        else:verb,cohort=args.action.split('_',1);globals()[verb](cohort)
    except BaseException:
        if OUT.exists():
            with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
