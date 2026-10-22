"""Strict decoded-image identity plus pinned ordering, before any quality scores."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,features,__version__ as pillow_version

ROOT=Path(__file__).resolve().parent;SOURCE=ROOT/'sources/tgs_gt_identity_v11';OUT=ROOT/'experiments/tgs_gt_quality_v11';PINNED=ROOT/'sources/tgs_real_checkpoint_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);target=OUT/'identity_audit.json';assert not target.exists();manifest=json.loads((SOURCE/'manifest.json').read_text());selection=json.loads((SOURCE/'selection.json').read_text());assert manifest['selection_sha256']==sha(SOURCE/'selection.json')
    for e in manifest['files']:
        p=ROOT/e['local'];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'] and e['zip_crc_checked']
    provenance=json.loads((PINNED/'source_manifest.json').read_text())
    for en in provenance['files']:assert sha(PINNED/'source'/en['path'])==en['sha256']
    source=PINNED/'source/render.py';tree=ast.parse(source.read_text());scenes=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Scene'];assert len(scenes)==1 and any(k.arg=='shuffle' and isinstance(k.value,ast.Constant) and k.value.value is False for k in scenes[0].keywords)
    text=source.read_text();assert 'for idx, view in enumerate(' in text and "'{0:05d}'.format(idx)" in text;readers=(PINNED/'source/scene/dataset_readers.py').read_text();assert 'llffhold=8' in readers.replace(' ','') and 'image_name' in readers
    rows=[]
    for row in selection['rows']:
        scene=row['scene'];folder=SOURCE/scene;original=Image.open(folder/'original.JPG');gt=Image.open(folder/'published_gt.png');a=np.asarray(original.convert('RGB'),dtype=np.uint8);b=np.asarray(gt.convert('RGB'),dtype=np.uint8);same_shape=a.shape==b.shape;count=int(np.count_nonzero(a!=b)) if same_shape else None;exact=bool(same_shape and count==0);delta=np.abs(a.astype(np.int16)-b.astype(np.int16)) if same_shape else None;camera=row['camera'];canonical=sorted(json.loads((ROOT/'sources/tgs_cross_scene_v10'/scene/'cameras.json').read_text()),key=lambda c:c['img_name'])[::8];assert canonical[row['expected_GT_index']]['img_name']==camera['img_name'];dest=OUT/scene;dest.mkdir();np.save(dest/'original_rgb.npy',a,allow_pickle=False);np.save(dest/'gt_rgb.npy',b,allow_pickle=False)
        q=dict(scene=scene,camera=camera['img_name'],predicted_GT_index=row['expected_GT_index'],original_archive_member=row['original']['path'],GT_archive_member=row['published_gt']['path'],original_shape=list(a.shape),GT_shape=list(b.shape),original_mode=original.mode,GT_mode=gt.mode,original_EXIF_orientation=original.getexif().get(274),EXIF_transform_applied=False,RGB_transform='Pillow convert RGB only; no resize/crop/registration/color transform',different_channel_values=count,maximum_uint8_difference=int(delta.max()) if delta is not None else None,original_decoded_RGB_sha256=hashlib.sha256(a.tobytes()).hexdigest(),GT_decoded_RGB_sha256=hashlib.sha256(b.tobytes()).hexdigest(),canonical_source_order_verified=True,exact_decoded_RGB_match=exact,gate=exact);rows.append(q);print(json.dumps(q),flush=True)
    out=dict(rows=rows,all_gates=all(r['gate'] for r in rows),quality_measurement_allowed=all(r['gate'] for r in rows),pillow=pillow_version,jpeg_codec=features.version_codec('jpg'),libjpeg_turbo=features.version_feature('libjpeg_turbo'),source_render_sha256=sha(source),source_manifest_sha256=sha(PINNED/'source_manifest.json'),manifest_sha256=sha(SOURCE/'manifest.json'),selection_sha256=sha(SOURCE/'selection.json'),script_sha256=sha(Path(__file__)),training=False,quality_scores_computed=False);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(all_gates=out['all_gates'],quality_measurement_allowed=out['quality_measurement_allowed'])),flush=True)
if __name__=='__main__':main()
