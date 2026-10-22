"""All old/fresh original RGB vs public GT identity, no scoring or transformations."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,__version__ as pillow_version
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT/'sources/tgs_sh_isolation_v12';OUT=ROOT/'experiments/tgs_sh_isolation_v12'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'identity_audit.json';assert not target.exists();plan=json.loads((OUT/'input_plan.json').read_text());manifest=json.loads((SOURCE/'manifest.json').read_text());rows=[]
    for row in plan['rows']:
        scene,cohort=row['scene'],row['cohort'];files={e['kind']:e for e in manifest['files'] if e['scene']==scene and e['cohort']==cohort};images={}
        for kind,e in files.items():
            p=ROOT/e['local'];assert sha(p)==e['sha256'] and e['zip_crc_checked'];images[kind]=np.array(Image.open(p).convert('RGB'),dtype=np.uint8)
        a,b=images['original'],images['published_gt'];same=a.shape==b.shape and bool(np.array_equal(a,b));dest=OUT/scene/cohort;dest.mkdir(parents=True);np.save(dest/'gt_rgb.npy',b,allow_pickle=False);count=int(np.count_nonzero(a!=b)) if a.shape==b.shape else None;q=dict(scene=scene,cohort=cohort,camera=row['camera']['img_name'],test_index=row['test_index'],shape=list(b.shape),different_channel_values=count,exact_RGB=bool(same),decoded_RGB_sha256=hashlib.sha256(b.tobytes()).hexdigest(),resizing_or_registration=False);rows.append(q);print(json.dumps(q),flush=True)
    out=dict(rows=rows,all_gates=all(r['exact_RGB'] for r in rows),quality_allowed=all(r['exact_RGB'] for r in rows),input_plan_sha256=sha(OUT/'input_plan.json'),source_manifest_sha256=sha(SOURCE/'manifest.json'),pillow=pillow_version,script_sha256=sha(Path(__file__)),training=False);target.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
