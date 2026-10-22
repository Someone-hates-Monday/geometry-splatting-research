"""CPU provenance and canonical scene/camera validation before new renderer outputs."""
import json,math,re,struct
from pathlib import Path
import numpy as np
import torch
from tgs_cross_scene_common_v10 import ROOT,SOURCE,OUT,V9,V9_SHA,sha,model_input

def main():
    OUT.mkdir(exist_ok=True);target=OUT/'input_audit.json';assert not target.exists();manifest=json.loads((SOURCE/'manifest.json').read_text());selection=json.loads((SOURCE/'selection.json').read_text());assert selection['protocol_sha256']==sha(ROOT/'TGS_CROSS_SCENE_PROTOCOL_V10.md')
    listing_path=ROOT/'sources/tgs_pretrained_listing_v1/zip_entries.json';assert sha(listing_path)==selection['listing_sha256'];listing={r['path']:r for r in json.loads(listing_path.read_text())['files']}
    assert selection['script_sha256']==manifest['script_sha256']==sha(ROOT/'extract_tgs_cross_scene_v10.py')
    for en in manifest['files']:
        p=ROOT/en['local'];frozen=listing[en['archive_member']];assert p.stat().st_size==en['bytes']==frozen['bytes'] and sha(p)==en['sha256'] and en['zip_crc_checked'];assert en['CRC32']==frozen['CRC'] and en['compressed_bytes']==frozen['compressed']
    acquisition=json.loads((SOURCE/'acquisition_audit.json').read_text());assert acquisition['complete_members']==10 and acquisition['successful_cached_network_payload_bytes']<=650*1024*1024 and acquisition['manifest_sha256']==sha(SOURCE/'manifest.json')
    assert manifest['new_network_bytes']<=650*1024*1024 and manifest['selection_sha256']==sha(SOURCE/'selection.json')
    original=ROOT/'sources/tgs_real_checkpoint_v1';provenance=json.loads((original/'source_manifest.json').read_text())
    for en in provenance['files']:assert sha(original/'source'/en['path'])==en['sha256']
    assert sha(V9/'evidence_manifest.json')==V9_SHA;prior=json.loads((V9/'evidence_manifest.json').read_text())
    for en in prior['files']:assert sha(ROOT/en['path'])==en['sha256']
    rows=[]
    for en in selection['scenes']:
        scene=en['scene'];model,hyper,cfg,cameras=model_input(scene);assert set(model)=={'triangles_points','sigma','active_sh_degree','features_dc','features_rest','opacity'};n=len(model['triangles_points']);assert tuple(model['triangles_points'].shape)==(n,3,3);assert tuple(model['sigma'].shape)==tuple(model['opacity'].shape)==(n,1);assert tuple(model['features_dc'].shape)==(n,1,3);assert model['features_rest'].shape==(n,15,3) and int(model['active_sh_degree'])==3
        for k,x in model.items():
            if isinstance(x,torch.Tensor):assert x.dtype==torch.float32 and torch.isfinite(x).all()
        sig=.01+torch.exp(model['sigma']);op=torch.sigmoid(model['opacity']);assert torch.isfinite(sig).all() and (sig>=.01).all();assert cfg['eval']
        names=[c['img_name'] for c in cameras];assert len(names)==len(set(names));ordered=sorted(cameras,key=lambda c:c['img_name']);test=ordered[::8];train=[c for i,c in enumerate(ordered) if i%8];assert names==[c['img_name'] for c in test+train]
        gt=list((SOURCE/scene/'test').rglob('gt/*.png'));assert len(gt)==1;header=gt[0].read_bytes()[:24];assert header[:8]==b'\x89PNG\r\n\x1a\n';size=struct.unpack('>II',header[16:24]);match=re.fullmatch(r'images_(\d+)',cfg['images']);assert match and cfg['resolution']==-1;factor=int(match[1]);assert size[0]<=1600
        # Official loadCam uses decoded image.size. The scene's GT PNG is the
        # raster-size witness: images_4 does NOT universally mean ceil(COLMAP/4).
        # flowers width5025 has published width1256, whereas stump4978 had1245.
        for c in cameras:
            for dim,actual in zip(['width','height'],size):assert math.floor(c[dim]/factor)<=actual<=math.ceil(c[dim]/factor)
        max_rotation=0.
        for c in cameras:
            rot=np.array(c['rotation']);pos=np.array(c['position']);assert rot.shape==(3,3) and pos.shape==(3,) and np.isfinite(rot).all() and np.isfinite(pos).all();err=float(np.abs(rot.T@rot-np.eye(3)).max());max_rotation=max(max_rotation,err);assert err<1e-5 and abs(np.linalg.det(rot)-1)<1e-5 and min(c['width'],c['height'],c['fx'],c['fy'])>0
        chosen=[]
        for split,group in [('test',test),('train',train)]:
            c=min(group,key=lambda c:__import__('hashlib').sha256(('tgs_cross_scene_v10/'+scene+'/'+split+'/'+c['img_name']).encode()).hexdigest());chosen.append(dict(c,split=split,selected_before_rendering=True))
        row=dict(scene=scene,group=en['group'],iteration='iteration_7000',triangles=n,active_sh_degree=3,actual_resolution=list(size),resolution_evidence='Published PNG raster size; official loadCam reads decoded image.size. COLMAP dimensions/image-factor only bracket integer rounding, not forced ceil. Training images not individually decoded.',camera_count=len(cameras),test_count=len(test),train_count=len(train),selected_cameras=chosen,cfg_args=cfg,sigma_range=[float(sig.min()),float(sig.max())],opacity_range=[float(op.min()),float(op.max())],max_rotation_error=max_rotation,hyperparameter_keys=sorted(hyper),GT_camera_identity_verified=False,GT_used_only_for_dimensions=True,model_sha256=sha(SOURCE/scene/'point_cloud/iteration_7000/point_cloud_state_dict.pt'));rows.append(row);print(json.dumps({k:row[k] for k in ['scene','triangles','actual_resolution','sigma_range','selected_cameras']}),flush=True)
    out=dict(passed=True,scenes=rows,source_manifest_sha256=sha(SOURCE/'manifest.json'),selection_sha256=sha(SOURCE/'selection.json'),protocol_sha256=sha(ROOT/'TGS_CROSS_SCENE_PROTOCOL_V10.md'),v9_manifest_sha256=V9_SHA,prior_hash_checks=len(prior['files']),script_sha256=sha(Path(__file__)),common_sha256=sha(ROOT/'tgs_cross_scene_common_v10.py'),training=False,GT_quality_evaluated=False)
    target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,scenes=[r['scene'] for r in rows],prior_hash_checks=len(prior['files']))),flush=True)

if __name__=='__main__':main()
