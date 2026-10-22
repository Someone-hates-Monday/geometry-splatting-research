"""Validate real published tensors/cameras on CPU, before any scene rendering."""
import ast
import hashlib
import json
import math
import re
import struct
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'sources/tgs_real_checkpoint_v1'
OUT=ROOT/'experiments/tgs_real_event_v1'


def cfg_parse(text):
    node=ast.parse(text.strip(),mode='eval').body
    assert isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Namespace'
    assert not node.args and all(x.arg is not None for x in node.keywords)
    return {x.arg:ast.literal_eval(x.value) for x in node.keywords}


def load_input():
    selection=json.loads((INPUT/'selection.json').read_text())
    manifest=json.loads((INPUT/'manifest.json').read_text())
    for item in manifest['files']:
        p=ROOT/item['local'].replace('\\','/')
        assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    scene=INPUT/selection['scene']
    directory=scene/'point_cloud'/selection['iteration']
    model=torch.load(directory/'point_cloud_state_dict.pt',map_location='cpu',weights_only=True)
    # Official hyperparameters include a NumPy scalar. Permit only its known
    # dtype constructors, keeping the restricted weights unpickler enabled.
    unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(directory/'hyperparameters.pt')
    assert set(unsafe)<={'numpy._core.multiarray.scalar','numpy.core.multiarray.scalar','numpy.dtype'}
    with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,
                                           type(np.dtype('float64')),type(np.dtype('float32'))]):
        hyper=torch.load(directory/'hyperparameters.pt',map_location='cpu',weights_only=True)
    cfg=cfg_parse((scene/'cfg_args').read_text())
    cameras=json.loads((scene/'cameras.json').read_text())
    return model,hyper,cfg,cameras,selection


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    model,hyper,cfg,cameras,selection=load_input()
    required={'triangles_points','sigma','active_sh_degree','features_dc','features_rest','opacity'}
    assert set(model)==required
    n=len(model['triangles_points'])
    assert tuple(model['triangles_points'].shape)==(n,3,3)
    assert tuple(model['sigma'].shape)==tuple(model['opacity'].shape)==(n,1)
    assert tuple(model['features_dc'].shape)==(n,1,3)
    degree=int(model['active_sh_degree'])
    assert degree in [0,1,2,3]
    assert model['features_rest'].shape[0]==n and model['features_rest'].shape[2]==3
    assert model['features_rest'].shape[1]+1>=(degree+1)**2
    tensors={}
    for key,value in model.items():
        if isinstance(value,torch.Tensor):
            value=value.detach()
            assert value.dtype==torch.float32 and torch.isfinite(value).all()
            tensors[key]={'shape':list(value.shape),'dtype':str(value.dtype),
                          'min':float(value.min()),'max':float(value.max())}
    sigma=.01+torch.exp(model['sigma'].detach())
    opacity=torch.sigmoid(model['opacity'].detach())
    assert torch.isfinite(sigma).all() and (sigma>=.01).all()
    names=[x['img_name'] for x in cameras]
    assert len(set(names))==len(names) and cfg['eval']
    sorted_cameras=sorted(cameras,key=lambda c:c['img_name'])
    test=[x for i,x in enumerate(sorted_cameras) if i%8==0]
    train=[x for i,x in enumerate(sorted_cameras) if i%8!=0]
    # Scene.__init__ serializes test first then train, before shuffling.
    assert names==[x['img_name'] for x in test+train]
    errors=[]
    for c in cameras:
        rot=np.array(c['rotation']);pos=np.array(c['position'])
        assert rot.shape==(3,3) and pos.shape==(3,) and np.isfinite(pos).all()
        error=float(np.abs(rot.T@rot-np.eye(3)).max())
        assert error<1e-5 and abs(np.linalg.det(rot)-1)<1e-5
        assert c['width']>0 and c['height']>0 and c['fx']>0 and c['fy']>0
        errors.append(error)
    chosen=[]
    for label,group in [('test',test),('train',train)]:
        ordered=sorted(group,key=lambda c:hashlib.sha256(('tgs_real_event_v1/'+label+'/'+c['img_name']).encode()).hexdigest())
        chosen.extend([{**x,'split':label} for x in ordered[:2]])
    gt=list((INPUT/selection['scene']/'test').rglob('gt/*.png'))
    assert len(gt)==1
    header=gt[0].read_bytes()[:24]
    assert header[:8]==b'\x89PNG\r\n\x1a\n'
    gt_size=struct.unpack('>II',header[16:24])
    # CameraInfo.width/height are COLMAP dimensions; its image may already be
    # downsampled by the configured images_4 folder. Do not apply -r=-1 to
    # the original dimensions. FoV still uses original width/focal ratios.
    factor=re.fullmatch(r'images_(\d+)',cfg['images'])
    assert factor and cfg['resolution']==-1 and gt_size[0]<=1600
    downsample=int(factor[1])
    assert set((math.ceil(c['width']/downsample),math.ceil(c['height']/downsample)) for c in cameras)=={gt_size}
    report={'passed':True,'scene':selection['scene'],'iteration':selection['iteration'],
            'triangles':n,'active_sh_degree':degree,'tensor_metadata':tensors,'cfg_args':cfg,
            'sigma_activation':'0.01 + exp(raw_sigma)',
            'sigma_range':[float(sigma.min()),float(sigma.max())],
            'opacity_range':[float(opacity.min()),float(opacity.max())],
            'mask_semantics':'not saved in model dict; official load() resets raw mask to ones; effective forward STE mask is 1',
            'hyperparameter_keys':sorted(hyper),
            'camera_count':len(cameras),'train_count':len(train),'test_count':len(test),
            'split_verified':'sorted names modulo8; serialized test+train order matched pinned loader',
            'max_rotation_orthogonality_error':max(errors),'actual_resolution':list(gt_size),
            'resolution_evidence':'one published GT PNG header; all COLMAP dimensions and images_4 ceiling sizes agree',
            'resolution_for_train_cameras':'same declared scene resolution, not individually decoded training images',
            'camera_selection_rule':'first two SHA256 ranks per split; pre-rendering',
            'selected_cameras':chosen,'event_count_cap':8,
            'event_selection_rule':'first eight SHA256 ranks among active finite parents with native radius>24 in at least one selected camera',
            'event_label':'offline midpoint-split simulation on real checkpoint; not captured training events',
            'training_commit_proven':False,'optimizer_state_available':False,
            'full_training_authorized_or_started':False}
    (OUT/'input_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','scene','iteration','triangles','active_sh_degree','actual_resolution','camera_count','train_count','test_count','sigma_range']}),flush=True)


if __name__=='__main__':
    main()
