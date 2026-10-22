"""Official camera-center alignment with scratch Scene metadata and fresh output."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from research.benchmark_median_cuda import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    model=args.model.resolve()
    target=model/'recon_aligned.ply'
    if target.exists():
        parser.error('Fresh aligned mesh required.')
    from scene import GaussianModel,Scene
    from evaluate_dtu_mesh import best_fit_transform,load_dtu_camera
    ds,pipe=load_config(model)
    gt_centers=np.asarray([p[:,3] for p in load_dtu_camera(str(ROOT/'dtu_eval/Offical_DTU_Dataset'))])
    with tempfile.TemporaryDirectory(prefix='align-scene-',dir=model.parent) as temp:
        scratch=Path(temp)
        (scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True)
        ds.model_path=str(scratch)
        gs=GaussianModel(ds.sh_degree,ds.sg_degree)
        scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
        centers=np.asarray([(c.world_view_transform.T).inverse()[:3,3].cpu().numpy() for c in scene.getTrainCameras()])
        truth=gt_centers[:len(centers)]
        scale=np.linalg.norm(truth-truth.mean(0),axis=1).mean()/np.linalg.norm(centers-centers.mean(0),axis=1).mean()
        _,rotation,translation=best_fit_transform(centers*scale,truth)
    source=model/'recon_post.ply'
    mesh=trimesh.load(source,process=True)
    mesh.vertices=mesh.vertices*scale @ rotation.T+translation
    mesh.export(target)
    record={'scale':float(scale),'rotation':rotation.tolist(),'translation':translation.tolist(),
        'source_mesh_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'aligned_mesh_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'evaluate_dtu_mesh.py')},
        'convention':'Same ordered train camera centers and first N official DTU calibration centers as historical evaluator.'}
    (model/'alignment.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
