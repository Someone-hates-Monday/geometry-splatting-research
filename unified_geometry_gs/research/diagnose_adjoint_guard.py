"""Locate guarded/rejected refinements in the failing frozen backward camera."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.affine_rays_cpu import read_model,project_model,ray_parameters
from research.verify_adjoint_refinement import quantities,refine
from research.benchmark_median_cuda import load_config
from research.median_adjoint_backend import activate,load_backend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu",type=Path,required=True)
    parser.add_argument("--camera",required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Fresh output required.")
    ref=json.loads(args.cpu.read_text())
    model_path=Path(ref["model"])
    ply=model_path/"point_cloud/iteration_30000/point_cloud.ply"
    ply_hash=hashlib.sha256(ply.read_bytes()).hexdigest()
    if [v for k,v in ref["source_sha256"].items() if k.endswith("point_cloud.ply")]!=[ply_hash]:
        raise ValueError("Model hash changed.")
    cameras=json.loads((model_path/"cameras.json").read_text())
    camera=next(c for c in cameras if c["img_name"]==args.camera)
    projected=project_model(read_model(ply),camera)
    from scene import GaussianModel,Scene
    from gaussian_renderer import render
    dataset,pipe=load_config(model_path)
    rows=[]
    with tempfile.TemporaryDirectory(prefix="guard-scene-",dir=args.out.parent.resolve()) as temp:
        scratch=Path(temp)
        (scratch/"point_cloud").symlink_to(model_path/"point_cloud",target_is_directory=True)
        dataset.model_path=str(scratch)
        gaussians=GaussianModel(dataset.sh_degree,dataset.sg_degree)
        scene=Scene(dataset,gaussians,load_iteration=30000,shuffle=False)
        cam=next(c for c in scene.getTrainCameras() if c.image_name==args.camera)
        bg=torch.tensor([0.,0.,0.],device="cuda")
        with torch.no_grad():
            depths={}
            for variant in ("control","four"):
                with activate(load_backend(variant)):
                    depths[variant]=render(cam,gaussians,pipe,bg,dataset.kernel_size,require_depth=True)["median_depth"].squeeze().cpu().numpy()
        for r in ref["rows"]:
            if r["camera"]!=args.camera or not (r["finite_root"] and r["reference_fixed_bracket_ok"] and r["renderer_final_T_gate"] and not r["negative_camera_depth"]):
                continue
            x,y=r["pixel"]
            ray_length=np.sqrt(1+((x-cam.Cx)/cam.Fx)**2+((y-cam.Cy)/cam.Fy)**2)
            t0=float(depths["four"][y,x]*ray_length)
            t5=float(depths["control"][y,x]*ray_length)
            a,p,w,_=ray_parameters(projected,r["pixel"])
            T,ht,htt,num=quantities(t0,a,p,w)
            correction=(T-.5)/(T*ht)
            predicted=ht+htt*correction
            guarded=refine(t0,a,p,w)
            _,ht_true,_,num_true=quantities(r["median_ray_depth"],a,p,w)
            oracle=-num_true/ht_true
            _,ht5,_,num5=quantities(t5,a,p,w)
            g0=-num/ht
            g5=-num5/ht5
            distance=abs(t0-p)
            near=np.argmin(distance)
            crossing=distance<=abs(correction)
            cusp_coeff=a/(4*w*w*(1-a))
            predicted_gradient=-quantities(t0+correction,a,p,w)[3]/predicted
            rows.append({"pixel":[x,y],"contributors":len(a),"coarse_ray_depth":t0,
                "control_ray_depth":t5,"reference_ray_depth":r["median_ray_depth"],
                "newton_correction":float(correction),"nearest_peak_distance":float(distance[near]),
                "guard_accept":guarded is not None,"nearest_peak_alpha":float(a[near]),"nearest_peak_width":float(w[near]),
                "crossed_peaks":int(crossing.sum()),"crossing_cusp_relative_bound":float(2*cusp_coeff[crossing].sum()*abs(correction)/(.5*ht)),
                "coarse_effective_jacobian_error":float(np.linalg.norm(g0-oracle)/np.linalg.norm(oracle)),
                "five_effective_jacobian_error":float(np.linalg.norm(g5-oracle)/np.linalg.norm(oracle)),
                "unguarded_refined_effective_jacobian_error":float(np.linalg.norm(predicted_gradient-oracle)/np.linalg.norm(oracle)),
                "alpha":a.tolist(),"peaks":p.tolist(),"widths":w.tolist()})
    result={"model":str(model_path),"camera":args.camera,"rays":len(rows),
        "guard_rejections":sum(not r["guard_accept"] for r in rows),
        "rows":sorted(rows,key=lambda r:r["coarse_effective_jacobian_error"],reverse=True),
        "source_sha256":{str(ply):ply_hash,str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "limitations":["Independent float64 affine transcription; guard decisions can differ near float32 boundaries.",
            "Effective-ray Jacobian errors are mechanism diagnostics, not full Gaussian parameter gradient errors.",
            "Ungarded refinement is diagnostic only; no change to the production/experimental CUDA guard."]}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"rays":len(rows),"guard_rejections":result["guard_rejections"],
        "top":[{k:v for k,v in r.items() if k not in ("alpha","peaks","widths")} for r in result["rows"][:6]]},indent=2))


if __name__=="__main__":
    main()
