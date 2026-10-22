"""No-fit CPU SH directional rank/projection diagnostic on actual selected points."""
import importlib.util,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_joint_policy_data_v29 import ROOT,OUT,sha,write
def main():
    choice=json.loads((OUT/'choices.json').read_text());p=json.loads((OUT/'input_plan.json').read_text());helper_path=ROOT/'sources/tgs_real_checkpoint_v1/source/utils/sh_utils.py';spec=importlib.util.spec_from_file_location('pinned_SH_observability',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    deps=[Path(__file__),helper_path,OUT/'choices.json',OUT/'input_plan.json',OUT/'execution_plan.json',ROOT/'TGS_SH_OBSERVABILITY_NOTE_V29.md'];models={s['scene']:ROOT/f"sources/tgs_cross_scene_v10/{s['scene']}/point_cloud/iteration_7000/point_cloud_state_dict.pt" for s in choice['scenes']};deps.extend(models.values());frozen=json.loads((OUT/'execution_plan.json').read_text())['inputs']
    for path in models.values():assert sha(path)==frozen[path.relative_to(ROOT).as_posix()]
    write(OUT/'SH_observability_plan.json',dict(inputs={path.relative_to(ROOT).as_posix():sha(path) for path in deps},post_selection=True,new_native_forwards=0,fitting=False,GT_used=False,ideal_float64_field_not_native_bits=True,svd_tolerance='max(shape)*float64eps*largest_singular_value'))
    scenes=[]
    for sc in choice['scenes']:
        scene=sc['scene'];model=torch.load(models[scene],map_location='cpu',weights_only=True);points=model['triangles_points'].detach().numpy();metadata=[r for r in p['rows'] if r['scene']==scene];centers=np.array([r['camera']['position'] for r in metadata],dtype=np.float64);assert [r['cohort'] for r in metadata]==['calibration','calibration','heldout','heldout'];rows=[]
        for parent in sc['selected']['signed_mixed']:
            triangle=points[parent].astype(np.float32);a,b,c=triangle;ab,ac,bc=(a+b)/np.float32(2),(a+c)/np.float32(2),(b+c)/np.float32(2);triangles=[triangle,np.stack([a,ab,ac]),np.stack([b,ab,bc]),np.stack([c,ac,bc]),np.stack([ab,ac,bc])]
            for ti,tri in enumerate(triangles):
                centroid=tri.astype(np.float64).mean(0);dirs=centroid[None]-centers;dirs/=np.linalg.norm(dirs,axis=1)[:,None];Y=helper.eval_sh(3,np.eye(16,dtype=np.float64)[None],dirs);assert Y.shape==(4,16)
                u,s,vt=np.linalg.svd(Y[:2],full_matrices=False);tol=max(Y[:2].shape)*np.finfo(float).eps*s[0];rank=int(np.count_nonzero(s>tol));V=vt[:rank];tests=[]
                for v in [2,3]:
                    projected=Y[v]-V.T@(V@Y[v]);rho=float(np.linalg.norm(projected));z=projected/rho if rho>1e-14 else np.zeros(16);residual=float(np.max(np.abs(Y[:2]@z)));assert residual<1e-12
                    tests.append(dict(view=metadata[v]['view'],nullspace_projection_norm=rho,projection_fraction_of_basis_norm=rho/float(np.linalg.norm(Y[v])),unit_invisible_direction=z.tolist(),calibration_invisibility_residual=residual))
                rows.append(dict(parent_id=parent,primitive='parent' if ti==0 else f'child_{ti-1}',directional_design_rank=rank,nullspace_dimension=16-rank,singular_values=s.tolist(),rank_tolerance=float(tol),tests=tests))
        assert len(rows)==40;scenes.append(dict(scene=scene,rows=rows,min_nullspace_dimension=min(r['nullspace_dimension'] for r in rows),median_test_projection_fraction=float(np.median([v['projection_fraction_of_basis_norm'] for r in rows for v in r['tests']])),all_design_ranks_two=all(r['directional_design_rank']==2 for r in rows)));del model,points
    write(OUT/'SH_observability_results.json',dict(scenes=scenes,plan_sha256=sha(OUT/'SH_observability_plan.json'),GT_used=False,fitting=False,choices_changed=False,new_native_forwards=0,training=False,actual_image_Jacobian_rank_measured=False))
    print(json.dumps(dict(stage='SH_DIRECTIONAL_OBSERVABILITY_DONE',scenes=[{k:s[k] for k in ['scene','min_nullspace_dimension','median_test_projection_fraction','all_design_ranks_two']} for s in scenes])),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        with (OUT/'SH_observability_failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
