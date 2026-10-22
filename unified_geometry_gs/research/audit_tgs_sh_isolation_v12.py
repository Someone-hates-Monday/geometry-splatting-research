"""Independent source/control byte comparisons and exact quality audit on old/fresh cohort."""
import json,math
from pathlib import Path
import numpy as np
import torch
from tgs_sh_isolation_common_v12 import ROOT,OUT,VARIANTS,sha,exact
from audit_tgs_gt_quality_v11 import independent_ssim,patch

def main():
    target=OUT/'independent_audit.json';assert not target.exists();frozen=json.loads((OUT/'execution_plan.json').read_text());render=json.loads((OUT/'render_results.json').read_text());quality=json.loads((OUT/'quality_results.json').read_text());plan=json.loads((OUT/'input_plan.json').read_text());identity=json.loads((OUT/'identity_audit.json').read_text());assert identity['all_gates'] and len(render['rows'])==36 and len(quality['rows'])==20 and render['model_inputs_unchanged']
    for name,h in frozen['scripts'].items():assert sha(ROOT/name)==h
    for en in plan['source_edits']:
        assert sha(ROOT/en['original'])==en['original_sha256'] and sha(ROOT/en['generated'])==en['generated_sha256'];old='(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)';new='glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)';assert (ROOT/en['generated']).read_text()==(ROOT/en['original']).read_text().replace(old,new)
    for row in render['rows']:
        folder=OUT/row['scene']/row['cohort']/row['mode']/row['variant']
        for key,h in row['files'].items():assert sha(folder/(key+'.npy'))==h
    gates=[]
    for comp in render['comparisons']:
        base=OUT/comp['scene']/comp['cohort'];checks=[]
        for mode in ['main','degree0','precomputed']:
            keys=['aux','radii','maxblend','contributors']+([] if mode=='main' else ['rgb'])
            for key in keys:checks.append(exact(np.load(base/mode/'archived_native'/(key+'.npy')),np.load(base/mode/'sh_only'/(key+'.npy'))))
        orig=next(r for r in render['rows'] if r['scene']==comp['scene'] and r['cohort']==comp['cohort'] and r['mode']=='main' and r['variant']=='archived_native');fixed=next(r for r in render['rows'] if r['scene']==comp['scene'] and r['cohort']==comp['cohort'] and r['mode']=='main' and r['variant']=='sh_only');checks.append(orig['tile_instances']==fixed['tile_instances']);gate=all(checks);assert gate==comp['isolation_gate'];gates.append(gate)
    verified=[];patchcount=0
    for row in quality['rows']:
        base=OUT/row['scene']/row['cohort'];folder=base/'main'/row['variant'];gt=np.load(base/'gt_rgb.npy');truth=gt.astype(float)/255;raw=np.load(folder/'rgb.npy');p=torch.from_numpy(raw.copy()).mul(255).add_(.5).clamp_(0,255).permute(1,2,0).to(torch.uint8).numpy();assert exact(p,np.load(folder/'png_rgb.npy'));diff=p.astype(np.int64)-gt.astype(np.int64);ssq=int(np.sum(diff*diff,dtype=np.int64))
        for kind,x in [('primary_PNG8',p.astype(float)/255),('secondary_clipped_float',np.minimum(np.maximum(raw.transpose(1,2,0),0),1).astype(float))]:
            if kind=='primary_PNG8':mse=ssq/(gt.size*255**2)
            else:
                d=(x-truth).ravel();terms=[]
                for start in range(0,len(d),262139):a=d[start:start+262139];terms.append(float(np.einsum('i,i->',a,a)))
                mse=math.fsum(terms)/len(d)
            q=row[kind];psnr=-10*math.log10(mse) if mse else float('inf');assert math.isclose(mse,q['MSE'],rel_tol=1e-12,abs_tol=1e-16) and abs(psnr-q['PSNR_dB'])<=1e-10;ss,g=independent_ssim(x,truth);assert abs(float(ss.mean())-q['SSIM'])<=1e-10
            for sample in q['SSIM_samples']:
                xx,yy=sample['pixel']
                for c in range(3):assert abs(patch(x,truth,xx,yy,c,g)-sample['values'][c])<=1e-10;patchcount+=1
        verified.append(dict(scene=row['scene'],cohort=row['cohort'],variant=row['variant'],exact_PNG_squared_error=ssq));print(json.dumps(dict(scene=row['scene'],cohort=row['cohort'],variant=row['variant'],quality_verified=True)),flush=True)
    out=dict(all_verified=True,all_isolation_gates=all(gates),source_intervention_verified=True,noncolor_and_two_negative_controls_independently_verified=True,quality_metrics_independently_verified=True,SSIM_direct_patch_channel_checks=patchcount,rows=verified,script_sha256=sha(Path(__file__)),render_results_sha256=sha(OUT/'render_results.json'),quality_results_sha256=sha(OUT/'quality_results.json'),execution_plan_sha256=sha(OUT/'execution_plan.json'),training=False,scientific_goal_complete=False);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
