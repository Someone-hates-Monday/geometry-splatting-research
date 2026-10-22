"""Unchanged v11 quality definitions; source edit effects and subsequent residuals."""
import json
from pathlib import Path
import numpy as np
from tgs_sh_isolation_common_v12 import ROOT,OUT,VARIANTS,sha
from tgs_image_metrics_v11 import png8,score

def main():
    target=OUT/'quality_results.json';assert not target.exists();render=json.loads((OUT/'render_results.json').read_text());frozen=json.loads((OUT/'execution_plan.json').read_text())
    for name,h in frozen['scripts'].items():assert sha(ROOT/name)==h
    rows=[];contrasts=[]
    for comp in render['comparisons']:
        scene,cohort=comp['scene'],comp['cohort'];base=OUT/scene/cohort;gt=np.load(base/'gt_rgb.npy').astype(float)/255;case=[]
        for v in VARIANTS:
            folder=base/'main'/v;raw=np.load(folder/'rgb.npy');p=png8(raw);clipped=np.clip(raw.transpose(1,2,0),0,1).astype(float);np.save(folder/'png_rgb.npy',p,allow_pickle=False);np.save(folder/'clipped_rgb.npy',clipped,allow_pickle=False);row=dict(scene=scene,cohort=cohort,camera=comp['camera'],variant=v,primary_PNG8=score(p.astype(float)/255,gt),secondary_clipped_float=score(clipped,gt));case.append(row);rows.append(row);print(json.dumps(dict(scene=scene,cohort=cohort,variant=v,PSNR=row['primary_PNG8']['PSNR_dB'],SSIM=row['primary_PNG8']['SSIM'])),flush=True)
        by={r['variant']:r for r in case};cs={}
        for label,num,den in [('sh_only_minus_original','sh_only','archived_native'),('coarse_minus_sh_only','coarse','sh_only'),('geometry_minus_sh_only','geometry_fixed','sh_only'),('piecewise_minus_geometry','piecewise','geometry_fixed')]:cs[label]={kind:{key:by[num][kind][key]-by[den][kind][key] for key in ['MSE','PSNR_dB','SSIM']} for kind in ['primary_PNG8','secondary_clipped_float']}
        contrasts.append(dict(scene=scene,cohort=cohort,camera=comp['camera'],isolation_gate=comp['isolation_gate'],contrasts=cs))
    target.write_text(json.dumps(dict(rows=rows,contrasts=contrasts,render_results_sha256=sha(OUT/'render_results.json'),script_sha256=sha(Path(__file__)),metrics_sha256=sha(ROOT/'tgs_image_metrics_v11.py'),execution_plan_sha256=sha(OUT/'execution_plan.json'),training=False,training_effect_identified=False),indent=2)+'\n')
if __name__=='__main__':main()
