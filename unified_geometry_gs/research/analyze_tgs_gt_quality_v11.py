"""Prespecified per-view image metrics and differences; all four operators retained."""
import hashlib,json
from pathlib import Path
import numpy as np
import scipy
from tgs_image_metrics_v11 import png8,score
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_gt_quality_v11'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'quality_results.json';assert not target.exists();plan=json.loads((OUT/'quality_plan.json').read_text());render=json.loads((OUT/'render_results.json').read_text());assert render['model_inputs_unchanged']
    for name,h in plan['scripts'].items():assert sha(ROOT/name)==h
    rows=[]
    for scene in ['flowers','room']:
        gt=np.load(OUT/scene/'gt_rgb.npy');truth=gt.astype(np.float64)/255
        for v in ['oracle','coarse','piecewise','archived_native']:
            folder=OUT/scene/v;raw=np.load(folder/'rgb.npy');eight=png8(raw);clipped=np.clip(raw.transpose(1,2,0),0,1).astype(np.float64);np.save(folder/'png_rgb.npy',eight,allow_pickle=False);np.save(folder/'clipped_rgb.npy',clipped,allow_pickle=False);primary=score(eight.astype(np.float64)/255,truth);secondary=score(clipped,truth);row=dict(scene=scene,variant=v,primary_PNG8=primary,secondary_clipped_float=secondary);rows.append(row);print(json.dumps(dict(scene=scene,variant=v,PNG_PSNR=primary['PSNR_dB'],PNG_SSIM=primary['SSIM'],float_PSNR=secondary['PSNR_dB'],float_SSIM=secondary['SSIM'])),flush=True)
        base=np.load(OUT/scene/'oracle/rgb.npy');coarse=np.load(OUT/scene/'coarse/rgb.npy');assert np.array_equal(base.view(np.uint8),coarse.view(np.uint8))
        for v in ['piecewise','archived_native']:
            candidate=np.load(OUT/scene/v/'rgb.npy');np.save(OUT/scene/(v+'_minus_oracle_RGB.npy'),candidate.astype(np.float64)-base.astype(np.float64),allow_pickle=False)
    contrasts=[]
    for scene in ['flowers','room']:
        by={r['variant']:r for r in rows if r['scene']==scene};contrasts.append(dict(scene=scene,coarse_minus_piecewise={kind:{key:by['coarse'][kind][key]-by['piecewise'][kind][key] for key in ['MSE','PSNR_dB','SSIM']} for kind in ['primary_PNG8','secondary_clipped_float']},coarse_minus_archived_native={kind:{key:by['coarse'][kind][key]-by['archived_native'][kind][key] for key in ['MSE','PSNR_dB','SSIM']} for kind in ['primary_PNG8','secondary_clipped_float']}))
    out=dict(rows=rows,contrasts=contrasts,script_sha256=sha(Path(__file__)),metrics_sha256=sha(ROOT/'tgs_image_metrics_v11.py'),plan_sha256=sha(OUT/'quality_plan.json'),render_results_sha256=sha(OUT/'render_results.json'),numpy=np.__version__,scipy=scipy.__version__,training=False,complete_test_set=False,causal_training_effect_estimated=False);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(contrasts=contrasts)),flush=True)
if __name__=='__main__':main()
