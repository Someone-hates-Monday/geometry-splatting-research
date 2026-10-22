"""Small analytic controls and independent Torch PNG conversion, before real scores."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from tgs_image_metrics_v11 import png8,score,gaussian
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_gt_quality_v11'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'metric_selfcheck.json';assert not target.exists();x=np.linspace(0,1,17*19*3,dtype=np.float64).reshape(17,19,3);identity=score(x,x);assert identity['MSE']==0 and identity['SSIM']==1;z=np.zeros_like(x);o=np.ones_like(x);opposite=score(z,o);assert opposite['MSE']==1 and opposite['PSNR_dB']==0;assert abs(gaussian().sum()-1)<1e-15
    vals=np.r_[np.linspace(-.2,1.2,3000,dtype=np.float32),(np.arange(256,dtype=np.float32)+np.float32(.5))/np.float32(255)];raw=np.resize(vals,(3,34,35)).astype(np.float32);expected=torch.from_numpy(raw.copy()).mul(255).add_(.5).clamp_(0,255).permute(1,2,0).to(torch.uint8).numpy();assert np.array_equal(png8(raw),expected)
    out=dict(all_verified=True,identical_image_MSE=0,identical_image_SSIM=1,zero_vs_one_MSE=1,zero_vs_one_PSNR_dB=0,PNG_quantization_bitwise_Torch=True,metrics_sha256=sha(ROOT/'tgs_image_metrics_v11.py'),script_sha256=sha(Path(__file__)),torch=torch.__version__,training=False,real_quality_scores_computed=False);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
