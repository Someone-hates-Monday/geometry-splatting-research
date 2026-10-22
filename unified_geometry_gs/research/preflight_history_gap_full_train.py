"""Verify the actual complete train-only common state on CPU, without training."""
from pathlib import Path
import hashlib,json,time
import torch
from research.checkpoint_state import state_digest

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    r=ROOT/'research/experiments/factor_holdout55_v2/common'
    p=r/'chkpnt6999.pth';s=r/'chkpnt6999.runtime.pth'
    params,iteration=torch.load(p,map_location='cpu',weights_only=False)
    side=torch.load(s,map_location='cpu',weights_only=False)
    assert iteration==side['iteration']==6999
    assert sha(p)==side['checkpoint_sha256']
    digest=state_digest((params,side['supplemental']))
    assert digest==side['initial_state_sha256']
    opacity=params[7]
    assert not bool(torch.isnan(opacity).any() or torch.isneginf(opacity).any())
    names=[x[0] for x in side['supplemental']['train_camera_order']]
    assert len(names)==len(set(names))==42
    assert not set(names)&{f'{n:04d}' for n in range(0,49,8)}
    result=dict(status='complete',epoch=time.time(),iteration=iteration,
        checkpoint_sha256=sha(p),sidecar_sha256=sha(s),initial_state_sha256=digest,
        gaussian_count=len(params[2]),positive_inf_opacity=int(torch.isposinf(opacity).sum()),
        train_camera_order=names,camera_queue=side['supplemental']['camera_queue'],
        source_sha256=sha(Path(__file__)),scope='CPU complete checkpoint digest; runtime camera guards still required.')
    out=ROOT/'research/results/history_gap_full_train_preflight_v1.json'
    assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
