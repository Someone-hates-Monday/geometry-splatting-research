"""V2 preserves all real derivatives; nonzero-kernel root finite differences."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from research.world_root_vjp_v2 import torch_fields,torch_vjp,polish
from research.benchmark_csr_decode import read,save,sha,now
R=Path(__file__).resolve().parent;BASE=R/'experiments/world_root_vjp_v1';OUT=R/'results/world_root_vjp_v2_checks.json'
def main():
    assert not OUT.exists();torch.set_num_threads(1);s=read(BASE/'summary.json');real=0;different=[]
    for f in sorted(BASE.glob('scan*_gradients.npz')):
        stem=f.name.removesuffix('_gradients.npz');v=read(BASE/(stem+'.json'));assert v['meta']['kernel']==0
        with np.load(f) as d:source=d['source'];raw=d['raw'];sections=d['sections'];saved=d['factor_refined'];pixels=d['pixels']
        for j,row in enumerate(v['rows']):
            a,b=map(int,sections[j:j+2]);g,info=torch_vjp(raw[a:b],source[a:b],v['meta'],row['branches']['factor_refined']['t'],'factor')
            if not np.array_equal(g.view(np.uint64),saved[a:b].view(np.uint64)):different.append(dict(view=stem,pixel=int(pixels[j]),max_abs=float(np.max(abs(g-saved[a:b])))))
            real+=1
    assert real==891 and not different
    source=np.zeros((3,36));source[:,:3]=np.array([[.4,-.2,2],[.61,-.32,3],[.8,-.4,4]],np.float32).astype(float);source[:,3:6]=np.array([.08,.09,.11],np.float32).astype(float);source[:,6:10]=np.array([1.,.1,.05,-.02],np.float32);source[:,6:10]/=np.linalg.norm(source[:,6:10],axis=1)[:,None];source[:,10]=1;source[:,20:29]=np.eye(3).reshape(1,9);source[:,32:35]=[.2,-.1,1];source[:,35]=[.6,.35,.2];raw=np.zeros((3,12));raw[:,:3]=source[:,:3];raw[:,3:6]=np.log(source[:,3:6]);raw[:,6:10]=[1,.1,.05,-.02];raw[:,10]=[.8,-.4,-1.2];raw[:,11]=.006;theta=torch.tensor(raw[:,:11],dtype=torch.float64);rng=np.random.default_rng(13017);direction=rng.normal(size=(3,11));direction/=np.linalg.norm(direction);checks=[]
    for kernel in [0.,.3,1.]:
        meta=dict(fx=300.,fy=310.,limx=.8,limy=.6,kernel=kernel,floor=float(np.float32(1e-6)));fields=torch_fields(theta,raw,source,meta,'factor');assert np.array_equal(fields[3].numpy(),source[:,35]);t,status,steps=polish(fields,1.8,2.5);assert t is not None;g,info=torch_vjp(raw,source,meta,t,'factor');predicted=float(np.sum(g*direction));errors=[]
        for eps in [1e-4,3e-5,1e-5]:
            roots=[]
            for sign in [-1,1]:
                varied=theta+torch.tensor(sign*eps*direction,dtype=torch.float64);cur= torch_fields(varied,raw,source,meta,'factor');root,st,n=polish(cur,1.8,2.5);assert root is not None;roots.append(root)
            qnorm=float(fields[5]);fd=(roots[1]-roots[0])/(2*eps*qnorm);errors.append(dict(epsilon=eps,finite_difference=fd,relative_error=abs(fd-predicted)/max(abs(predicted),1e-30)))
        assert errors[-1]['relative_error']<=1e-5;checks.append(dict(synthetic=True,kernel=kernel,root=t,predicted_directional_z_derivative=predicted,finite_differences=errors))
    result=dict(status='complete',checked_utc=now(),all891_real_refined_VJP_bits_unchanged=True,real_mismatches=different,synthetic_nonzero_kernel_checks=checks,gate_pass=True,sources_sha256={p.name:sha(p) for p in [Path(__file__),R/'world_root_vjp.py',R/'world_root_vjp_v2.py',R/'build_world_root_vjp_v2.py']},scope='V1 real data kernel0 remains unchanged; V2 fixes frozen screen baseline for grad-disabled perturbed forward, needed by finite differences. Nonzero-kernel cases synthetic, not reconstruction/training evidence.')
    save(OUT,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
