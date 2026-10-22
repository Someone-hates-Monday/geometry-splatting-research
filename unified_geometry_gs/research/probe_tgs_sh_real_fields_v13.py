"""CPU-only shell transport of presampled real public SH fields; no image/training claim."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_sh_shell_v13/real_fields'
SOURCE=ROOT/'sources/tgs_cross_scene_v10'
HELPER=ROOT/'sources/tgs_real_checkpoint_v1/source/utils/sh_utils.py'
PROTOCOL=ROOT/'TGS_SH_REAL_FIELD_PROTOCOL_V13.md'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save_new(p,obj):
    with p.open('x') as f:f.write(json.dumps(obj,indent=2)+'\n')


def prepare():
    OUT.mkdir(exist_ok=True)
    manifests=json.loads((SOURCE/'manifest.json').read_text())['files']
    files=[]
    for scene in ['flowers','room']:
        rel=f'sources/tgs_cross_scene_v10/{scene}/point_cloud/iteration_7000/point_cloud_state_dict.pt'
        expected=next(e for e in manifests if e['local']==rel)
        assert sha(ROOT/rel)==expected['sha256']
        files.append(dict(scene=scene,path=rel,sha256=expected['sha256']))
    save_new(OUT/'plan.json',dict(protocol_sha256=sha(PROTOCOL),script_sha256=sha(Path(__file__)),
        helper_sha256=sha(HELPER),models=files,seed=20261021,sample_size_per_scene=64,
        rho=[.01,.05,.1,.25],axes=['x','y','z'],grids=[[32,64],[37,71],[48,96]],
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,training=False))
    print(json.dumps(dict(stage='plan_frozen',path=str(OUT/'plan.json'))),flush=True)


def run():
    import numpy as np
    import scipy
    from scipy.special import sph_harm_y
    import torch
    torch.set_num_threads(2)
    plan=json.loads((OUT/'plan.json').read_text())
    assert plan['script_sha256']==sha(Path(__file__)) and plan['helper_sha256']==sha(HELPER)
    assert plan['protocol_sha256']==sha(PROTOCOL) and not (OUT/'results.json').exists()
    spec=importlib.util.spec_from_file_location('pinned_sh_cpu_v13',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

    def grid(nz,nphi,offset):
        z,w=np.polynomial.legendre.leggauss(nz)
        phi=(np.arange(nphi)+offset)*2*np.pi/nphi
        zz,pp=np.meshgrid(z,phi,indexing='ij')
        rr=np.sqrt(1-zz*zz)
        d=np.stack([rr*np.cos(pp),rr*np.sin(pp),zz],axis=-1).reshape(-1,3)
        weights=np.repeat(w/(2*nphi),nphi)
        return d,weights

    def basis(d,degree=4):return helper.eval_sh(degree,np.eye((degree+1)**2),d)
    grids=[grid(32,64,.5),grid(37,71,.25),grid(48,96,.125)]
    bases=[basis(d) for d,w in grids]
    for (d,w),Y in zip(grids,bases):
        theta=np.arccos(d[:,2]);phi=np.arctan2(d[:,1],d[:,0]);independent=[]
        for ell in range(5):
            for m in range(-ell,ell+1):
                yy=sph_harm_y(ell,abs(m),theta,phi)
                independent.append(yy.real if m==0 else np.sqrt(2)*(yy.imag if m<0 else yy.real))
        assert np.max(np.abs(Y-np.stack(independent,axis=1)))<=2e-13
        assert np.max(np.abs(4*np.pi*(Y.T*w)@Y-np.eye(25)))<=2e-13
    rows=[]
    for model in plan['models']:
        path=ROOT/model['path'];assert sha(path)==model['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        count=len(state['features_dc'])
        ids=np.sort(np.random.default_rng(plan['seed']).choice(count,64,replace=False))
        # IDs/coefficients are saved before this scene's field outputs exist.
        save_new(OUT/(model['scene']+'_selection.json'),dict(ids=ids.tolist(),count=count,
            selection='uniform PCG64 without replacement; no filtering',seed=plan['seed']))
        A=torch.cat([state['features_dc'][ids],state['features_rest'][ids]],1).numpy().astype(np.float64)
        assert A.shape==(64,16,3) and np.isfinite(A).all()
        np.save(OUT/(model['scene']+'_coefficients.npy'),A,allow_pickle=False)
        del state
        flat=A.transpose(1,0,2).reshape(16,-1)
        YY=bases[1][:,:16];copy=(YY@flat+.5).reshape(-1,64,3)
        bnorm=sum(np.linalg.norm(A[:,ell*ell:(ell+1)**2,:],axis=1)*
            np.sqrt(ell*(ell+1)*(2*ell+1)/(8*np.pi)) for ell in [1,2,3])
        zero=4*np.pi*(bases[0].T*grids[0][1])@(bases[0][:,:16]@flat)
        assert np.max(np.abs(zero[:16]-flat))<=2e-11 and np.max(np.abs(zero[16:]))<=2e-11
        for axis in range(3):
            v=np.eye(3)[axis]
            for rho in plan['rho']:
                targets=[]
                for d,w in grids:
                    e=d+rho*v;e/=np.linalg.norm(e,axis=1,keepdims=True)
                    targets.append(basis(e,3)@flat)
                coeffs=[4*np.pi*(bases[g].T*grids[g][1])@targets[g] for g in [0,2]]
                convergence=float(np.max(np.abs(coeffs[0]-coeffs[1])))
                assert convergence<=1e-8
                truth=(targets[1]+.5).reshape(-1,64,3)
                preds={'copy':copy}
                for degree in [3,4]:
                    for idx,name in enumerate(['fit32','fit48']):
                        coef=coeffs[idx][: (degree+1)**2]
                        preds[f'L{degree}_{name}']=(bases[1][:,:len(coef)]@coef+.5).reshape(-1,64,3)
                bound=bnorm*np.arcsin(rho)
                maxcopy=np.max(np.abs(copy-truth),axis=0)
                assert np.all(maxcopy<=bound+1e-10+1e-10*bound)
                for i,primitive in enumerate(ids):
                    metrics={}
                    for name,pred in preds.items():
                        diff=pred[:,i]-truth[:,i]
                        clipped=np.maximum(pred[:,i],0)-np.maximum(truth[:,i],0)
                        weights=grids[1][1]
                        metrics[name]=dict(raw_RMS=float(np.sqrt(np.sum(weights[:,None]*diff**2)/3)),
                            lower_clamped_RMS=float(np.sqrt(np.sum(weights[:,None]*clipped**2)/3)),
                            grid_max_abs=float(np.max(np.abs(diff))))
                    rows.append(dict(scene=model['scene'],primitive=int(primitive),axis=plan['axes'][axis],
                        relative_displacement=rho,metrics=metrics,coefficient_convergence_max=convergence,
                        copy_color_bound_per_channel=bound[i].tolist(),
                        L4_band_coefficient_norm=float(np.linalg.norm(coeffs[0][16:,i*3:i*3+3]))))
                np.savez(OUT/f"{model['scene']}_{plan['axes'][axis]}_{rho}_projection.npz",
                         fit32=coeffs[0].reshape(25,64,3),fit48=coeffs[1].reshape(25,64,3))
                print(json.dumps(dict(scene=model['scene'],axis=plan['axes'][axis],rho=rho,
                                      convergence=convergence,cases=len(rows))),flush=True)
        assert sha(path)==model['sha256']
    assert len(rows)==1536
    save_new(OUT/'results.json',dict(rows=rows,plan_sha256=sha(OUT/'plan.json'),
        script_sha256=sha(Path(__file__)),helper_sha256=sha(HELPER),numpy=np.__version__,
        scipy=scipy.__version__,torch=torch.__version__,all_defined_checks_pass=True,
        camera_shell_is_constructed=True,real_coefficients=True,real_image_metrics=False,
        CPU_only=True,training=False,novelty_established=False))
    print(json.dumps(dict(stage='complete',cases=len(rows),all_defined_checks_pass=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
