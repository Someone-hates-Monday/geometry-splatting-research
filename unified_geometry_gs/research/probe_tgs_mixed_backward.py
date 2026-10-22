"""Fixed three-layer RGB/SH VJP checks; no optimizer or training.

Inputs, probes and numerical gates are written before the first CUDA call.
All perturbations are retained. Loading cached modules does not mutate prior
sealed receipts. A failing scene stays in the result and makes its gate false.
"""
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from tgs_mixed_reference import ROOT, OUT, MODULES, Native, fixtures, load_module, reference

STEPS = [.01, .003, .001]


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),1e-12))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def directions(f):
    size, n = len(f['theta']), f['n']
    probes = [(f'coordinate_{j}', np.eye(1,size,j)[0]) for j in range(11*n)]
    if f['degree'] >= 0:
        k = (f['degree']+1)**2
        for i in range(n):
            for b,c in [(0,0),(k-1,2)]:
                j = 11*n+i*k*3+b*3+c
                probes.append((f'sh_coordinate_{i}_{b}_{c}',np.eye(1,size,j)[0]))
    rng = np.random.default_rng(92132026)
    for i in range(8):
        d = rng.normal(size=size)
        d /= np.linalg.norm(d)
        probes.append((f'dense_{i}',d))
    return probes


def prepare():
    assert not OUT.exists(), 'Existing experiment must be audited or explicitly versioned, never overwritten'
    OUT.mkdir(parents=True)
    plan = dict(scope='three fixed synthetic layers, RGB VJP, nonzero background, degree1/3 SH; no training',
                steps=STEPS, modules=MODULES, scenes=[],
                gates=dict(fixed_group_relative_error=1e-3, forward_max_abs=3e-5,
                           finite_difference_step=.003, native_fd_relative_error=1e-3,
                           cpu_fd_relative_error=1e-4, every_branch_unchanged=True),
                script_sha256=sha(Path(__file__)), reference_sha256=sha(ROOT/'tgs_mixed_reference.py'))
    for f in fixtures():
        x = torch.tensor(f['theta'].astype(float),requires_grad=True)
        loss,rgb,meta = reference(x,f)
        assert meta['edge_gap_min'] > 2 and meta['depth_gap_min'] > .2
        assert np.min(meta['raw_alpha']) > .02 and np.max(meta['raw_alpha']) < .9
        assert np.min(meta['raw_colors']) > .1
        assert np.min(meta['transmittance']) > .01
        assert np.all(meta['fingerprint']['included']) and not np.any(meta['fingerprint']['terminated'])
        ds = directions(f)
        path = OUT/(f['name']+'_input.npz')
        np.savez_compressed(path,**f,probe_directions=np.array([d for _,d in ds]),
                            probe_names=np.array([name for name,_ in ds]))
        plan['scenes'].append(dict(name=f['name'],input_sha256=sha(path),
                                   base_cpu_loss=float(loss.detach()),base_metadata=meta,
                                   dimensions=len(x),probes=len(ds)))
    (OUT/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(stage='frozen_before_cuda',scenes=len(plan['scenes']),
                          plan_sha256=sha(OUT/'protocol.json'))),flush=True)
    return plan


def main():
    start=time.time()
    plan=prepare()
    records=[]
    for variant in ['control','fixed']:
        mod=load_module(variant)
        for scene in plan['scenes']:
            path=OUT/(scene['name']+'_input.npz')
            assert sha(path)==scene['input_sha256']
            with np.load(path) as z:
                f={k:z[k].copy() for k in z.files}
            for key in ['n','w','h','degree']:
                f[key]=int(f[key])
            native=Native(mod,f)
            x=torch.tensor(f['theta'].astype(float),requires_grad=True)
            cpu_loss,cpu_rgb,meta=reference(x,f)
            exact=torch.autograd.grad(cpu_loss,x)[0].numpy()
            data=native.forward(f['theta'])
            gpu,arrays=native.backward(data)
            repeats=[native.backward(native.forward(f['theta']))[0] for _ in range(2)]
            repeat_max=max(float(np.max(np.abs(g-gpu))) for g in repeats)
            fd_records=[]
            for h in plan['steps']:
                rows=[]
                for label,d in zip(f['probe_names'],f['probe_directions']):
                    plus=(f['theta'].astype(float)+h*d).astype(np.float32)
                    minus=(f['theta'].astype(float)-h*d).astype(np.float32)
                    actual_direction=(plus.astype(float)-minus.astype(float))/(2*h)
                    # Float32 rounding makes dense perturbations slightly
                    # non-collinear. Compare the actual, fully retained path.
                    expected=float(exact@actual_direction)
                    lp,rp,_,_,_,sp=native.forward(plus)
                    lm,rm,_,_,_,sm=native.forward(minus)
                    cp,_,mp=reference(torch.tensor(plus.astype(float)),f)
                    cm,_,mm=reference(torch.tensor(minus.astype(float)),f)
                    smooth=(mp['fingerprint']==mm['fingerprint']==meta['fingerprint']
                            and min(mp['edge_gap_min'],mm['edge_gap_min'])>1
                            and min(mp['depth_gap_min'],mm['depth_gap_min'])>.1
                            and bool((rp[3]>0).all()) and bool((rm[3]>0).all()))
                    rows.append(dict(probe=str(label),expected=expected,native_fd=(lp-lm)/(2*h),
                                     cpu_fd=float(cp-cm)/(2*h),native_plus_loss=lp,native_minus_loss=lm,
                                     cpu_plus_loss=float(cp),cpu_minus_loss=float(cm),
                                     plus_samples=sp.tolist(),minus_samples=sm.tolist(),
                                     plus_meta=mp,minus_meta=mm,smooth=smooth))
                fd_records.append(dict(step=h,rows=rows,
                    native_fd_relative=relative([r['native_fd'] for r in rows],np.array([r['expected'] for r in rows])),
                    cpu_fd_relative=relative([r['cpu_fd'] for r in rows],np.array([r['expected'] for r in rows]))))
            n=f['n']
            groups={'geometry':relative(gpu[:9*n],exact[:9*n]),
                    'sigma_opacity':relative(gpu[9*n:11*n],exact[9*n:11*n])}
            if f['degree']>=0:
                groups['SH']=relative(gpu[11*n:],exact[11*n:])
            at_gate=next(s for s in fd_records if s['step']==plan['gates']['finite_difference_step'])
            forward_error=float(np.max(np.abs(data[-1]-cpu_rgb.detach().numpy())))
            smooth=all(r['smooth'] for s in fd_records for r in s['rows'])
            gate=all(v<1e-3 for v in groups.values()) and forward_error<3e-5 and smooth \
                and at_gate['native_fd_relative']<1e-3 and at_gate['cpu_fd_relative']<1e-4
            row=dict(variant=variant,name=scene['name'],native_loss=data[0],cpu_loss=float(cpu_loss.detach()),
                     gradients_relative=groups,forward_max_abs=forward_error,repeat_max_abs=repeat_max,
                     exact_gradient=exact.tolist(),native_gradient=gpu.tolist(),
                     native_radii=data[1][3].cpu().tolist(),metadata=meta,
                     all_perturbations_smooth=smooth,finite_differences=fd_records,gate=bool(gate))
            records.append(row)
            result_path=OUT/(variant+'_'+scene['name']+'.npz')
            np.savez_compressed(result_path,native_rgb=data[1][1].cpu().numpy(),
                native_aux=data[1][2].cpu().numpy(),native_samples=data[-1],
                cpu_samples=cpu_rgb.detach().numpy(),
                **{f'native_grad_{i}':a for i,a in enumerate(arrays)})
            (OUT/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps({k:row[k] for k in ['variant','name','gradients_relative','forward_max_abs',
                                               'all_perturbations_smooth','gate']}),flush=True)
    summary=dict(scope=plan['scope'],protocol_sha256=sha(OUT/'protocol.json'),
                 script_sha256=sha(Path(__file__)),reference_sha256=sha(ROOT/'tgs_mixed_reference.py'),
                 records=records,all_fixed_gates=all(r['gate'] for r in records if r['variant']=='fixed'),
                 total_probe_step_pairs=sum(len(s['rows']) for r in records for s in r['finite_differences']),
                 seconds=time.time()-start,training_performed=False,quality_gain_established=False)
    (OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['all_fixed_gates','total_probe_step_pairs','seconds']}),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        if OUT.exists():
            (OUT/'execution_failure.txt').write_text(traceback.format_exc())
        raise
