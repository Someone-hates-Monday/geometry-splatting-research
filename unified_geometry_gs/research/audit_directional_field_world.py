"""All-Gaussian world-field measurements and two-term interval certificates."""
from __future__ import annotations
from datetime import datetime,timezone
import argparse,hashlib,json,os,time,traceback,zipfile
from pathlib import Path
import numpy as np
from plyfile import PlyData
from research.affine_rays_cpu import read_model
from research.directional_field import orthogonal_witness,interval_pair_certificate

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)


def raw_row(data,index):
    d=data[index]
    return {'gaussian_index':int(index),'mean':[float(d[k]) for k in ('x','y','z')],
            'log_scale':[float(d[f'scale_{i}']) for i in range(3)],
            'raw_quaternion':[float(d[f'rot_{i}']) for i in range(4)],
            'opacity_logit':float(d['opacity']),'filter_3D':float(d['filter_3D'])}


def measure(model,raw,point,direction):
    xyz,scales2,amp,R=model;n=len(amp);B=np.empty((n,3));logv=0.;underflow=0
    for lo in range(0,n,65536):
        hi=min(n,lo+65536);delta=point-xyz[lo:hi]
        local=np.einsum('nij,ni->nj',R[lo:hi],delta)
        quadratic=np.sum(local*local/scales2[lo:hi],axis=1)
        G=amp[lo:hi]*np.exp(-quadratic/2)
        underflow+=int(sum(G==0))
        Pd=np.einsum('nij,nj->ni',R[lo:hi],local/scales2[lo:hi])
        B[lo:hi]=(G/(2*(1-G)))[:,None]*Pd
        logv+=float(.5*np.log1p(-G).sum())
    if not np.all(np.isfinite(B)):raise ArithmeticError('Nonfinite nominal gradient; no clipping allowed')
    norm2=np.sum(B*B,axis=1);first=int(np.argmax(norm2))
    if norm2[first]==0:raise ArithmeticError('All nominal gradients vanished; preserve case as failed')
    u=B[first]/np.sqrt(norm2[first]);projected=B-(B@u)[:,None]*u
    second=int(np.argmax(np.sum(projected*projected,axis=1)))
    pair=[raw_row(raw,i) for i in (first,second)]
    interval=interval_pair_certificate(pair,point)
    S=B.T@B;total=B.sum(axis=0);eigen=np.linalg.eigvalsh(S)
    nominal_bound=float(np.sqrt(max(0.,eigen[-2])))
    worlddirs=np.array([[1.,0,0],[0,1.,0],[0,0,1.],[1.,1.,1.],[1.,-1.,0],direction,orthogonal_witness(total,S)])
    worlddirs/=np.linalg.norm(worlddirs,axis=1)[:,None]
    K=np.zeros(len(worlddirs))
    for lo in range(0,n,65536):K+=abs(B[lo:lo+65536]@worlddirs.T).sum(axis=0)
    product=abs(worlddirs@total);gaps=K-product
    normsum=float(np.sqrt(norm2).sum())
    return {'gaussians_evaluated':n,'nominal_zero_G_count':underflow,'invalid_gradient_count':0,
            'log_product_vacancy':logv,'sum_gradient':total.tolist(),'gradient_gram':S.tolist(),
            'gram_eigenvalues_ascending':eigen.tolist(),'nominal_spectral_error_lower_bound':nominal_bound,
            'nominal_sum_gradient_norms':normsum,'nominal_lower_bound_over_sum_norms':nominal_bound/normsum,
            'witness_pair':pair,'witness_nominal_gradients':B[[first,second]].tolist(),'interval_pair_certificate':interval,
            'direction_names':['x','y','z','xyz','x-minus-y','query-camera','product-gradient-orthogonal-witness'],
            'directions':worlddirs.tolist(),'summed_attenuation':K.tolist(),'product_field_attenuation':product.tolist(),
            'directional_gap':gaps.tolist(),'relative_directional_gap':[float(g/k) if k>0 else None for g,k in zip(gaps,K)]}


def execute(out,manifest,oldfiles):
    results=[];start=time.perf_counter()
    for oldfile in oldfiles:
        legacy=json.loads(oldfile.read_text(encoding='utf-8'));scan=int(oldfile.stem.replace('median_cpu_scan',''))
        modelpath=ROOT/f'output_compare/ggs_pipeline/scan{scan}'
        ply=modelpath/'point_cloud/iteration_30000/point_cloud.ply'
        assert sha(ply)==next(v for k,v in legacy['source_sha256'].items() if k.endswith('point_cloud.ply'))
        model=read_model(ply);raw=PlyData.read(ply)['vertex'].data
        if not np.all((model[2]>0)&(model[2]<1)):raise ArithmeticError('Nominal amplitudes outside (0,1)')
        cameras=json.loads((modelpath/'cameras.json').read_text(encoding='utf-8'))
        byname={c['img_name']:c for c in cameras}
        cfg=(modelpath/'cfg_args').read_text(encoding='utf-8')
        if 'resolution=2,' not in cfg or 'kernel_size=0.0,' not in cfg:raise ValueError('Projection configuration changed')
        seen=set()
        for legacy_index,row in enumerate(legacy['rows']):
            name=row['camera']
            if name in seen or not row['finite_root']:continue
            seen.add(name);camera=byname[name]
            width,height=round(camera['width']/2),round(camera['height']/2)
            fx,fy=camera['fx']*width/camera['width'],camera['fy']*height/camera['height']
            px,py=row['pixel'];d=np.array([(px-(width-1)/2)/fx,(py-(height-1)/2)/fy,1.])
            d/=np.linalg.norm(d);worlddir=np.asarray(camera['rotation'])@d
            point=np.asarray(camera['position'])+worlddir*row['median_ray_depth']
            record={'scan':scan,'legacy_row':legacy_index,'camera':name,'pixel':row['pixel'],
                    'saved_affine_median_depth':row['median_ray_depth'],'world_point':point.tolist(),
                    'query_world_direction':worlddir.tolist(),**measure(model,raw,point,worlddir)}
            results.append(record)
            save(out/f'point_{len(results)-1:02d}.json',record)
            save(out/'progress.json',{'utc':now(),'points_completed':len(results),'scan':scan,'camera':name})
            print(f'{len(results)}/24 scan{scan} camera{name}: pair-certified={record["interval_pair_certificate"]["positive_certified"]}',flush=True)
        if len(seen)!=8:raise ValueError('Expected eight original saved views')
        del raw,model
    if len(results)!=24:raise ValueError('Expected 24 fixed query points')
    summary={'points':len(results),'positive_pair_certificates':sum(r['interval_pair_certificate']['positive_certified'] for r in results),
             'nominal_spectral_lower_bounds':[r['nominal_spectral_error_lower_bound'] for r in results],
             'interval_absolute_lower_bounds':[r['interval_pair_certificate']['conservative_double_lower_bound'] for r in results],
             'nominal_normalized_bounds':[r['nominal_lower_bound_over_sum_norms'] for r in results],
             'limits':['Pointwise single-normal absolute-gradient form only; general stochastic normal mixtures are not ruled out.',
                       'Queries are saved affine median backprojections, not GT surface or exact world-model median points.',
                       'Full float64 summaries are nominal; two-term raw-parameter interval certificates are separate.',
                       'No image, reconstruction, production affine renderer, training or novelty claim.']}
    save(out/'summary.json',summary)
    manifest.update(status='complete',completed_utc=now(),wall_seconds=time.perf_counter()-start,
                    artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',manifest)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve()
    if out.exists():raise ValueError('Fresh directory required')
    out.mkdir(parents=True)
    oldfiles=[ROOT/f'research/results/median_cpu_scan{s}.json' for s in (37,55,63)]
    sources=[Path(__file__),ROOT/'research/directional_field.py',ROOT/'research/affine_rays_cpu.py',ROOT/'research/DIRECTIONAL_FIELD_PROTOCOL.md']+oldfiles
    for s in (37,55,63):sources.extend([ROOT/f'output_compare/ggs_pipeline/scan{s}/cameras.json',ROOT/f'output_compare/ggs_pipeline/scan{s}/cfg_args'])
    manifest={'created_utc':now(),'status':'running','pid':os.getpid(),
              'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    save(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.writestr(str(p.relative_to(ROOT)),p.read_bytes())
    try:execute(out,manifest,oldfiles)
    except BaseException as exc:
        manifest.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc))
        save(out/'manifest.json',manifest)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
