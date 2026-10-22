"""All891 complete corrected-ray probes: source-preserving raw-parameter VJP."""
from pathlib import Path
import argparse,json,os,time,traceback,zipfile
import numpy as np
import torch
import mpmath as mp
import sympy as sp
from plyfile import PlyData
from research.world_root_vjp import torch_fields,torch_vjp,polish,mp_geometry,mp_vjp
from research.verify_dynamic_root import stable_residual
from research.exact_outlier_world_reference import FIELDS
from research.benchmark_csr_decode import read,sha,save,now
from research.exact_perspective_backend import ROOT
R=ROOT/'research';BASE=R/'experiments/dynamic_root_v1';TOPO=R/'experiments/full_topology_v1'

def symbols():
    A,E,p,rho,t=sp.symbols('A E p rho t',positive=True);alpha=A*sp.exp(-E/2);x=(t-p)*rho;u=alpha*sp.exp(-x*x/2);tail=u/(2*(1-u));checks=[]
    for after in [False,True]:
        H=-sp.log(1-alpha)+(sp.log(1-u)/2) if after else -sp.log(1-u)/2;signed=-tail if after else tail;ha=(alpha/(1-alpha) if after else 0)+signed
        checks.extend([sp.simplify(sp.diff(H,p)-signed*x*rho)==0,sp.simplify(sp.diff(H,rho)+signed*x*x/rho)==0,sp.simplify(sp.diff(H,E)+ha/2)==0,sp.simplify(A*sp.diff(H,A)-ha)==0])
    assert all(checks);return dict(layer_partial_identities=checks,sympy_version=sp.__version__)

def metadata(row):
    c=row['camera_scalar_inputs'];tx=np.float32(c['16']);ty=np.float32(c['17']);W=row['width'];H=row['height'];return dict(fx=float(np.float32(np.float32(W)/(np.float32(2)*tx))),fy=float(np.float32(np.float32(H)/(np.float32(2)*ty))),limx=float(np.float32(np.float32(1.3)*tx)),limy=float(np.float32(np.float32(1.3)*ty)),kernel=float(np.float32(c['18'])),floor=float(np.float32(1e-6)))

def root_reference(nodes,source,lo,hi):
    mp.mp.dps=240;lo=mp.mpf(float(lo));hi=mp.mpf(float(hi));l=stable_residual(nodes,source,lo,mp.mpf(0));h=stable_residual(nodes,source,hi,mp.mpf(0));assert l<0<h
    for iterations in range(200):
        mid=(lo+hi)/2;value=stable_residual(nodes,source,mid,mp.mpf(0))
        if value<0:lo=mid
        elif value>0:hi=mid
        else:raise ValueError('MP residual precision unresolved')
        if hi-lo<mp.mpf('1e-35'):break
    assert hi-lo<mp.mpf('1e-35');return (lo+hi)/2,lo,hi,iterations+1

def metrics(actual,reference):
    if not np.isfinite(actual).all():return dict(relative_l2=None,blocks=None,gate_pass=False,nonfinite=True)
    norm=np.linalg.norm(reference);rel=float(np.linalg.norm(actual-reference)/max(norm,1e-300));blocks={}
    for name,a,b in [('xyz',0,3),('log_scale',3,6),('quaternion',6,10),('opacity_logit',10,11)]:
        n=np.linalg.norm(reference[:,a:b]);blocks[name]=float(np.linalg.norm(actual[:,a:b]-reference[:,a:b])/max(n,1e-30))
    return dict(relative_l2=rel,blocks=blocks,gate_pass=rel<=1e-5 and max(blocks.values())<=1e-4,nonfinite=False)

def run(out,state):
    torch.set_num_threads(1);sym=symbols();save(out/'symbolic.json',sym);m=read(BASE/'manifest.json');summary=read(BASE/'summary.json');top=read(TOPO/'manifest.json');cases={}
    for c in read(R/'results/dynamic_root_audit_v1.json')['old_anomalies']:cases[(c['view'],c['pixel'])]=c['case']
    rows=[];totalnodes=0;reference_failures=[];expected891=0
    for scan in [37,55,63]:
        ply=ROOT/f'output_compare/ggs_pipeline/scan{scan}/point_cloud/iteration_30000/point_cloud.ply';expected=top['model_inputs'][str(scan)]['ply_sha256'];assert sha(ply)==expected;data=PlyData.read(ply).elements[0].data
        for r in [r for r in summary['rows'] if r['scan']==scan]:
            stem=f'scan{scan}_{r["camera"]}';path=BASE/(stem+'_probes.npz');assert sha(path)==m['artifacts_sha256'][path.name];meta=metadata(read(TOPO/(stem+'.json')))
            with np.load(path) as d:source=d['source'];sections=d['sections'];pixels=d['pixels'];ids=d['primitive_ids'];output=d['output']
            raw=np.stack([data[k][ids] for k in FIELDS],axis=1).astype(np.float64);assert np.array_equal(raw[:,:3],source[:,:3]);assert np.all(output[:,0]==1);expected891+=len(pixels)
            arrays={arm:np.full((len(source),11),np.nan,np.float64) for arm in ['reference','factor_candidate','factor_public','factor_refined','dense_refined']};rayrows=[]
            for j,pixel in enumerate(pixels):
                start=time.perf_counter();begin,end=int(sections[j]),int(sections[j+1]);src=source[begin:end];params=raw[begin:end];old=output[j];save(out/'progress.json',dict(utc=now(),completed_rays=len(rows),active_view=stem,active_pixel=int(pixel),phase='world_reference'))
                try:
                    mp.mp.dps=140;nodes=[mp_geometry(v) for v in src];mp_root,mp_lo,mp_hi,mp_iter=root_reference(nodes,src,old[4],old[5]);mpgrad,slope,screen=mp_vjp(nodes,src,params,meta,mp_root);reference=np.array([[float(v) for v in row] for row in mpgrad]);arrays['reference'][begin:end]=reference
                    # Reference matrices are built at140;240 used for root/scaled
                    # arithmetic. Independent Torch implementations receive only
                    # original source and existing certified endpoints.
                    theta=torch.tensor(params[:,:11],dtype=torch.float64)
                    with torch.no_grad():factorfields=torch_fields(theta,params,src,meta,'factor');densefields=torch_fields(theta,params,src,meta,'dense')
                    factor_t,factor_status,factor_steps=polish(factorfields,old[4],old[5]);dense_t,dense_status,dense_steps=polish(densefields,old[4],old[5]);center=float(old[2])*float(np.linalg.norm(src[0,32:35]));evaluations=[('factor_candidate','factor',float(old[3])),('factor_public','factor',center),('factor_refined','factor',factor_t),('dense_refined','dense',dense_t)];branchrows={}
                    for label,arm,t in evaluations:
                        try:
                            if t is None:raise ValueError('Implementation root bracket failed')
                            grad,info=torch_vjp(params,src,meta,t,arm);arrays[label][begin:end]=grad;branchrows[label]=dict(status='finite',t=t,root_error=float(abs(mp.mpf(t)-mp_root)),**info,**metrics(grad,reference))
                        except (ValueError,FloatingPointError,RuntimeError) as e:branchrows[label]=dict(status=type(e).__name__,message=str(e),gate_pass=False)
                    row=dict(view=stem,pixel=int(pixel),old_anomaly=cases.get((stem,int(pixel))),nodes=end-begin,reference_status='regular',reference_root=mp.nstr(mp_root,55),reference_bracket=[mp.nstr(mp_lo,55),mp.nstr(mp_hi,55)],reference_iterations=mp_iter,reference_log_H_slope=mp.nstr(mp.log(slope),45),reference_gradient_norm=mp.nstr(mp.sqrt(mp.fsum(v*v for rr in mpgrad for v in rr)),45),screen_floor0=sum(v.get('floor0',False) for v in screen),screen_floor1=sum(v.get('floor1',False) for v in screen),branches=branchrows,factor_root_polish=dict(status=factor_status,steps=factor_steps),dense_root_polish=dict(status=dense_status,steps=dense_steps),wall_seconds=time.perf_counter()-start)
                except (ValueError,FloatingPointError,RuntimeError,ZeroDivisionError) as e:
                    row=dict(view=stem,pixel=int(pixel),old_anomaly=cases.get((stem,int(pixel))),nodes=end-begin,reference_status=type(e).__name__,message=str(e),branches={},wall_seconds=time.perf_counter()-start);reference_failures.append(row)
                rows.append(row);rayrows.append(row);totalnodes+=end-begin
                if (j+1)%4==0 or j+1==len(pixels):save(out/'progress.json',dict(utc=now(),completed_rays=len(rows),active_view=stem,completed_in_view=j+1,coefficients=totalnodes,reference_failures=len(reference_failures)));print(f'{stem}: VJP rays{j+1}/{len(pixels)}, total{len(rows)}, reference_failures={len(reference_failures)}',flush=True)
            np.savez_compressed(out/(stem+'_gradients.npz'),pixels=pixels,sections=sections,primitive_ids=ids,source=source,raw=raw,**arrays);save(out/(stem+'.json'),dict(meta=meta,rows=rayrows))
        del data;assert sha(ply)==expected
    assert len(rows)==expected891==891 and totalnodes==51524
    counts={};maxima={}
    for branch in ['factor_candidate','factor_public','factor_refined','dense_refined']:
        finite=[r['branches'][branch] for r in rows if branch in r['branches'] and r['branches'][branch]['status']=='finite'];counts[branch]=dict(pass_count=sum(v['gate_pass'] for v in finite),finite=len(finite),failed_or_undefined=891-sum(v['gate_pass'] for v in finite));maxima[branch]=max([v['relative_l2'] for v in finite if v['relative_l2'] is not None],default=None)
    result=dict(status='complete',rows=rows,rays=891,nodes=51524,reference_failures=reference_failures,branch_counts=counts,max_relative_l2=maxima,refined_factor_gate_pass=counts['factor_refined']['pass_count']==891,all9_old_anomalies_present=sum(r['old_anomaly'] is not None for r in rows)==9,world_geometry_precision=140,root_and_IFT_arithmetic_precision=240,scope='CPU continuous extension through actual captured values, full11 per-node raw-parameter VJP including3D filter and screen compensation, not fullCUDA/train/GT or IEEE literal differentiation.')
    save(out/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h,n
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(out/'manifest.json',state);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','reference_failures','scope']},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);out=p.parse_args().out.resolve();assert not out.exists();out.mkdir(parents=True);sources=[Path(__file__),R/'world_root_vjp.py',R/'WORLD_ROOT_VJP_PROTOCOL.md',R/'verify_dynamic_root.py',R/'exact_outlier_world_reference.py',R/'benchmark_csr_decode.py',R/'exact_perspective_backend.py',R/'scene_dummy_unused' if False else ROOT/'scene/gaussian_model.py',BASE/'manifest.json',BASE/'summary.json',TOPO/'manifest.json',R/'results/dynamic_root_audit_v1.json',ROOT/'submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu'];state=dict(status='running',created_utc=now(),pid=os.getpid(),source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources});save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state)
    except BaseException as e:state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
