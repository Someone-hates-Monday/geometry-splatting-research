from pathlib import Path
import tempfile,time,zipfile
import numpy as np
import torch
import mpmath as mp
from research.benchmark_csr_decode import src_rows,read,sha,save,now
from research.benchmark_csr_capture import pinned
from research.benchmark_dynamic_root import compact
from research.benchmark_full_topology import digest
from research.exact_perspective_backend import ROOT
from research.exact_outlier_capture_rebuilt import native_args
from research.benchmark_median_cuda import load_config
from research.verify_csr_decode import reference
from research.verify_dynamic_root import stable_residual
R=ROOT/'research';BASE=R/'experiments/dynamic_root_v1';OUT=R/'experiments/dynamic_root_rare_v1';AUDIT=R/'results/dynamic_root_rare_audit_v1.json'
def main():
    from scene import Scene,GaussianModel
    assert not OUT.exists() and not AUDIT.exists();OUT.mkdir();torch.set_num_threads(1);main=read(BASE/'manifest.json');assert main['status']=='complete';summary=read(BASE/'summary.json');top=read(R/'experiments/full_topology_v1/manifest.json');c=read(R/'experiments/shared_cache_v1/manifest.json');bfile=next(n for n in c['generated_sha256'] if n.endswith('.so'));geometry=pinned(dict(binary_relative_path=(R/'experiments/shared_cache_v1/generated'/bfile).relative_to(ROOT).as_posix(),binary_sha256=c['generated_sha256'][bfile]));b=main['backend'];module=pinned(dict(binary_relative_path=Path(b['binary_path']).relative_to(ROOT).as_posix(),binary_sha256=b['binary_sha256']));rows=[];rays=0;coeff=0;no_roots=0;finite=0;beta_gaps=[];failures=[]
    sourcefiles=[Path(__file__),R/'DYNAMIC_ROOT_RARE_PROTOCOL.md',R/'verify_dynamic_root.py',R/'verify_csr_decode.py',R/'benchmark_dynamic_root.py',R/'benchmark_csr_decode.py',BASE/'manifest.json',BASE/'summary.json',R/'results/dynamic_root_audit_v1.json'];sources={p.relative_to(ROOT).as_posix():sha(p) for p in sourcefiles};save(OUT/'manifest.json',dict(status='running',created_utc=now(),source_sha256=sources,binary_sha256=b['binary_sha256']))
    with zipfile.ZipFile(OUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sourcefiles:z.write(p,p.relative_to(ROOT).as_posix())
    for scan in [37,55,63]:
        model=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=model/'point_cloud/iteration_30000/point_cloud.ply';expected=top['model_inputs'][str(scan)]['ply_sha256'];assert sha(ply)==expected;ds,pipe=load_config(model)
        with tempfile.TemporaryDirectory(prefix='rare-source-scene-',dir=OUT) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch);gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False);cams={cam.image_name:cam for cam in scene.getTrainCameras()+scene.getTestCameras()};bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
            for r in [r for r in summary['rows'] if r['scan']==scan]:
                stem=f'scan{scan}_{r["camera"]}';rootpath=BASE/(stem+'_output.npz');assert sha(rootpath)==main['artifacts_sha256'][rootpath.name]
                with np.load(rootpath) as d:allpixels=d['pixels'];alloutput=d['output']
                chosen=((alloutput[:,0]==2)|(alloutput[:,0]==3)|((alloutput[:,0]==1)&((alloutput[:,1]==1)|(alloutput[:,12]==2))));pixels=allpixels[chosen];output=alloutput[chosen]
                if not len(pixels):continue
                with np.load(R/'experiments/csr_decode_v1'/(stem+'_query_amp.npz')) as d:q=d['query_xy'];amp=d['amplitude']
                with np.load(R/'experiments/csr_capture_v1'/(stem+'.npz')) as d:records=d['records'];offsets=d['offsets']
                args=native_args(cams[r['camera']],gs,pipe,bg,ds.kernel_size);assert {str(i):digest(args[i]) for i in range(11)}==top['model_inputs'][str(scan)]['tensors'];assert {str(i):digest(args[i]) for i in [14,15,21]}==read(R/'experiments/full_topology_v1'/(stem+'.json'))['camera_tensor_inputs']
                with torch.no_grad():
                    src=src_rows(args);cache=geometry.build(src);at=torch.tensor(amp,device='cuda');sections,indices,ids,query=compact(pixels,offsets,records,q);st=torch.tensor(sections,device='cuda');it=torch.tensor(ids,dtype=torch.int32,device='cuda');qt=torch.tensor(query,device='cuda');prosrc=src[it.long()].clone();prosrc[:,32:34]=torch.tensor(q[np.repeat(pixels,np.diff(sections))],dtype=torch.float64,device='cuda');prosrc[:,34]=1;prosrc[:,35]=torch.tensor(amp[ids],dtype=torch.float64,device='cuda');source=prosrc.cpu().numpy();packets=np.full((len(ids),26),np.nan,np.float64)
                    for mode in [0,1]:
                        data=module.decode(cache,it,st,qt,at,mode).cpu().numpy()
                        for j in np.flatnonzero(output[:,1]==mode):packets[sections[j]:sections[j+1]]=data[sections[j]:sections[j+1]]
                    np.savez_compressed(OUT/(stem+'_probes.npz'),pixels=pixels,output=output,source=source,packets=packets,sections=sections,record_indices=indices,primitive_ids=ids,query=q[pixels]);assert np.all(packets[:,0]==0) and np.isfinite(packets).all()
                for j,pixel in enumerate(pixels):
                    begin,end=int(sections[j]),int(sections[j+1]);nodes=[]
                    for i in range(begin,end):
                        mp.mp.dps=80;v80=reference(source[i]);mp.mp.dps=140;v=reference(source[i]);nodes.append(v)
                        for key,a,bb in [('p',1,2),('rho',3,4),('E',5,6),('alpha',24,25)]:
                            value=v[key];assert abs(value-v80[key])<=mp.mpf('1e-65')*max(1,abs(value));slack=mp.mpf('1e-120')*max(1,abs(value));coeff+=1
                            if not mp.mpf(float(packets[i,a]))-slack<=value<=mp.mpf(float(packets[i,bb]))+slack:failures.append(dict(view=stem,pixel=int(pixel),record=i,field=key))
                    o=output[j];mp.mp.dps=240
                    if o[0] in [2,3]:
                        beta=mp.fprod([1-v['alpha'] for v in nodes]);assert beta>=.5;no_roots+=1;beta_gaps.append(beta-.5)
                    else:
                        center=mp.mpf(float(o[2]))*mp.sqrt(mp.mpf(float(q[pixel,0]))**2+mp.mpf(float(q[pixel,1]))**2+1);tau=mp.mpf(float.fromhex('0x1.0624dd2f1a9fbp-10'));assert center-mp.mpf(float(o[4]))<=tau and mp.mpf(float(o[5]))-center<=tau
                        for t,a,bb,scale in [(o[4],6,7,o[10]),(o[5],8,9,o[11])]:
                            value=stable_residual(nodes,source[begin:end],mp.mpf(float(t)),mp.mpf(float(scale)))
                            if o[12]==1:value=mp.exp(-(value+mp.log(2)))
                            slack=mp.mpf('1e-110')*max(1,abs(value))
                            if not mp.mpf(float(o[a]))-slack<=value<=mp.mpf(float(o[bb]))+slack:failures.append(dict(view=stem,pixel=int(pixel),field='endpoint',value=mp.nstr(value,65)))
                            assert (value>.5 if a==6 else value<.5) if o[12]==1 else (value<0 if a==6 else value>0)
                        finite+=1
                    rays+=1
                rows.append(dict(view=stem,rays=len(pixels),nodes=len(source),no_finite_root=int(np.sum((output[:,0]==2)|(output[:,0]==3))),high=int(np.sum((output[:,0]==1)&(output[:,1]==1))),short=int(np.sum((output[:,0]==1)&(output[:,12]==2)))));save(OUT/'progress.json',dict(utc=now(),completed_views=len(rows),rays=rays,coefficients=coeff,failures=failures));print(f'{stem}: rare subset rays={rays}, coefficients={coeff}',flush=True);del args,src,cache,at,st,it,qt,prosrc,source,packets,records,offsets;torch.cuda.empty_cache()
            del scene,gs,cams;torch.cuda.empty_cache()
        assert sha(ply)==expected
    for n,h in sources.items():assert sha(ROOT/n)==h,n
    assert no_roots==256 and sum(r['high'] for r in rows)==29 and sum(r['short'] for r in rows)==26
    result=dict(status='complete',checked_utc=now(),conditional_subset_rays=rays,world_nodes=coeff//4,coefficient_checks=coeff,no_finite_root=no_roots,finite=finite,high_mode_rays=29,short_certificate_rays=26,minimum_true_beta_minus_half=mp.nstr(min(beta_gaps),80),world_precisions=[80,140],scaled_endpoint_precision=240,failures=failures,gate_pass=not failures,rows=rows,source_sha256=sources,binary_sha256=b['binary_sha256'],scope='Entire conditional union of rare branches, not representative/population holdout or complete speed/GT/gradient evidence.')
    save(AUDIT,result);assert not failures;save(OUT/'manifest.json',dict(status='complete',completed_utc=now(),source_sha256=sources,binary_sha256=b['binary_sha256'],artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']}));print({k:v for k,v in result.items() if k not in ['rows','source_sha256']})
if __name__=='__main__':main()
