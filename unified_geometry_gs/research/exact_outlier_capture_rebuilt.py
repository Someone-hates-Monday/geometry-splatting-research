"""New diagnostic: pinned rebuilt binaries and strict whole-image output replay."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,tempfile,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from research.exact_perspective_backend import ROOT,RASTER,InferenceBackend,metadata
from research.exact_outlier_capture_backend import load as load_probe,payloads as probe_payloads
from research.benchmark_median_cuda import load_config

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def savez(p,**arrays):
    temp=p.with_suffix('.npz.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(p)

def fields(buffer,spec):
    assert buffer.dtype==torch.uint8 and buffer.is_cuda and buffer.is_contiguous()
    assert buffer.data_ptr()%128==0
    offset=0;values={};layout={}
    for name,dtype,shape in spec:
        offset=(offset+127)//128*128
        count=math.prod(shape);size=torch.empty((),dtype=dtype).element_size();end=offset+count*size
        assert end<=buffer.numel(),(name,offset,end,buffer.numel())
        values[name]=buffer[offset:end].view(dtype).reshape(shape)
        layout[name]={'offset':offset,'bytes':count*size,'shape':list(shape),'dtype':str(dtype)};offset=end
    return values,{'base_pointer':int(buffer.data_ptr()),'base_alignment_mod128':int(buffer.data_ptr()%128),
        'actual_buffer_bytes':buffer.numel(),'parsed_prefix_end':offset,'fields':layout}

def parse(native,variant,P,W,H):
    count=int(native[0]);tiles=((W+15)//16)*((H+15)//16)
    spec=[('depths',torch.float32,(P,)),('plane',torch.float32,(P,4))]
    if variant=='exact_factor':spec.append(('factor',torch.float32,(P,12)))
    spec += [('normal',torch.float32,(P,3)),('clamped',torch.uint8,(P,3)),('radii',torch.int32,(P,)),
        ('uv',torch.float32,(P,2)),('conic',torch.float32,(P,4)),('rgb',torch.float32,(P,3)),('tiles_touched',torch.int32,(P,))]
    g,gl=fields(native[6],spec);b,bl=fields(native[7],[('point_list',torch.int32,(count,))])
    i,il=fields(native[8],[('n_contrib',torch.int32,(H,W)),('normal_length',torch.float32,(H,W))])
    t,tl=fields(native[9],[('ranges',torch.int32,(tiles,2)),('max_contributor',torch.int32,(tiles,))])
    assert bool(torch.all(b['point_list']>=0)) and bool(torch.all(b['point_list']<P))
    assert bool(torch.all(t['ranges']>=0)) and bool(torch.all(t['ranges']<=count))
    assert bool(torch.all(t['ranges'][:,0]<=t['ranges'][:,1]))
    # The external radii tensor is supplied by RasterizeGaussiansCUDA, so the
    # allocated internal_radii field is unused/uninitialized in this call.
    gl['fields']['radii']['unused_internal_field']=True
    return {'geom':g,'binning':b,'image':i,'tile':t,'radii':native[5],'num_rendered':count}, {'geometry':gl,'binning':bl,'image':il,'tile':tl}

def selected(previous):
    summary=read(previous/'summary.json');assert summary['status']=='complete' and summary['views']==24
    rows=[]
    for row in summary['rows']:
        path=previous/f'scan{row["scan"]}_{row["camera"]}.npz'
        with np.load(path,allow_pickle=False) as z:
            b=z['exact_packet_depth'].squeeze();c=z['exact_factor_depth'].squeeze()
            mask=(b>0)&(c>0)&(abs(b-c)>1e-3)
            for y,x in np.argwhere(mask):rows.append({'case':len(rows),'scan':row['scan'],'camera':row['camera'],
                'pixel':[int(x),int(y)],'saved_packet_z_depth':float(b[y,x]),'saved_factor_z_depth':float(c[y,x]),
                'saved_absolute_z_depth_difference':float(abs(b[y,x]-c[y,x]))})
    assert len(rows)==9,len(rows)
    return rows

def native_args(cam,gs,pipe,bg,kernel):
    scales,opacities=gs.get_scaling_n_opacity_with_3D_filter
    return (bg,gs.get_xyz,torch.empty(0,device='cuda'),opacities,scales,gs.get_rotation,torch.empty(0,device='cuda'),
        gs.get_features,gs.get_sg_axis,gs.get_sg_sharpness,gs.get_sg_color,gs.active_sh_degree,gs.active_sg_degree,1.,
        cam.world_view_transform,cam.full_proj_transform,math.tan(cam.FoVx*.5),math.tan(cam.FoVy*.5),kernel,
        cam.image_height,cam.image_width,cam.camera_center,False,True,pipe.debug)

def load_pinned_backend(variant,pinned):
    pin=pinned[variant];path=ROOT/pin['binary_relative_path']
    assert sha(path)==pin['rebuilt_binary_sha256']
    for name,digest in pin['generated_sha256'].items():assert sha(ROOT/name)==digest,name
    spec=importlib.util.spec_from_file_location(path.stem,str(path))
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert sha(path)==pin['rebuilt_binary_sha256']
    record={'variant':variant,'signature':pin['signature'],'binary_sha256':pin['rebuilt_binary_sha256'],
        'frozen_old_binary_sha256':pin['frozen_old_binary_sha256'],
        'scope':'Pinned rebuilt binary diagnostic; output replay required; not historical binary recovery.'}
    return InferenceBackend(module,variant,record)

def execute(out,state,previous,cases):
    torch.set_num_threads(1);probe,probe_meta=load_probe(out);state['probe']=probe_meta
    backends={v:load_pinned_backend(v,state['pinned_rebuilt_binaries']) for v in ('exact_packet','exact_factor')}
    prior=read(previous/'manifest.json')
    for name,backend in backends.items():
        meta=metadata(backend);assert meta['binary_sha256']==state['pinned_rebuilt_binaries'][name]['rebuilt_binary_sha256']
        state.setdefault('backends',{})[name]={'binary_path':backend.__file__,'binary_sha256':meta['binary_sha256'],'signature':meta['signature']}
    state['status']='capturing';save(out/'manifest.json',state)
    from scene import GaussianModel,Scene
    completed=[]
    for scan in sorted(set(c['scan'] for c in cases)):
        model=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=model/'point_cloud/iteration_30000/point_cloud.ply'
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json');expected=next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
        assert sha(ply)==expected;state.setdefault('model_sha256',{})[ply.relative_to(ROOT).as_posix()]=expected;save(out/'manifest.json',state)
        ds,pipe=load_config(model)
        with tempfile.TemporaryDirectory(prefix='capture-scene-',dir=out) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch)
            gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
            cameras={c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()}
            bg=torch.tensor([1,1,1] if ds.white_background else [0,0,0],device='cuda',dtype=torch.float32)
            with torch.no_grad():
                for camera in sorted(set(c['camera'] for c in cases if c['scan']==scan)):
                    group=[c for c in cases if c['scan']==scan and c['camera']==camera];cam=cameras[camera]
                    args=native_args(cam,gs,pipe,bg,ds.kernel_size);W,H=cam.image_width,cam.image_height;P=gs.get_xyz.shape[0]
                    native={};parsed={};layouts={};full_equal={}
                    for variant in ('exact_packet','exact_factor'):
                        native[variant]=backends[variant]._module.rasterize_gaussians(*args);torch.cuda.synchronize()
                        parsed[variant],layouts[variant]=parse(native[variant],variant,P,W,H)
                        with np.load(previous/f'scan{scan}_{camera}.npz',allow_pickle=False) as z:
                            olddepth=torch.tensor(z[variant+'_depth'],device='cuda');oldslope=torch.tensor(z[variant+'_slope'],device='cuda')
                            full_equal[variant]={'depth_bit_pattern_equal':torch.equal(native[variant][4][0:1].view(torch.int32),olddepth.view(torch.int32)),
                                'slope_bit_pattern_equal':torch.equal(native[variant][4][1:2].view(torch.int32),oldslope.view(torch.int32))}
                            for field,index in (('render',1),('mask',2),('normal',3)):
                                base=z['affine_'+field];delta=z[variant+'_'+field+'_delta']
                                reconstruction=np.where(delta==0,base,base+delta).astype(np.float32,copy=False)
                                expected_image=torch.tensor(reconstruction,device='cuda')
                                full_equal[variant][field+'_archived_reconstruction_bit_pattern_equal']=torch.equal(native[variant][index].view(torch.int32),expected_image.view(torch.int32))
                        assert all(full_equal[variant].values()),full_equal
                    B,C=parsed['exact_packet'],parsed['exact_factor']
                    assert B['num_rendered']==C['num_rendered'] and torch.equal(B['binning']['point_list'],C['binning']['point_list'])
                    assert torch.equal(B['tile']['ranges'],C['tile']['ranges']) and torch.equal(B['image']['n_contrib'],C['image']['n_contrib'])
                    save(out/f'layout_scan{scan}_{camera}.json',{'scan':scan,'camera':camera,'primitive_count':P,'num_rendered':B['num_rendered'],
                        'width':W,'height':H,'layouts':layouts,'whole_image_replay':full_equal})
                    fx=np.float32(W)/(np.float32(2)*np.float32(args[16]));fy=np.float32(H)/(np.float32(2)*np.float32(args[17]))
                    for case in group:
                        x,y=case['pixel'];tile=(y//16)*((W+15)//16)+x//16
                        start,end=B['tile']['ranges'][tile].cpu().tolist();ids=B['binning']['point_list'][start:end].long()
                        ncontrib=int(B['image']['n_contrib'][y,x]);tilemax=int(B['tile']['max_contributor'][tile])
                        assert 0<ncontrib<=tilemax<=end-start
                        for name in ('uv','conic','plane','depths'):
                            assert torch.equal(B['geom'][name][ids],C['geom'][name][ids]),name
                        assert torch.equal(B['radii'][ids],C['radii'][ids]) and bool(torch.all(B['radii'][ids]>0))
                        xy=B['geom']['uv'][ids].contiguous();conic=B['geom']['conic'][ids].contiguous();plane=B['geom']['plane'][ids].contiguous();factor=C['geom']['factor'][ids].contiguous()
                        coeff,init=probe.forward(xy,conic,plane,factor,W,H,float(fx),float(fy),x,y);torch.cuda.synchronize()
                        cn=coeff.cpu().numpy();initial=init.cpu().numpy();assert int(initial[4])==ncontrib,(initial,ncontrib)
                        ranks=np.arange(1,len(ids)+1,dtype=np.int32);prefix=ranks<=ncontrib;eligibility=cn[:,2]>0;accepted=prefix&eligibility
                        raw=torch.cat((gs._xyz[ids],gs._scaling[ids],gs._rotation[ids],gs._opacity[ids],gs.filter_3D[ids]),dim=1)
                        arrays={'primitive_ids':ids.cpu().numpy(),'rank_one_based':ranks,'actual_prefix_mask':prefix,'copied_eligibility_mask':eligibility,'accepted_mask':accepted,
                            'uv':xy.cpu().numpy(),'conic_opacity':conic.cpu().numpy(),'ray_plane':plane.cpu().numpy(),'factor12':factor.cpu().numpy(),
                            'copied_coefficients_all':cn,'copied_initialization':initial,'raw_parameters':raw.cpu().numpy(),
                            'filtered_scales':args[4][ids].cpu().numpy(),'normalized_quaternions':args[5][ids].cpu().numpy(),'filtered_opacity':args[3][ids].cpu().numpy(),
                            'world_view_transform':args[14].cpu().numpy(),'full_proj_transform':args[15].cpu().numpy(),'camera_center':args[21].cpu().numpy(),
                            'focal':np.array([fx,fy],dtype=np.float32),'pixel':np.array([x,y],dtype=np.int32),'image_wh':np.array([W,H],dtype=np.int32),
                            'query_xy':initial[6:8],'rln':initial[5:6],'derived_qray_norm_from_probe_rln':np.array([np.float32(1)/initial[5]],dtype=np.float32),
                            'packet_selected_ids':ids.cpu().numpy()[accepted],'factor_selected_ids':ids.cpu().numpy()[accepted],
                            'packet_alpha':cn[accepted,7],'packet_peak':cn[accepted,5],'packet_rho':cn[accepted,6],
                            'factor_alpha':cn[accepted,11],'factor_peak':cn[accepted,9],'factor_rho':cn[accepted,10],
                            'affine_alpha':cn[accepted,1],'affine_peak':cn[accepted,3],'affine_rho':cn[accepted,4]}
                        outputs={}
                        for variant in ('exact_packet','exact_factor'):
                            depth=np.float32(native[variant][4][0,y,x].cpu());slope=np.float32(native[variant][4][1,y,x].cpu());raydepth=np.float32(depth/initial[5])
                            arrays[variant+'_output_z_depth']=np.array([depth],dtype=np.float32);arrays[variant+'_output_slope']=np.array([slope],dtype=np.float32)
                            arrays[variant+'_derived_output_ray_depth']=np.array([raydepth],dtype=np.float32)
                            outputs[variant]={'original_z_depth':float(depth),'original_interface_slope':float(slope),'derived_ray_depth_using_probe_rln':float(raydepth)}
                        arrays['copied_mDepthinit']=initial[0:1];arrays['copied_initial_depth_min']=initial[1:2];arrays['copied_initial_depth_max']=initial[2:3]
                        arrays['actual_n_contrib']=np.array([ncontrib],dtype=np.int32);arrays['actual_tile_range']=np.array([start,end],dtype=np.int32)
                        arrays['actual_tile_max_contributor']=np.array([tilemax],dtype=np.int32)
                        index=case['case'];savez(out/f'ray_{index:02d}.npz',**arrays)
                        row=dict(case,width=W,height=H,focal=[float(fx),float(fy)],inputs_in_tile=len(ids),actual_n_contrib=ncontrib,actual_tile_range=[start,end],actual_tile_max_contributor=tilemax,
                            accepted_inputs=int(sum(accepted)),unaccepted_inputs=int(sum(~accepted)),accepted_packet_invalid=int(sum(cn[accepted,8]==0)),
                            accepted_factor_invalid=int(sum(cn[accepted,12]==0)),copied_initialization={'mDepthinit':float(initial[0]),'depth_min':float(initial[1]),
                            'depth_max':float(initial[2]),'final_affine_T':float(initial[3]),'last_contributor':int(initial[4]),'rln':float(initial[5]),'qray_xy':initial[6:8].tolist()},
                            outputs=outputs,coefficient_columns=['affine_power','affine_alpha_capped','affine_eligible','affine_peak','affine_rho','packet_peak','packet_rho','packet_alpha','packet_valid','factor_peak','factor_rho','factor_alpha','factor_valid'],
                            probe_scope='Copied expression coefficients/initialization, not original instruction trace; accepted prefix length is actual img.n_contrib.')
                        save(out/f'ray_{index:02d}.json',row);completed.append(row);save(out/'progress.json',{'utc':now(),'completed_rays':len(completed),'last_case':index})
                        print(f'captured {len(completed)}/9: scan{scan}/{camera} pixel{x},{y}, prefix={ncontrib}, accepted={sum(accepted)}',flush=True)
                    del native,parsed,layouts;torch.cuda.empty_cache()
            del scene,gs,cameras,args;torch.cuda.empty_cache()
        assert sha(ply)==expected
    assert len(completed)==9
    summary={'status':'complete','captured_rays':len(completed),'affected_views':len(set((c['scan'],c['camera']) for c in cases)),
        'all_nine_common_positive_abs_depth_errors_gt_1e_minus3_retained':True,'all_full_image_replays_bitwise_equal':True,
        'all_actual_prefix_counts_match_copied_init_replay':True,'binary_scope':'New pinned rebuilt binary execution, not historical binary recovery.',
        'all_affected_views':8,'all_full_image_depth_slope_and_archived_rgb_alpha_normal_reconstructions_bit_pattern_equal':True,
        'rows':sorted(completed,key=lambda r:r['case']),
        'limits':['Original buffers provide actual ranges, n_contrib, source packets and original output only.',
        'Coefficient/eligibility/initialization values come from a separate copied-expression CUDA probe, not original instruction instrumentation.',
        'Ray-depth is derived from original z-depth with probe rln, not an additional original output.',
        'No original final bracket or T endpoint trace is available; no root-cause, root-error bound or quality claim.',
        'Output equality does not prove historical hidden buffers or machine instructions identical.',
        'Captured buffers are from new pinned rebuilt binaries, whose SHA differ from frozen24.',
        'RGB/alpha/normal target is archived affine-base-plus-delta reconstruction, not separately archived original B/C RGB outputs.']}
    save(out/'summary.json',summary)
    for name,digest in state['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for pin in state['pinned_rebuilt_binaries'].values():assert sha(ROOT/pin['binary_relative_path'])==pin['rebuilt_binary_sha256']
    state.update(status='complete',completed_utc=now(),captured_rays=9,artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',state);print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if os.environ.get('TORCH_CUDA_ARCH_LIST')!='12.0':raise ValueError('Frozen binary replay requires explicit original TORCH_CUDA_ARCH_LIST=12.0 before any extension load.')
    if out.exists():raise ValueError('Fresh output required; inspect actual handle before retry.')
    previous=ROOT/'research/experiments/frozen_exact_render_v1';prior=read(previous/'manifest.json');assert prior['status']=='complete'
    for name,digest in prior['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in prior['artifacts_sha256'].items():assert sha(previous/name)==digest,name
    pinned={}
    for variant in ('exact_packet','exact_factor'):
        old=prior['binaries'][variant];binary=Path(old['path'])
        digest=sha(binary);assert digest!=old['sha256']
        checks={}
        for name,expected in prior['generated_sha256'][variant].items():
            source=Path(name);assert sha(source)==expected,name
            checks[source.relative_to(ROOT).as_posix()]=expected
        assert len(checks)==17
        pinned[variant]={'binary_relative_path':binary.relative_to(ROOT).as_posix(),
            'frozen_old_binary_sha256':old['sha256'],'rebuilt_binary_sha256':digest,
            'signature':old['metadata']['signature'],'generated_sha256':checks}
    cases=selected(previous)
    sources={ROOT/name for name in prior['source_sha256']};sources.update([Path(__file__),ROOT/'research/exact_outlier_capture_backend.py',
        ROOT/'research/EXACT_OUTLIER_CAPTURE_PROTOCOL.md',ROOT/'research/EXACT_OUTLIER_REBUILT_PROTOCOL.md',RASTER/'cuda_rasterizer/rasterizer_impl.cu',RASTER/'cuda_rasterizer/rasterizer_impl.h',
        ROOT/'research/exact_perspective_ray.inc',RASTER/'cuda_rasterizer/config.h',previous/'manifest.json',previous/'summary.json'])
    out.mkdir(parents=True);state={'created_utc':now(),'status':'building','pid':os.getpid(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'CUDA_HOME':os.environ.get('CUDA_HOME'),'TORCH_CUDA_ARCH_LIST':os.environ.get('TORCH_CUDA_ARCH_LIST'),
        'CC':os.environ.get('CC'),'CXX':os.environ.get('CXX'),'MAX_JOBS':os.environ.get('MAX_JOBS'),
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)},'selected_cases':cases,
        'pinned_rebuilt_binaries':pinned,'scope':'New rebuilt-binary diagnostic with strict whole-image output replay.'}
    save(out/'manifest.json',state);save(out/'selected_cases.json',cases);save(out/'pinned_rebuilt_binaries.json',pinned)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for source in sorted(sources):z.write(source,source.relative_to(ROOT).as_posix())
        for n,v in probe_payloads().items():z.writestr('generated/'+n,v)
    try:execute(out,state,previous,cases)
    except BaseException as exc:
        state.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc));save(out/'manifest.json',state)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
