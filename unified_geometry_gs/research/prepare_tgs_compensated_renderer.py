"""Generate isolated v8 source/tests and freeze inherited inputs BEFORE native outputs."""
import hashlib,json,shutil
from pathlib import Path
from tgs_compensated_common import ROOT,OUT,sha
def once(t,a,b):assert t.count(a)==1,(a,t.count(a));return t.replace(a,b)
def main():
    assert not OUT.exists();OUT.mkdir();v5=ROOT/'experiments/tgs_phase_cost_v5';v7=ROOT/'experiments/tgs_real_work_v7'
    expected={v5:'c8ab7c32e93b5aca6a830d41fdc04b2bc1e8205b852a9a0a39f6bfdd8a9af08d',v7:'9f246c3a4638b29fdae09680f49a74a20f910679a4e2cd03617251b001490aee'}
    for folder,h in expected.items():
        p=folder/'evidence_manifest.json';assert sha(p)==h
        for e in json.loads(p.read_text())['files']:assert sha(ROOT/e['path'])==e['sha256']
    dst=OUT/'generated/native';shutil.copytree(v5/'generated/native',dst);cu=dst/'cuda_rasterizer'
    (cu/'stable_window.cuh').write_text((ROOT/'tgs_compensated_window_v8.cuh').read_text())
    for name in ['forward.cu','backward.cu']:
        p=cu/name;t=p.read_text();assert 'stable_error+10*' in t;t=t.replace('stable_error+10*','stable_error+48*');t=t.replace('stable_error[10*idx+','stable_error[48*idx+')
        if name=='forward.cu':t=once(t,'rect_min[idx] = rect_min_triangle_test;','rect_min[idx] = rect_min_triangle_test;\n    comp_precompute(stable_error+48*idx+10,stable_coeff+10*idx,center_triangle_2D,rect_min_triangle_test,rect_max_triangle_test,W,H);')
        p.write_text(t)
    p=cu/'rasterizer_impl.cu';p.write_text(once(p.read_text(),'obtain(chunk, geom.stable_error, P*10, 128);','obtain(chunk, geom.stable_error, P*48, 128);'))
    # Reuse exact frozen v5 test logic, explicitly adapting the new cache reader and camera cohort.
    for old,new in [('probe_tgs_phase_cases.py','probe_tgs_compensated_cases.py'),('probe_tgs_phase_fd.py','probe_tgs_compensated_fd.py'),('probe_tgs_phase_real.py','probe_tgs_compensated_real.py'),('audit_tgs_phase_binary.py','audit_tgs_compensated_binary.py')]:
        p=ROOT/new;assert not p.exists();t=(ROOT/old).read_text().replace('from tgs_phase_common import','from tgs_compensated_common import').replace('from probe_tgs_compositing_guard import state','from tgs_compensated_common import state').replace('from probe_tgs_adaptive_window import decode','from tgs_compensated_common import decode').replace('TGS_PHASE_COST_PROTOCOL_V5.md','TGS_COMPENSATED_RENDERER_PROTOCOL_V8.md')
        if old=='probe_tgs_phase_real.py':
            t=t.replace('from tgs_compensated_common import ROOT,OUT,VARIANTS,sha,modules,workload','from tgs_compensated_common import ROOT,OUT,VARIANTS,sha,modules,workload,v5_oracle')
            t=once(t,'mods=modules();work=workload();','mods=modules();work=workload();oldmod=v5_oracle();')
            t=once(t,'for ci in range(4):','for ci in range(len(work.cameras)):')
            start=t.index('        old=None');end=t.index('        finite=all',start)
            t=t[:start]+'''        oldres=work.forward(oldmod,ci);torch.cuda.synchronize()
        old=dict(rgb=float(np.abs(oldres[1].cpu().numpy()-b['rgb']).max()),aux=np.abs(oldres[2].cpu().numpy()-b['aux']).reshape(7,-1).max(1).tolist(),radii_equal=bool(np.array_equal(oldres[3].cpu().numpy(),b['radii'])),maxblend=float(np.abs(oldres[9].cpu().numpy()-b['maxblend']).max()))
        assert old['rgb']==0 and max(old['aux'])==0 and old['maxblend']==0 and old['radii_equal']
        del oldres
        regression=None
        if ci<2:
            x,y=([116,742] if ci==0 else [740,268]);regression=dict(pixel=[x,y],flag=int(a['flags'][y,x]),RGB_abs=float(np.abs(a['rgb'][:,y,x]-b['rgb'][:,y,x]).max()),aux_abs=np.abs(a['aux'][:,y,x]-b['aux'][:,y,x]).tolist(),paths_identical=bool(np.array_equal(a['contributors'][:,y,x],b['contributors'][:,y,x])))
            assert regression['paths_identical'] and max(regression['aux_abs'])<=3e-5
''' +t[end:]
            t=t.replace('old_v4_oracle_comparison=old','old_v5_oracle_comparison=old,regression_pixel=regression')
        p.write_text(t)
    prior=json.loads((v5/'input_plan.json').read_text());vp=json.loads((v7/'input_plan.json').read_text());cams=vp['cameras'];cp=ROOT/'sources/tgs_real_checkpoint_v1/stump/cameras.json';allcam=json.loads(cp.read_text());tests=sorted(allcam,key=lambda c:c['img_name'])[::8]
    used={c['img_name'] for c in cams};new=min([c for c in tests if c['img_name'] not in used],key=lambda c:hashlib.sha256(('tgs_compensated_renderer_v8/'+c['img_name']).encode()).hexdigest());cams=cams+[dict(new,split='test',heldout_for_v8=True)]
    prior.update(parent_plan_sha256=sha(ROOT/'experiments/tgs_compositing_guard_v4/input_plan_extended.json'),v5_plan_sha256=sha(v5/'input_plan.json'),protocol_sha256=sha(ROOT/'TGS_COMPENSATED_RENDERER_PROTOCOL_V8.md'),phase_common_sha256=sha(ROOT/'tgs_compensated_common.py'),helper_sha256=sha(ROOT/'tgs_compensated_window_v8.cuh'),prepare_sha256=sha(Path(__file__)),resolution=vp['resolution'],cameras=cams,camera_source_sha256=sha(cp),prior_manifests={folder.name:h for folder,h in expected.items()},sources={p.relative_to(ROOT).as_posix():sha(p) for p in dst.rglob('*') if p.is_file()},test_scripts={name:sha(ROOT/name) for name in ['probe_tgs_compensated_cases.py','probe_tgs_compensated_fd.py','probe_tgs_compensated_real.py','audit_tgs_compensated_binary.py']},training=False,timing=False)
    (OUT/'input_plan.json').write_text(json.dumps(prior,indent=2)+'\n');print(json.dumps(dict(cases=len(prior['cases']),cameras=[c['img_name'] for c in cams],new_holdout=new['img_name'],input_plan_sha256=sha(OUT/'input_plan.json'))),flush=True)
if __name__=='__main__':main()
