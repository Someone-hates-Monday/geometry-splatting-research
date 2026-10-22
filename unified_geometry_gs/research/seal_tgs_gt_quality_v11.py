"""Seal verified GT identity, all fixed-weight quality outcomes and direction evidence."""
import ast,datetime,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import scipy,torch,PIL
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_gt_quality_v11';SOURCE=ROOT/'sources/tgs_gt_identity_v11';V10=ROOT/'experiments/tgs_cross_scene_v10'
V10_SHA='9d8b5e537c4110881a6ea1696c2c1f41a93ec77cb8ab4e5eba4006c5095ee463'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b=f.read(8*1024*1024)
            if not b:break
            h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def dependencies(paths):
    found=set(paths);queue=[p for p in found if p.suffix=='.py']
    while queue:
        p=queue.pop()
        for node in ast.walk(ast.parse(p.read_text())):
            names=[node.module] if isinstance(node,ast.ImportFrom) and node.module else ([x.name for x in node.names] if isinstance(node,ast.Import) else [])
            for name in names:
                q=ROOT/(name.split('.')[0]+'.py')
                if q.is_file() and q not in found:found.add(q);queue.append(q)
    return found
def main():
    target=OUT/'evidence_manifest.json';assert not target.exists();identity=read(OUT/'identity_audit.json');audit=read(OUT/'independent_quality_audit.json');quality=read(OUT/'quality_results.json');plan=read(OUT/'quality_plan.json');render=read(OUT/'render_results.json');rp=read(OUT/'render_plan.json');selection=read(SOURCE/'selection.json');acq=read(SOURCE/'manifest.json')
    assert identity['all_gates'] and identity['quality_measurement_allowed'] and all(r['different_channel_values']==0 and r['original_decoded_RGB_sha256']==r['GT_decoded_RGB_sha256'] for r in identity['rows']);assert len(identity['rows'])==2
    assert audit['all_verified'] and audit['all_variants_retained'] and audit['primary_MSE_exact_integer_verified'] and audit['full_SSIM_independent_filter_order_verified'] and audit['direct_SSIM_patch_channel_checks']==3888
    assert audit['results_sha256']==sha(OUT/'quality_results.json') and audit['script_sha256']==sha(ROOT/'audit_tgs_gt_quality_v11.py')
    assert quality['metrics_sha256']==sha(ROOT/'tgs_image_metrics_v11.py') and quality['script_sha256']==sha(ROOT/'analyze_tgs_gt_quality_v11.py') and quality['render_results_sha256']==sha(OUT/'render_results.json')
    for name,h in plan['scripts'].items():assert sha(ROOT/name)==h
    assert plan['identity_audit_sha256']==sha(OUT/'identity_audit.json') and plan['metric_selfcheck_sha256']==sha(OUT/'metric_selfcheck.json') and read(OUT/'metric_selfcheck.json')['all_verified']
    assert plan['protocol_sha256']==selection['protocol_sha256']==sha(ROOT/'TGS_GT_IDENTITY_QUALITY_PROTOCOL_V11.md') and plan['control_addendum_sha256']==selection['control_addendum_sha256']==sha(ROOT/'TGS_GT_CONTROL_ADDENDUM_V11.md')
    assert render['model_inputs_unchanged'] and len(render['rows'])==8 and len(quality['rows'])==8 and rp['quality_plan_sha256']==sha(OUT/'quality_plan.json')
    for v,m in rp['modules'].items():assert sha(Path(m['path']))==m['sha256']
    for row in render['rows']:
        assert row['all_RGB_finite']
        if row['v10_forward_literal_bits'] is not None:assert all(row['v10_forward_literal_bits'].values())
        for key,h in row['files'].items():assert sha(OUT/row['scene']/row['variant']/(key+'.npy'))==h
    assert acq['total_cached_range_payload_bytes']<=96*1024*1024 and acq['selection_sha256']==sha(SOURCE/'selection.json')
    for en in acq['files']:assert sha(ROOT/en['local'])==en['sha256'] and en['zip_crc_checked']
    assert sha(V10/'evidence_manifest.json')==V10_SHA;prior=read(V10/'evidence_manifest.json')['files']
    for i,en in enumerate(prior):
        assert sha(ROOT/en['path'])==en['sha256']
        if i%200==199:print(json.dumps(dict(stage='prior_hash_verified',count=i+1,total=len(prior))),flush=True)
    fm=read(OUT/'figure_metadata.json');assert fm['script_sha256']==sha(ROOT/'plot_tgs_gt_quality_v11.py') and fm['results_sha256']==sha(OUT/'quality_results.json') and fm['audit_sha256']==sha(OUT/'independent_quality_audit.json')
    for name,h in fm['files'].items():assert sha(OUT/name)==h
    decision=dict(GT_identity_verified=True,quality_metrics_independently_verified=True,full_exact_window_research_priority='Lower as a standalone quality/efficiency method; retain diagnostic reference.',coarse_piecewise_PSNR_deltas=[c['coarse_minus_piecewise']['primary_PNG8']['PSNR_dB'] for c in quality['contrasts']],coarse_archived_PSNR_deltas=[c['coarse_minus_archived_native']['primary_PNG8']['PSNR_dB'] for c in quality['contrasts']],specific_edit_cause_established=False,retrained_quality_effect_established=False,geometry_quality_evaluated=False,novelty_established=False,training_run=False,scientific_goal_complete=False,next='Single-source-edit causal diagnostic, fresh camera validation and appearance-function reparameterization analysis; do not infer a cause from mixed variants.')
    (OUT/'research_direction_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    (OUT/'figure_visual_review.json').write_text(json.dumps(dict(file='quality_views.png',sha256=sha(OUT/'quality_views.png'),all_full_images_retained=True,labels_metrics_and_colorbar_visible=True,shared_scale_saturation_disclosed=True,no_alignment_or_color_adjustments=True),indent=2)+'\n')
    (OUT/'runtime_info.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,torch_cuda_build=torch.version.cuda,Pillow=PIL.__version__,platform=platform.platform()),indent=2)+'\n')
    names=['TGS_GT_IDENTITY_QUALITY_PROTOCOL_V11.md','TGS_GT_CONTROL_ADDENDUM_V11.md','TGS_GT_QUALITY_RESULT_2026_10_21.md','remote_public_zip_v11.py','catalogue_tgs_gt_sources_v11.py','extract_tgs_gt_identity_v11.py','audit_tgs_gt_identity_v11.py','tgs_image_metrics_v11.py','selfcheck_tgs_image_metrics_v11.py','render_tgs_gt_quality_v11.py','analyze_tgs_gt_quality_v11.py','audit_tgs_gt_quality_v11.py','plot_tgs_gt_quality_v11.py','seal_tgs_gt_quality_v11.py'];sources=dependencies({ROOT/name for name in names});files=sorted(sources|{p for d in [OUT,SOURCE] for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts});entries=[]
    for p in files:entries.append(dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)))
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries,source_dependency_files=len(sources),prior_v10_hash_checks=len(prior),GT_identity_verified=True,quality_metric_audit=True,standalone_method_promotion=False,training=False,scientific_goal_complete=False);target.write_text(json.dumps(manifest,indent=2)+'\n')
    for en in entries:assert sha(ROOT/en['path'])==en['sha256']
    print(json.dumps(dict(files=len(entries),bytes=sum(e['size'] for e in entries),source_dependency_files=len(sources),manifest_sha256=sha(target),goal_active=True)),flush=True)
if __name__=='__main__':main()
