"""CPU receipt audit against sealed v23 references; does not rerun CUDA checks."""
import json
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_local_image_loss_v24';V23=ROOT/'experiments/tgs_view_transfer_v23'

def main():
    plan=json.loads((OUT/'plan.json').read_text())
    for n,h in plan['inputs'].items():assert sha(ROOT/n)==h,n
    result=json.loads((OUT/'results.json').read_text());assert result['all_passed'] and len(result['rows'])==32 and not result['renderer_integration'] and not result['timing']
    assert result['new_native_forwards']==0 and result['source_sha256']==sha(ROOT/'run_tgs_local_image_loss_v24.py')
    b=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(b['path']))==b['sha256'] and sha(ROOT/'tgs_local_image_loss_v24.cu')==b['source_sha256']
    assert json.loads((OUT/'fixtures.json').read_text())['all_passed']
    refs=json.loads((V23/'analysis.json').read_text());lookup={(r['scene'],r['view'],r['event']):r for r in refs['rows']}
    assert len(lookup)==32 and len({(r['scene'],r['view'],r['event']) for r in result['rows']})==32
    maxima=dict(score_error=0.,blend_error=0.,full_score_error=0.,local_full_scalar_error=0.,CPU_map_error=0.)
    signs_cpu_gpu=0;ambiguous=0;png_disagreements=[0,0];primary_disagreements=[0,0];view_rows=[]
    for batch in result['batches']:
        raw=json.loads((OUT/f"{batch['scene']}_{batch['view']}.json").read_text());assert raw['rows']==[r for r in result['rows'] if r['scene']==batch['scene'] and r['view']==batch['view']]
        view_rows.append(dict(scene=batch['scene'],view=batch['view'],temporary_vertical_bytes=raw['temporary_vertical_bytes'],temporary_partial_bytes=raw['temporary_partial_bytes']))
    for row in result['rows']:
        ref=lookup[(row['scene'],row['view'],row['event'])];assert ref['parent_id']==row['parent_id'] and ref['appearance']==row['appearance'] and ref['rgb_sha256']==row['trial_hash']
        expected=[ref['changes'][k] for k in ['raw_MSE','raw_L1','raw_SSIM']];assert row['expected']==expected
        assert row['local_full_maps_literal'] and row['full_map_change_outside_A_literal_zero'] and row['normal_debug_scalars_literal']
        for key,values in [('score_error',row['absolute_error']),('blend_error',row['blend_absolute_error']),('full_score_error',row['full_absolute_error']),('local_full_scalar_error',row['local_full_scalar_error'])]:maxima[key]=max(maxima[key],max(values))
        maxima['CPU_map_error']=max(maxima['CPU_map_error'],row['max_CPU_map_error']);assert row['max_CPU_map_error']<=plan['map_tolerance']
        for actual,reference,errors in [(row['score'],expected,row['absolute_error']),(row['full_gpu_score'],expected,row['full_absolute_error']),(row['score'],row['full_gpu_score'],row['local_full_scalar_error']),(row['blends'],row['reference_blends'],row['blend_absolute_error'])]:
            actual=np.array(actual);reference=np.array(reference);assert np.array_equal(np.abs(actual-reference),errors) and np.all(np.abs(actual-reference)<=plan['score_atol']+plan['score_rtol']*np.abs(reference))
        for j in range(2):
            cpu=row['reference_blends'][j];gpu=row['blends'][j];signs_cpu_gpu+=int(np.sign(cpu)==np.sign(gpu));ambiguous+=row['ambiguous_blend_signs'][j]
            mismatch=np.sign(ref['changes']['PNG8_MSE'])!=np.sign(gpu);png_disagreements[j]+=int(mismatch)
            if row['appearance']=='moved':primary_disagreements[j]+=int(mismatch)
    assert signs_cpu_gpu==64
    out=dict(all_verified=True,maxima=maxima,rows=32,view_batches=4,CPU_GPU_blend_sign_agreements=signs_cpu_gpu,blend_comparisons=64,
        blend_signs_within_declared_scalar_allowance=ambiguous,PNG8_vs_blend_sign_disagreements_all32=png_disagreements,
        PNG8_vs_blend_sign_disagreements_primary16=primary_disagreements,planned_scratch_bytes_not_peak_memory=view_rows,
        new_native_forwards=0,training=False,renderer_integration=False,timing=False,not_an_independent_map_rerun=True,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'plan.json',OUT/'results.json',OUT/'build_receipt.json',V23/'analysis.json',Path(__file__)]})
    write_new(OUT/'receipt_audit.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':main()
