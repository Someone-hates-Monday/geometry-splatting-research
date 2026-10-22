"""Reuse and extend the independently checked v5 precision audit, before test outcomes."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=ROOT/'audit_tgs_compensated_precision.py';assert not p.exists();t=(ROOT/'audit_tgs_phase_precision.py').read_text()
    names={'tgs_phase_common':'tgs_compensated_common','probe_tgs_phase_cases':'probe_tgs_compensated_cases','probe_tgs_phase_fd':'probe_tgs_compensated_fd','probe_tgs_phase_real':'probe_tgs_compensated_real','audit_tgs_phase_binary':'audit_tgs_compensated_binary','tgs_phase_cost_backend':'tgs_compensated_renderer_backend','TGS_PHASE_COST_PROTOCOL_V5':'TGS_COMPENSATED_RENDERER_PROTOCOL_V8'}
    for a,b in names.items():t=t.replace(a,b)
    t=t.replace("'tgs_compositing_guard_v4']:","'tgs_compositing_guard_v4','tgs_phase_cost_v5','tgs_compensated_edge_v6','tgs_real_work_v7']:")
    s=t.index('    required=[');end=t.index('    index=',s)
    t=t[:s]+'''    required=['__fsub_rn(pixel.x,anchor.x)','comp_dot2(p+6*k,x,y)','__fsub_rd(r[k],e[k])','__fadd_ru(r[k],e[k])','const float E=e[z.active]','comp_distance_down(z.alpha,1.f/255.f)','__dmul_rn(-best,coeff[9])','__double2float_rn(fmin(.99,raw))']
    assert all(s in helper for s in required) and '__fmul_rn(T,__fsub_rn(1.0f,alpha))' in forward and '#if WINDOW_COUNTERS' in forward
    assert 'stable_error+48*j_id' in forward and 'comp_precompute(stable_error+48*idx+10' in forward
    assert 'obtain(chunk, geom.stable_error, P*48, 128)' in (OUT/'generated/native/cuda_rasterizer/rasterizer_impl.cu').read_text()
    assert 'stable_error+48*window_id' in (OUT/'generated/native/cuda_rasterizer/backward.cu').read_text()
    baseline=json.loads((OUT/'baseline_cache_audit.json').read_text());assert baseline['all_gates'] and baseline['script_sha256']==sha(ROOT/'audit_tgs_compensated_baseline.py')
''' +t[end:]
    t=t.replace("==['_DSC9258','_DSC9322','_DSC9269','_DSC9221']","==[c['img_name'] for c in plan['cameras']]")
    t=t.replace("==['train','train','test','test']","==['train','train','test','test','test','test']")
    t=t.replace("real_rows.append(dict(camera=ci,","assert rr['old_v5_oracle_comparison']['rgb']==0 and max(rr['old_v5_oracle_comparison']['aux'])==0\n        real_rows.append(dict(camera=ci,")
    t=t.replace("binary_audit_sha256=sha(OUT/'binary_audit.json'),","binary_audit_sha256=sha(OUT/'binary_audit.json'),baseline_cache_audit_sha256=sha(OUT/'baseline_cache_audit.json'),")
    p.write_text(t);print(str(p))
if __name__=='__main__':main()
