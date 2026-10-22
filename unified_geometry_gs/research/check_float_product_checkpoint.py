"""Terminal certificate stage, actual compiled note and recoverable archive."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    out=R/'results/float_product_checkpoint_v1.json';assert not out.exists();p=R/'snapshots/after_float_product_v1.zip';m=read(p.with_suffix('.manifest.json'));assert sha(p)==m['archive_sha256']
    c=read(R/'results/latex_compile_status_v34.json');assert c['kind']=='success' and sha(R/'median_depth_note.tex')==c['source_sha256']
    g=read(R/'results/canvas_float_product_guard_v1.json');assert sha(Path(g['destination']))==g['staged_sha256'] and not read(R/'results/canvas_float_product_syntax_v1.json')['errors']
    s=read(R/'experiments/float_product_v1/summary.json');a=read(R/'results/float_product_audit_v1.json');f=read(R/'results/float_product_failure_analysis_v1.json');assert s['status']=='complete' and a['gate_pass'] and a['world_probe_records']==145859 and f['counts']=={'root_after_error_band':114,'root_before_error_band':18}
    for n,h in f['sources_sha256'].items():assert sha(R.parent/n)==h,n
    for n,h in f['inputs_sha256'].items():assert sha(R.parent/n)==h,n
    names=['research/float_product.cuh','research/float_product_cuda.py','research/benchmark_float_product.py','research/verify_float_product.py','research/FLOAT_PRODUCT_PROTOCOL.md','research/FLOAT_PRODUCT_RESULT.md','research/experiments/float_product_v1/summary.json','research/experiments/float_product_v1/scan55_0000_probes.npz','research/experiments/float_product_v1/scan55_0000_output.npz','research/results/float_product_audit_v1.json','research/results/float_product_failure_analysis_v1.json','research/median_depth_note.tex','research/results/latex_compile_status_v34.json','research/canvas_result_snapshot.tsx']
    with zipfile.ZipFile(p) as z:
        assert json.loads(z.read('SNAPSHOT_MANIFEST.json'))['files_sha256']==m['files_sha256']
        for n in names:assert hashlib.sha256(z.read(n)).hexdigest()==m['files_sha256'][n]
    result=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,current_goal_turn_classification='progress: all24 original-depth sufficient-condition certificates plus independent full-ray probes and exploratory complete failure subset',totals=s['totals'],valid_original_certified_fraction=s['totals']['certified']/s['totals']['original_valid'],MP80_140_coefficient_checks=a['coefficient_checks'],world_probe_rays=1545,world_probe_records=145859,scalar_checks=4104,probe_failures=[],exploratory_unresolved=f['counts'],terminal_handles=dict(main_session26752=0,independent_session10530=0,exploratory_analysis_exit_code=0),source_latex_matches_builtin_success=True,source_canvas_matches_syntax_checked_stage=True,snapshot=dict(sha256=m['archive_sha256'],files=len(m['files_sha256']),bytes=p.stat().st_size,selected_member_checks=names),limitations=['Only sufficient-condition checks of original depths; unresolved root correction not implemented.','Probe mathematical error is not GT reconstruction error; conditional failure subset is not population error-rate estimate.','57.714–134.755ms/view prototype remains slower than original renderer; no full efficiency/training/novelty claim.','Canvas syntax only, SDK/runtime unverified; historical rows24 constant not active tab count.'],next='Dynamic-length ray-parallel strict correction for unresolved sources; then world VJP, complete costs and GT/holdout training gates. Do not relaunch completed diagnostics or claim research goal achieved.')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['limitations','next']},indent=2))
if __name__=='__main__':main()
