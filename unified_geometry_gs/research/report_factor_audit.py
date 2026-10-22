"""Verify actual archived bytes and summarize the bounded factor audit."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    evidence={}
    for name in ('world_preprocess_stress_v1','factor_preprocess_stress_v1','factor_preprocess_regular_v1'):
        d=ROOT/'research/experiments'/name;s=read(d/'manifest.json');r=read(d/'results.json')
        if s['status']!='complete' or sha(d/'results.json')!=s['results_sha256'] or sha(d/'sources.zip')!=s['source_archive_sha256']:raise ValueError('Artifact hash/status mismatch.')
        with zipfile.ZipFile(d/'sources.zip') as z:
            for n,h in s['source_sha256'].items():
                if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Archive mismatch: '+n)
            generated=[n for n in z.namelist() if n.startswith('generated/')]
            if not generated:raise ValueError('Generated CUDA bytes missing.')
        evidence[name]={'manifest_sha256':sha(d/'manifest.json'),'source_archive_sha256':s['source_archive_sha256'],
            'verified_archived_sources':len(s['source_sha256']),'generated_files':len(generated),'results_sha256':s['results_sha256'],
            'summary':r['summary']['all64'] if name.startswith('world_') else r['summary']['factor64']['physical'],
            'gate_pass':all(x['failed_groups']==0 and x['nonfinite_groups']==0 for x in r['summary']['all64'].values()) if name.startswith('world_') else r['factor64_primary_gate_pass']}
    for name in ('factor_preprocess_stress_v1','factor_preprocess_regular_v1'):
        if not evidence[name]['gate_pass']:raise ValueError('Admission failure.')
    if evidence['world_preprocess_stress_v1']['gate_pass']:raise ValueError('Expected retained roundtrip failure missing.')
    proof=ROOT/'research/results/precision_roundtrip_defect_v1.json';math=read(proof)
    if sha(ROOT/'research/verify_precision_roundtrip_defect.py')!=math['source_sha256'] or not all(math['symbolic_quaternion_components_pass']) or not math['all_bound_checks_pass']:raise ValueError('Proof check changed/failed.')
    compilefile=ROOT/'research/results/latex_compile_status_v8.json';compiled=read(compilefile)
    if compiled['status']!='success' or sha(ROOT/'research/median_depth_note.tex')!=compiled['source_sha256']:raise ValueError('LaTeX source not compiled.')
    pair=ROOT/'research/experiments/factor_quality55_v1';q=read(pair/'manifest.json')
    with zipfile.ZipFile(pair/'sources.zip') as z:
        if sha(pair/'sources.zip')!=q['source_archive_sha256']:raise ValueError('Quality archive changed.')
        for n,h in q['source_sha256'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h or sha(ROOT/n)!=h:raise ValueError('Queued source mismatch: '+n)
    smoke=pair/'factor64_smoke/factor_training_integration.json';s=read(smoke)
    control_reference=q['control_reference']
    # WSL paths in manifests are mapped for the Windows read-only reporter.
    if control_reference.startswith('/mnt/c/'):
        control_reference='C:/'+control_reference[7:]
    control=Path(control_reference)
    c=read(control/'corrected_training_integration.json')
    for k in ('common_checkpoint_sha256','sidecar_sha256','initial_state_sha256','first_camera','first_loss_scalars','initial_gaussians'):
        if s[k]!=c[k]:raise ValueError('Smoke differs from fixed control: '+k)
    if s['status']!='complete' or s['steps_observed']!=20 or s['nonfinite_gradient_groups'] or not s['kernel_identity']['identical_function_bytes']:raise ValueError('Smoke failed.')
    total=sum(x['total_groups'] for n in ('factor_preprocess_stress_v1','factor_preprocess_regular_v1') for x in evidence[n]['summary'].values())
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'evidence':evidence,'factor_parameter_groups_passed':total,
        'case_count':164,'proof_sha256':sha(proof),'latex_status_sha256':sha(compilefile),
        'max_precision_covariance_identity_residual':math['max_precision_covariance_residual'],
        'quality_snapshot':{'status':q['status'],'current_stage':q.get('current_stage'),'manifest_sha256_at_snapshot':sha(pair/'manifest.json'),
            'preflight':q['factor_backend'],'smoke_integration_sha256':sha(smoke),'retained_control_integration_sha256':sha(control/'corrected_training_integration.json')},
        'limits':['All 3664 checks are local parameter blocks, including structural zeros, not independent statistical units.',
            'Four original extreme forward-component failures remain; forward is unchanged.',
            'Combined correction, arithmetic and factorization intervention; no single-change attribution.',
            'Quality snapshot is not live monitoring or evidence of improvement. Full-renderer correctness and novelty remain unestablished.']}
    out=ROOT/'research/results/factor_audit_report_v1.json'
    if out.exists():raise ValueError('Keep prior reports; use a new version.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'passed_local_groups':total,'quality_snapshot':result['quality_snapshot'],'archives_verified':True},indent=2))
if __name__=='__main__':main()
