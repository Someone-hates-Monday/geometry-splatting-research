"""Verify actual archive/build bytes and the bounded 20-update result."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    run=ROOT/'research/experiments/gram_smoke55_v1';s=read(run/'manifest.json')
    p=run/'gram_factor64/gram_training_integration.json';rec=read(p)
    ref=ROOT/'research/experiments/factor_quality55_v1/factor64_smoke/factor_training_integration.json';r=read(ref)
    if s['status']!='complete' or not s['smoke_only'] or s['full_quality_training_started'] or s['returncode']!=0:raise ValueError('Smoke not complete or scope changed.')
    if sha(p)!=s['integration_sha256'] or sha(ref)!=s['reference_integration_sha256'] or rec['status']!='complete' or rec['steps_observed']!=20 or rec['nonfinite_gradient_groups']:raise ValueError('Integration mismatch.')
    for k in ('initial_state_sha256','common_checkpoint_sha256','sidecar_sha256','first_camera','initial_gaussians'):
        if rec[k]!=r[k]:raise ValueError('Matched state differs: '+k)
    if rec['backend_sha256']!=s['backend_sha256'] or rec['kernel_identity']!=s['kernel_identity']:raise ValueError('Preflight differs from loaded training backend.')
    errors={k:abs(v-r['first_loss_scalars'][k])/max(abs(r['first_loss_scalars'][k]),1e-5) for k,v in rec['first_loss_scalars'].items()}
    if errors!=s['first_loss_scaled_errors'] or max(errors.values())>1e-4:raise ValueError('Predefined first-loss gate failed.')
    if sha(run/'sources.zip')!=s['source_archive_sha256']:raise ValueError('Archive changed.')
    binary=[p for p in (ROOT/'research/build').glob('gram_factor64_*/*.so') if sha(p)==s['backend_sha256']]
    if len(binary)!=1:raise ValueError('Unique actual binary not found.')
    with zipfile.ZipFile(run/'sources.zip') as z:
        for n,h in s['source_sha256'].items():
            if sha(ROOT/n)!=h or hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Actual/archived source differs: '+n)
        generated=[n for n in z.namelist() if n.startswith('generated/gram_factor64/')]
        for n in generated:
            relative=n.removeprefix('generated/gram_factor64/')
            if (binary[0].parent/'src'/relative).read_bytes().replace(b'\r\n',b'\n')!=z.read(n).replace(b'\r\n',b'\n'):raise ValueError('Actual generated build source differs: '+relative)
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'actual_archive_bytes_verified':True,
        'source_count':len(s['source_sha256']),'generated_build_files_verified':len(generated),
        'actual_binary':str(binary[0]),'actual_binary_sha256':sha(binary[0]),'integration_sha256':sha(p),
        'same_complete_initial_state':True,'finite_20_updates':True,'first_loss_gate_pass':True,'max_first_loss_scaled_error':max(errors.values()),
        'source_archive_sha256':s['source_archive_sha256'],'kernel_identity':s['kernel_identity'],
        'full_quality_training_started':False,'limits':s['limits']}
    out=ROOT/'research/results/gram_smoke_report_v1.json'
    if out.exists():raise ValueError('Keep prior report.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='kernel_identity'},indent=2))
if __name__=='__main__':main()
