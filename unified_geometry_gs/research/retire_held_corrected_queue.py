"""Retire only the verified stopped queue after its control child completed."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/experiments/corrected_quality55_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    control=OUT/'legacy64';record=json.loads((control/'corrected_training_integration.json').read_text())
    ply=control/'point_cloud/iteration_30000/point_cloud.ply'
    if record['status']!='complete' or record['steps_observed']!=23000 or not ply.is_file():raise ValueError('Control not complete.')
    if (OUT/'corrected64').exists():raise ValueError('Original candidate already started.')
    proc=Path('/proc/405');cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'research/run_corrected_quality_pair.py' not in cmd or '--out research/experiments/corrected_quality55_v1' not in cmd:raise ValueError('PID identity mismatch.')
    status=(proc/'status').read_text()
    if not any(x.startswith('State:\tT') for x in status.splitlines()):raise ValueError('Queue not stopped.')
    child=Path('/proc/425/status').read_text()
    if not any(x.startswith('State:\tZ') for x in child.splitlines()):raise ValueError('Control child not finished.')
    before=OUT/'manifest_before_supersede.json'
    if before.exists():raise ValueError('Retirement already recorded; inspect state.')
    before.write_bytes((OUT/'manifest.json').read_bytes())
    # Pending default SIGTERM is delivered on SIGCONT before Python resumes;
    # do not resume a live queue and allow it to dispatch the rejected candidate.
    os.kill(405,signal.SIGTERM);os.kill(405,signal.SIGCONT)
    for _ in range(50):
        if not proc.exists():break
        time.sleep(.1)
    if proc.exists():raise ValueError('Held queue still exists; no state rewritten.')
    state=json.loads(before.read_text());state['status']='superseded_before_candidate'
    state['superseded_utc']=datetime.now(timezone.utc).isoformat()
    state['superseded_reason']='Original all64 candidate failed the pre-quality extreme-axis derivative gate; direct-factor replacement requires a separate protocol.'
    for step in state['steps']:
        if step['stage']=='train_legacy64':step.update(status='training_artifacts_complete',returncode=None,
            completion_evidence='legacy64/corrected_training_integration.json; parent return code was not collected before termination')
    (OUT/'manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    deviation={'created_utc':state['superseded_utc'],'status':'retired','verified_pid':405,
        'verified_cmdline':cmd,'control_child_pid':425,'candidate_started':False,
        'action':'SIGTERM then SIGCONT to stopped parent after control child became zombie; parent disappearance verified',
        'control_integration_sha256':sha(control/'corrected_training_integration.json'),'control_model_sha256':sha(ply),
        'original_manifest_sha256':sha(before),'replacement':'factor64, subject to all 44 stress and all 120 regular-case gates and complete-backend smoke',
        'control_reuse_reason':'Preselected same-FP64 control completed before candidate or mesh/image quality was observed; retain this control regardless of subsequent outcome.',
        'goal_status':'active'}
    (OUT/'protocol_deviation_factor_v1.json').write_text(json.dumps(deviation,indent=2)+'\n')
    print(json.dumps(deviation,indent=2))
if __name__=='__main__':main()
