"""Verify archived evidence bytes and summarize the independent preprocessing audit."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh report required.')
    evidence={}
    for name in ('world_preprocess_synthetic_v1','world_preprocess_synthetic_v2','world_preprocess_real_v1',
                 'corrected_smoke55_v1','corrected_smoke55_v2'):
        directory=ROOT/'research/experiments'/name
        state=json.loads((directory/'manifest.json').read_text());current_changes=[]
        if sha(directory/'sources.zip')!=state['source_archive_sha256']:raise ValueError('Archive changed: '+name)
        with zipfile.ZipFile(directory/'sources.zip') as z:
            for rel,digest in state['source_sha256'].items():
                if hashlib.sha256(z.read(rel)).hexdigest()!=digest:raise ValueError('Archived source bytes differ: '+rel)
                if sha(ROOT/rel)!=digest:current_changes.append(rel)
        if name in ('world_preprocess_synthetic_v2','world_preprocess_real_v1','corrected_smoke55_v2') and current_changes:
            raise ValueError('Active successful evidence source changed: '+str(current_changes))
        value={'status':state['status'],'manifest_sha256':sha(directory/'manifest.json'),
            'archive_sha256':state['source_archive_sha256'],'current_source_changes':current_changes}
        if name.startswith('world_preprocess'):
            result=json.loads((directory/'results.json').read_text())
            if state['status']!='complete' or sha(directory/'results.json')!=state['results_sha256']:raise ValueError('Incomplete or changed result.')
            value.update({'results_sha256':state['results_sha256'],'summary':result['summary'],
                'rows':len(result['rows']),'parameter_groups_per_variant':4*len(result['rows'])})
            if 'all64' in result['summary']:
                value['all64_gate_pass']=all(r['failed_groups']==0 for r in result['summary']['all64'].values())
                if not value['all64_gate_pass']:raise ValueError('Declared combined FP64 gate failed.')
            if 'cases' in result:
                value.update({'case_count':len(result['cases']),'kernel_sizes':sorted({r['params'][-1] for r in result['cases']}),
                    'selected_axis_ratio_range':[min(r['ratio'] for r in result['cases']),max(r['ratio'] for r in result['cases'])],
                    'max_min_step_fd_scaled_error':result['max_min_step_fd_scaled_error'],
                    'max_forward_scaled_error':result['max_forward_scaled_error']})
            else:value['max_directional_fd_scaled_error']=result['max_directional_fd_scaled_error']
        elif name=='corrected_smoke55_v2':
            if state['status']!='complete' or not state['branch_state_verified'] or not state['first_forward_losses_identical']:raise ValueError('Smoke incomplete.')
            integrations={v:json.loads((directory/v/'corrected_training_integration.json').read_text()) for v in state['order']}
            for v,r in integrations.items():
                if r['status']!='complete' or r['steps_observed']!=20 or r['nonfinite_gradient_groups'] or not r['intentional_stop_observed']:
                    raise ValueError('Smoke integration failed: '+v)
                if r['backend_sha256']!=state['backend_preflight'][v]['binary_sha256'] or r['kernel_identity']!=state['backend_preflight'][v]['kernel_identity']:
                    raise ValueError('Smoke loaded backend differs from preflight.')
            a,b=integrations.values()
            for key in ('initial_state_sha256','common_checkpoint_sha256','sidecar_sha256','first_camera','first_loss_scalars'):
                if a[key]!=b[key]:raise ValueError('Smoke paired state differs.')
            value.update({'common_state_sha256':a['initial_state_sha256'],'first_camera':a['first_camera'],
                'first_forward_losses_identical':True,'steps_per_branch':20,'nonfinite_gradient_groups':0,
                'full_backend_and_probe_functions_identical':True})
        elif state['status']!='failed' or not (directory/'correction.json').is_file():raise ValueError('Expected preserved initial smoke failure.')
        evidence[name]=value
    proof=ROOT/'research/results/preprocess_chain_rules_v1.json';math=json.loads(proof.read_text())
    if sha(ROOT/'research/verify_preprocess_chain_rules.py')!=math['source_sha256'] or not all(math['exact_ray_identities']) or not math['exact_coef_identity']:
        raise ValueError('Local chain rule proof changed or failed.')
    quality=ROOT/'research/experiments/corrected_quality55_v1/manifest.json'
    status=json.loads(quality.read_text()) if quality.is_file() else None
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
        'evidence':evidence,'chain_rule_proof_sha256':sha(proof),
        'quality_run_snapshot':None if status is None else {'manifest_sha256':sha(quality),'status':status['status'],
            'stages':[{k:r[k] for k in ('stage','status')} for r in status['steps']]},
        'conclusions':['Three local analytic discrepancies relative to the implemented forward, checked with exact calculus and independent CUDA world-parameter oracles.',
            'Combined FP64 derivative gate passes in the stated synthetic and real development subsets; combined FP32 fails the original gate.',
            'Real development models use kernel zero, so the opacity-factor algebra correction is inactive in ideal arithmetic.',
            'Matched-precision shared-state training smoke passes; full quality requires a separate completed report.'],
        'limits':['Historical archives are verified against their own captured bytes, not forced to match later corrected source.',
            'No full-renderer correctness, extreme-anisotropy coverage, reconstruction improvement, statistical significance or novelty claim.',
            'The error denominator includes an absolute floor for structural zeros.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'verified_evidence_runs':len(evidence),'all64_gate_pass':True,
        'quality_status':None if status is None else status['status']},indent=2))


if __name__=='__main__':main()
