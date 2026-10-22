"""Descriptive summary of every frozen case/FD; no new acceptance thresholds."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_stable_window_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=OUT/'diagnostic_summary.json';assert not target.exists()
    results=json.loads((OUT/'results.json').read_text());audit=json.loads((OUT/'independent_audit.json').read_text())
    v1=json.loads((ROOT/'experiments/tgs_radius_contraction_v1/results.json').read_text())
    plan=json.loads((OUT/'input_plan.json').read_text())
    stress=[]
    for i in range(48):
        name=f'stress_{i:02d}';old=next(r for r in v1['records'] if r['name']==name and r['variant']=='piecewise')
        direct=next(r for r in results['records'] if r['name']==name and r['variant']=='direct')
        contraction=next(r for r in results['records'] if r['name']==name and r['variant']=='contraction')
        stress.append(dict(case=i,old_forward=max(p['forward_max_abs'] for p in old['patterns']),new_forward=max(p['forward_max_abs'] for p in direct['patterns']),old_geometry=max(p['groups']['geometry']['relative'] for p in old['patterns']),direct_geometry=max(p['groups']['geometry']['relative'] for p in direct['patterns']),contraction_geometry=max(p['groups']['geometry']['relative'] for p in contraction['patterns'])))
    with (OUT/'stress_precision.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(stress[0]));writer.writeheader();writer.writerows(stress)
    fd=[]
    for r in results['records']:
        for d in r['finite_differences']:
            expected=d['expected'];err=abs(d['native_fd']-expected)
            fd.append(dict(name=r['name'],variant=r['variant'],step=d['step'],pattern=d['pattern'],direction=d['direction'],expected=expected,native_fd=d['native_fd'],cpu_fd=d['cpu_fd'],abs_error=err,normalized_error=err/max(abs(expected),1e-6)))
    with (OUT/'finite_differences.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(fd[0]));writer.writeheader();writer.writerows(fd)
    step_summaries=[]
    for rank in range(3):
        rows=[]
        for r in results['records']:
            if r['variant']!='direct' or not r['finite_differences']:continue
            e=next(e for family in ['stress_cases','holdout_cases'] for e in plan[family] if e['name']==r['name'])
            with np.load(ROOT/e['input']) as z:step=float(z['fd_steps'][rank])
            rows.extend(d for d in fd if d['name']==r['name'] and d['variant']=='direct' and d['step']==step)
        norms=np.array([d['normalized_error'] for d in rows]);ab=np.array([d['abs_error'] for d in rows])
        step_summaries.append(dict(step_rank=rank,label=['largest','middle','smallest'][rank],records=len(rows),normalized_error_quantiles=np.quantile(norms,[0,.5,.9,.99,1]).tolist(),absolute_error_quantiles=np.quantile(ab,[0,.5,.9,.99,1]).tolist()))
    result=dict(script_sha256=sha(Path(__file__)),audit_sha256=sha(OUT/'independent_audit.json'),old_results_sha256=sha(ROOT/'experiments/tgs_radius_contraction_v1/results.json'),stress_forward_old_max=max(r['old_forward'] for r in stress),stress_forward_new_max=max(r['new_forward'] for r in stress),stress_geometry_direct_max=max(r['direct_geometry'] for r in stress),stress_geometry_contraction_max=max(r['contraction_geometry'] for r in stress),fd_records=len(fd),fd_step_descriptions_direct_only=step_summaries,fd_worst_normalized=sorted(fd,key=lambda x:x['normalized_error'],reverse=True)[:10],fd_worst_absolute=sorted(fd,key=lambda x:x['abs_error'],reverse=True)[:10],max_sigma_opacity_relative=max(p['groups']['sigma_opacity']['relative'] for r in audit['rows'] for p in r['patterns']),max_SH_relative=max(p['groups']['SH']['relative'] for r in audit['rows'] for p in r['patterns'] if 'SH' in p['groups']),
        interpretation='All VJP/forward/support gates pass, but FD accuracy had no frozen numeric gate. Finite precision cancellation remains in native FD, particularly at smaller steps and small directional derivatives; do not claim 5184 FD values agree within 1e-3.',training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
