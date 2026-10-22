"""Retain CPU-vs-native threshold disagreements; no hidden relabeling as exact math."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compositing_guard_v4';OLD=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'diagnostic_summary.json';assert not target.exists();audit=json.loads((OUT/'independent_audit.json').read_text());result=json.loads((OUT/'results.json').read_text());rows=[];prior=[]
    for r in result['records']:
        if r['variant']!='adaptive':continue
        if r['family']=='compositing_cases':
            name=r['name']
            with np.load(OUT/('adaptive_'+name+'.npz')) as z:native=z['native_aux']
            with np.load(OUT/('cpu_reference_'+name+'.npz')) as z:cpu=z['aux']
            rows.append(dict(name=name,CPU_full_RGB_abs=r['CPU_full_rgb_abs'],CPU_full_aux_abs=r['CPU_full_aux_abs'],CPU_selected_gate=r['CPU_gate'],CPU_median_different_pixels=int((np.abs(native[5]-cpu[5])>3e-5).sum()),native_oracle_gate=r['gate']))
        else:
            with np.load(OUT/('oracle64_'+r['name']+'.npz')) as z:new={k:z[k].copy() for k in ['native_rgb','native_aux','native_radii']}
            with np.load(OLD/('oracle64_'+r['name']+'.npz')) as z:old={k:z[k].copy() for k in ['native_rgb','native_aux','native_radii']}
            prior.append(dict(name=r['name'],rgb=float(np.abs(new['native_rgb']-old['native_rgb']).max()),aux=float(np.abs(new['native_aux']-old['native_aux']).max()),radii_equal=bool(np.array_equal(new['native_radii'],old['native_radii']))))
    report=dict(script_sha256=sha(Path(__file__)),audit_sha256=sha(OUT/'independent_audit.json'),new_threshold_cases=rows,retained_v3_oracle_comparison=prior,
        new_CPU_selected_failed=[r['name'] for r in rows if not r['CPU_selected_gate']],new_CPU_full_RGB_max=max(r['CPU_full_RGB_abs'] for r in rows),new_CPU_full_aux_max=max(r['CPU_full_aux_abs'] for r in rows),
        retained_independent_full_RGB_max=max(r['independent_full_RGB_abs'] for r in audit['rows'] if r['family']!='compositing_cases' and r['variant']=='adaptive'),
        interpretation='V4 reproduces the declared double-window/float-compositing oracle path, not exact-real threshold decisions. CPU64 disagreements at discrete thresholds are retained even when native-oracle tests pass. No GT quality or speed claim.',training=False)
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['new_threshold_cases','retained_v3_oracle_comparison']}),flush=True)
if __name__=='__main__':main()
