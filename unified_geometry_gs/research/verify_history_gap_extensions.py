from pathlib import Path
from decimal import Decimal,localcontext
import json,hashlib
import numpy as np
from research.verify_history_gap_raw import verify_manifest
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    out=R/'results/history_gap_extensions_audit_v2.json';assert not out.exists();M=R/'experiments/history_gap_shared_micro_v1';E=R/'experiments/history_gap_extension_v1';provenance={p.name:verify_manifest(p) for p in [M,E]};micro=read(M/'summary.json');shared=read(R/'experiments/history_gap_shared_v1/summary.json')
    with np.load(M/'losses.npz') as d:
        for row in micro['float64_all_ray_micro']:
            h=row['h'];minus=d[f'{h}_-1'];plus=d[f'{h}_1'];assert minus.shape==plus.shape==(891,) and np.isfinite(minus).all() and np.isfinite(plus).all();fd=float((plus.mean()-minus.mean())/(2*h));assert row['derivative']==fd;pred=-shared['scaled_gradient_norms']['full'];assert pred==row['predicted'];assert row['gate_pass']==(abs(fd-pred)<=max(1e-9,1e-5*abs(pred)))
    assert [r['gate_pass'] for r in micro['float64_all_ray_micro']]==[False,True,False]
    decomposition=read(M/'old_error_decomposition.json')['rows'];assert len(decomposition)==891 and decomposition[0]['view']=='scan37_0034' and decomposition[0]['pixel']==214675
    ext=read(E/'summary.json');assert ext['status']=='complete' and len(ext['synthetic'])==16 and len(ext['seam_cases'])==16 and len(ext['real_missing_updates'])==2
    assert all(float(err)<=1e-30 for r in ext['synthetic'] for err in r['relative_errors']) and all(float(r['gradient_error'])<1e-120 for r in ext['seam_cases'])
    with localcontext() as ctx:
        ctx.prec=70
        for row in micro['posthoc_dominant_case']['mp_rows']:
            fd=Decimal(row['derivative']);pred=Decimal(row['predicted']);assert row['gate_pass']==(abs(fd-pred)<=max(Decimal('1e-9'),Decimal('1e-5')*abs(pred))) and not row['cap_changed']
        for row in ext['real_missing_updates']:
            assert Decimal(row['log2_minus_omega'])>0 and row['index']==109 and row['pixel']==57513 and row['view']=='scan37_0020'
            assert abs(row['float64_loss']-float(row['extended_loss']))==row['loss_error'] and row['loss_error']<1e-10
            for step in row['steps']:assert step['gate_pass'] and step['relative_error']<=1e-5
    result=dict(status='complete',audit_gate_pass=True,provenance=provenance,shared_original_gate_pass=False,original_gate_verdict_preserved=True,metadata_correction='V1 field shared_original_gate_preserved was mislabeled false; it meant the original gate failed. V2 corrects labels only, no numerical result changed.',shared_micro_gate_sequence=[False,True,False],extension_gate_pass=True,extension_regular_models=16,extension_seam_cases=16,extension_max_relative_fd=max(float(e) for r in ext['synthetic'] for e in r['relative_errors']),real_missing_updates=ext['real_missing_updates'],scope='Complete source/artifact and saved-array/statistics audit. Shared micro MP is one posthoc diagnostic point; extension real checks are two original missing updates, not full MP branch validation, training or GT evidence.')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
