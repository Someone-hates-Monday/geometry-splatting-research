"""Seal v6 and prove raw sign errors remain undecidable under the validated interval."""
import datetime,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compensated_edge_v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(x):return F.from_float(float(x))
def main():
    manifest=OUT/'evidence_manifest.json';assert not manifest.exists()
    audit=json.loads((OUT/'independent_audit.json').read_text());end=json.loads((OUT/'endpoint_checks.json').read_text());plan=json.loads((OUT/'input_plan.json').read_text())
    assert audit['all_edge_gates'] and audit['in_domain_strict_ftz_trace_bitwise_identical']
    assert end['audit_sha256']==sha(OUT/'independent_audit.json')
    assert end['script_sha256']==sha(ROOT/'check_tgs_compensated_edge_endpoints.py')
    assert audit['audit_script_sha256']==sha(ROOT/'audit_tgs_compensated_edge.py')
    assert plan['prepare_sha256']==sha(ROOT/'prepare_tgs_compensated_edge.py')
    assert plan['source_sha256']==sha(ROOT/'tgs_compensated_edge_probe.cu')
    assert plan['protocol_sha256']==sha(ROOT/'TGS_COMPENSATED_EDGE_PROTOCOL_V6.md')
    for p,h in plan['files'].items():assert sha(OUT/p)==h
    for p,h in plan['used_sources'].items():assert sha(ROOT/p)==h
    with np.load(OUT/'inputs.npz') as z:c=z['coeff'];xy=z['xy'];domain=z['gate']
    checks={}
    for variant in ['strict','fast_ftz']:
        raw=(OUT/variant/'edge.bin').read_bytes();n=len(c);out=np.frombuffer(raw[:n*136],'<f4').reshape(n,34);bad=[];negative=0
        for i in np.flatnonzero(domain):
            P=sum(q(v)*q(w) for v,w in zip(c[i],[*xy[i],1]));r=q(out[i,1]);e=q(out[i,2])
            if r < -e:assert P<0;negative+=1
            if (r>0)-(r<0)!=(P>0)-(P<0):assert abs(r)<=e;bad.append(int(i))
        checks[variant]=dict(negative_certificates_correct=negative,wrong_raw_sign_all_uncertain_indices=bad)
    (OUT/'promotion_decision.json').write_text(json.dumps(dict(edge_gate=True,raw_sign_checks=checks,renderer_gate=False,power_interval_proof=False,performance_promotion=False,production_promotion=False,training=False),indent=2)+'\n')
    files=[ROOT/p for p in ['TGS_COMPENSATED_EDGE_PROTOCOL_V6.md','TGS_COMPENSATED_EDGE_RESULT_2026_10_20.md','tgs_compensated_edge_probe.cu','prepare_tgs_compensated_edge.py','run_tgs_compensated_edge.py','audit_tgs_compensated_edge.py','check_tgs_compensated_edge_endpoints.py','seal_tgs_compensated_edge.py']]+[p for p in OUT.rglob('*') if p.is_file()]
    entries=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    manifest.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries,used_v5_sources_checked=len(plan['used_sources']),v5_manifest_sha256=plan['v5_manifest_sha256']),indent=2)+'\n')
    print(json.dumps(dict(files=len(entries),bytes=sum(e['size'] for e in entries),manifest_sha256=sha(manifest),wrong_sign_uncertain_each=len(checks['strict']['wrong_raw_sign_all_uncertain_indices']),negative_certificate_counts={v:d['negative_certificates_correct'] for v,d in checks.items()})),flush=True)
if __name__=='__main__':main()
