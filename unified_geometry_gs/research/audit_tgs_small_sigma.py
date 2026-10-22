"""Independent exact rational decomposition of the cached boundary equation."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_stable_window_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path=OUT/'small_sigma_independent_audit.json';assert not path.exists()
    cert=json.loads((OUT/'small_sigma_boundary_certificate.json').read_text())
    with np.load(OUT/'small_sigma_native_cache.npz') as z:
        points=[[F(float(x)) for x in v] for v in z['local_vertices']]
        normals=[[F(float(x)) for x in v] for v in z['normals']]
        offsets=[F(float(x)) for x in z['offsets']];anchor=[F(float(x)) for x in z['anchor']]
        image=z['native_rgb'];alpha=z['native_aux'][1]
    rows=[]
    for record in cert['points']:
        x,y=record['pixel'];px,py=F(x)-anchor[0],F(y)-anchor[1]
        cached=[n[0]*px+n[1]*py+b for n,b in zip(normals,offsets)]
        crosses=[]
        for i in range(3):
            a=points[i];b=points[(i+1)%3]
            crosses.append((b[0]-a[0])*(py-a[1])-(b[1]-a[1])*(px-a[0]))
        # Independent closed-form projection of this dyadic 3-4-5 input.
        eps=F(1,10**7);t=F(2)/(2+eps)
        original=[(F(32),F(32)),(F(192),F(32)),(F(32),F(152))]
        q=[(t*a+(1-t)*F(255,2),t*b+(1-t)*F(255,2)) for a,b in original]
        dx=q[2][0]-q[1][0];dy=q[2][1]-q[1][1]
        cross=dx*(F(y)-q[1][1])-dy*(F(x)-q[1][0])
        length=F(200)*t;phi=-cross/length
        radius=F(40)*t;u=-phi/radius
        assert cross==F(record['exact_signed_cross'][1]) and u==F(377,8000000000)
        assert abs(float(phi)-float(record['active_phi_80_digits']))<1e-20
        assert float(alpha[y,x])==record['native_alpha']
        rows.append(dict(pixel=[x,y],original_exact_phi=str(phi),original_exact_u=str(u),
            exact_local_vertex_crosses=[str(v) for v in crosses],inside_cached_vertex_triangle=all(v>0 for v in crosses),
            exact_stored_normal_phi=[str(v) for v in cached],exact_stored_normal_max_phi=float(max(cached)),native_alpha=float(alpha[y,x]),
            result='Original and quantized-vertex triangles both contain the pixel. Rounded cached normal/offset equation and pixel arithmetic must also be considered; vertex quantization alone is not a causal explanation.'))
    decision=dict(protocol_gates_passed=True,expanded_full_image_gate_passed=False,performance_measurement_allowed=False,
        training_allowed_by_this_decision=False,reason='Whole-image expansion exposes a small-sigma edge-conditioning counterexample. Selected forward/VJP tests remain passed, but promotion is deferred.',
        original_audit_sha256=sha(OUT/'independent_audit.json'),full_image_diagnosis_sha256=sha(OUT/'full_old_image_diagnosis.json'),small_sigma_certificate_sha256=sha(OUT/'small_sigma_boundary_certificate.json'),
        next='Derive branch/conditioning certificates and design selective higher-precision evaluation from original geometry; preserve every old and newly discovered failure. Do not clamp sigma or silently discard boundary pixels.')
    result=dict(script_sha256=sha(Path(__file__)),certificate_sha256=sha(OUT/'small_sigma_boundary_certificate.json'),cache_sha256=sha(OUT/'small_sigma_native_cache.npz'),rows=rows,all_exact_original_predicates_verified=True,promotion=decision)
    path.write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'promotion_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
