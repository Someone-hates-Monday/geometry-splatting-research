from pathlib import Path
from collections import Counter
import json
import numpy as np
import mpmath as mp
from research.verify_csr_decode import reference
from research.verify_float_product import transmittance
from research.benchmark_csr_decode import read,sha,save,now
from research.exact_perspective_backend import ROOT
R=ROOT/'research';BASE=R/'experiments/float_product_v1';OUT=R/'results/float_product_failure_analysis_v1.json'

def main():
    assert not OUT.exists();s=read(BASE/'summary.json');rows=[];inputs={};margin=mp.mpf('1e-60')
    for view in s['rows']:
        name=f'scan{view["scan"]}_{view["camera"]}';path=BASE/(name+'_probes.npz');inputs[path.relative_to(ROOT).as_posix()]=sha(path)
        with np.load(path) as d:source=d['source'];sections=d['sections'];pixels=d['pixels'];q=d['query'];output=d['output']
        cases={a['pixel']:a['case'] for a in view['anomalies']}
        for j,pixel in enumerate(pixels):
            if output[j,0]!=0:continue
            byprec={}
            for precision in [80,140]:
                mp.mp.dps=precision;nodes=[reference(v) for v in source[int(sections[j]):int(sections[j+1])]];center=mp.mpf(float(output[j,11]))*mp.sqrt(mp.mpf(float(q[j,0]))**2+mp.mpf(float(q[j,1]))**2+1);epsilon=mp.mpf(1)/1000;beta=mp.fprod([1-v['alpha'] for v in nodes]);left=transmittance(nodes,center-epsilon);right=transmittance(nodes,center+epsilon);byprec[precision]=(beta,left,right)
            beta,left,right=byprec[140];agree=all(abs(x-y)<=mp.mpf('1e-65') for x,y in zip(byprec[80],byprec[140]))
            if not agree or min(abs(beta-.5),abs(left-.5),abs(right-.5))<margin:category='inconclusive'
            elif beta>.5:category='no_finite_root'
            elif left<.5:category='root_before_error_band'
            elif right>.5:category='root_after_error_band'
            elif left>.5 and right<.5:category='inside_error_band_certificate_too_wide'
            else:category='inconclusive'
            rows.append(dict(view=name,pixel=int(pixel),old_anomaly=cases.get(int(pixel)),nodes=int(sections[j+1]-sections[j]),category=category,precision_agreement=agree,beta=mp.nstr(beta,70),Tleft=mp.nstr(left,70),Tright=mp.nstr(right,70)))
        print(f'{name}: exploratory unresolved examined={len(rows)}',flush=True)
    counts=dict(Counter(r['category'] for r in rows));assert len(rows)==132 and sum(r['old_anomaly'] is not None for r in rows)==9
    result=dict(status='complete',checked_utc=now(),selection='All132 unresolved within predeclared1545 probes; exploratory conditional subset, not representative',counts=counts,rows=rows,inputs_sha256=inputs,sources_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),R/'FLOAT_PRODUCT_FAILURE_PROTOCOL.md',R/'verify_float_product.py',R/'verify_csr_decode.py']},independent_precisions=[80,140],epsilon='exact1/1000 camera-ray units',scope='Physical mathematical reference, not GT reconstruction evidence or population error-rate estimate.')
    save(OUT,result);print(json.dumps(counts))
if __name__=='__main__':main()
