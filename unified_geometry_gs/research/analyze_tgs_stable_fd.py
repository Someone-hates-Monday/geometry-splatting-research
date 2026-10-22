"""Exact signed split of native FD discrepancies, retaining every observation."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_stable_window_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=OUT/'finite_difference_decomposition.json';assert not target.exists()
    raw=json.loads((OUT/'results.json').read_text());rows=[]
    for r in raw['records']:
        for d in r['finite_differences']:
            e=d['expected'];native=d['native_fd'];cpu=d['cpu_fd']
            numerical=native-cpu;probe=cpu-e;total=native-e
            assert abs(total-(numerical+probe))<1e-10*max(abs(total),1)
            rows.append(dict(name=r['name'],variant=r['variant'],step=d['step'],pattern=d['pattern'],direction=d['direction'],expected=e,native_fd=native,cpu_fd=cpu,
                signed_native_minus_cpu=numerical,signed_cpu_minus_base_directional=probe,signed_total=total,
                native_cpu_normalized=abs(numerical)/max(abs(e),1e-6),cpu_directional_normalized=abs(probe)/max(abs(e),1e-6),total_normalized=abs(total)/max(abs(e),1e-6)))
    direct=[r for r in rows if r['variant']=='direct']
    summary={key:dict(quantiles_0_50_90_99_100=np.quantile([r[key] for r in direct],[0,.5,.9,.99,1]).tolist()) for key in ['native_cpu_normalized','cpu_directional_normalized','total_normalized']}
    result=dict(source_sha256=sha(OUT/'results.json'),script_sha256=sha(Path(__file__)),rows=rows,records=len(rows),direct_summary=summary,
        max_normalized=sorted(direct,key=lambda r:r['total_normalized'],reverse=True)[:5],max_absolute=sorted(direct,key=lambda r:abs(r['signed_total']),reverse=True)[:5],
        caveat='The CPU-minus-base term includes finite-step Taylor remainder, representable perturbation midpoint displacement, and CPU64 arithmetic. Native-minus-CPU isolates forward implementation discrepancy on those same endpoints, divided by 2h. No universal attribution to one machine instruction or guarantee of FD accuracy.',training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
