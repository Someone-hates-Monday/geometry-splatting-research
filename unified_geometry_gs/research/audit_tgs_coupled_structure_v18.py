"""Direct rational check of the prefix factorization and each local rounding envelope."""
import json
from fractions import Fraction as F
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new,exact
from tgs_interleaving_certificate_v17 import q,compose,layers
from tgs_coupled_certificate_v18 import accepted_data
OUT=ROOT/'experiments/tgs_coupled_v18'

def main():
    sources=[ROOT/'audit_tgs_coupled_structure_v18.py',ROOT/'tgs_coupled_certificate_v18.py',
        ROOT/'tgs_interleaving_certificate_v17.py',OUT/'development_audit.json',OUT/'fresh/independent_audit.json']
    write_new(OUT/'structure_plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in sources},training=False))
    rows=[];checks=zeros=0;max_relative=F(0)
    for label,folder in [('development',ROOT/'experiments/tgs_interleaving_v17/validation'),('fresh',OUT/'fresh')]:
        r=json.loads((folder/'results.json').read_text())
        for row in r['rows']:
            ap=folder/row['before'];bp=folder/row['after'];ar=np.load(ap);br=np.load(bp)
            a,sa=accepted_data(ap);b,sb=accepted_data(bp);k=0
            while k<min(len(a),len(b)) and a[k][1:]==b[k][1:]:k+=1
            if k:C0,T0=sa[k-1];assert (C0,T0)==sb[k-1]
            else:C0=[F(0)]*3;T0=F(1)
            t0=F(1)
            for _,al,_ in a[:k]:t0*=1-al
            afull=layers(ar['packed'],ar['full']);bfull=layers(br['packed'],br['full'])
            A=compose(afull);B=compose(bfull);asum=compose(a);bsum=compose(b);S0=compose(a[k:]);S1=compose(b[k:])
            for ch in range(3):
                taila=A[ch]-asum[ch];tailb=B[ch]-bsum[ch]
                assert B[ch]-A[ch]-tailb+taila==t0*(S1[ch]-S0[ch])
            u=F(1,2**24)
            for raw,ls,S in [(ar,a,S0),(br,b,S1)]:
                l=len(ls)-k;g=3*l*u/(1-3*l*u)
                for ch in range(3):
                    ideal=C0[ch]+T0*S[ch];limit=g*ideal;residual=abs(q(raw['summary'][ch])-ideal)
                    assert residual<=limit
                    if limit:max_relative=max(max_relative,residual/limit)
                    checks+=1
            identical=k==len(a)==len(b)
            if identical:assert exact(ar['summary'][:3],br['summary'][:3]);zeros+=1
            rows.append(dict(cohort=label,scene=row['scene'],event=row['event'],pixel=row['pixel'],common_prefix=k,
                exact_prefix_factorization=True,local_rounding_contained=True,identical_sequence=identical))
    write_new(OUT/'structure_audit.json',dict(all_passed=True,rows=rows,local_channel_checks=checks,literal_zero_pairs=zeros,
        max_local_rounding_over_envelope=float(max_relative),training=False,compiler_general_proof=False))
    print(json.dumps(dict(pairs=len(rows),local_channel_checks=checks,literal_zero_pairs=zeros,max_local_ratio=float(max_relative))),flush=True)

if __name__=='__main__':main()
