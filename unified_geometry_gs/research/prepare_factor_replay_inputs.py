"""Adapt captured regular/synthetic audit inputs without selecting on new errors."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);cases=[];rows=[];refs={}
    for kind,name in (('real','world_preprocess_real_v1'),('synthetic','world_preprocess_synthetic_v2')):
        directory=ROOT/'research/experiments'/name;state=json.loads((directory/'manifest.json').read_text())
        data=json.loads((directory/'results.json').read_text())
        if state['status']!='complete' or sha(directory/'results.json')!=state['results_sha256']:raise ValueError('Input evidence changed.')
        refs[name]=sha(directory/'results.json');offset=len(cases)
        if kind=='real':
            for x in data['cases']:
                q=torch.tensor(x['normalized_device_quaternion'],dtype=torch.float64)
                cases.append({**x,'filtered_scales':x['scales'],'axis_ratio':x['ratio'],
                    'device_quaternion_norm64_error':float(abs(torch.linalg.vector_norm(q)-1))})
        else:
            ids=list(dict.fromkeys(r['case'] for r in data['rows']))
            for i in ids:
                x=next(r['inputs'] for r in data['rows'] if r['case']==i)
                raw=torch.tensor(x['raw_quaternion'],dtype=torch.float32,device='cuda')
                # Match the captured auditor's one-vector normalization exactly.
                q=(raw/torch.linalg.vector_norm(raw)).cpu()
                scales=x['scales']
                cases.append({'model':'synthetic','id':i,'mean':x['mean'],'filtered_scales':scales,
                    'raw_quaternion':x['raw_quaternion'],'normalized_device_quaternion':q.tolist(),
                    'opacity':float(torch.tensor(.65,dtype=torch.float32)),'params':data['params'],
                    'view':data['view_transposed_storage'],'axis_ratio':max(scales)/min(scales),
                    'device_quaternion_norm64_error':float(abs(torch.linalg.vector_norm(q.double())-1))})
        for r in data['rows']:
            rows.append({'case':r['case']+offset,'component':r['component'],'upstream':r['upstream'] if kind=='real' else r['inputs']['upstream']})
    if len(cases)!=120 or len(rows)!=696:raise ValueError('Expected all 72 real and 48 synthetic cases/upstreams.')
    components=list(dict.fromkeys(r['component'] for r in rows))
    result={'artifact_kind':'replay_inputs_only','cases':cases,'rows':rows,
        'summary':{'all64':{k:{'input_group_count':4*sum(r['component']==k for r in rows),'not_a_vjp_result':True} for k in components}},
        'limits':['Adapter output contains inputs, not computed gradients or a passing audit.',
                  'Real device quaternions are captured values. Synthetic device values reconstruct the exact original one-vector normalization recipe.']}
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'manifest.json').write_text(json.dumps({'status':'complete','artifact_kind':'replay_inputs_only','source_sha256':sha(Path(__file__)),
        'input_results_sha256':refs,'results_sha256':sha(out/'results.json'),'case_count':120,'upstream_blocks':696},indent=2)+'\n')
    print('Prepared all 120 captured cases and 696 upstream blocks; no resampling or exclusions.',flush=True)


if __name__=='__main__':main()
