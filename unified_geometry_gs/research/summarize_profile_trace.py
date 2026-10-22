"""Summarize actual CUDA trace activities, excluding synthetic annotation ranges."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh output required.')
    manifest=args.profile/'manifest.json'
    meta=json.loads(manifest.read_text())
    if meta['status']!='complete' or not meta['resume_state_verified']:
        raise ValueError('Expected completed profiling with verified initial state.')
    trace=args.profile/'trace.json'
    data=json.loads(trace.read_text())
    categories=Counter(e.get('cat','') for e in data['traceEvents'])
    allowed={'kernel','gpu_memcpy','gpu_memset'}
    activities=[e for e in data['traceEvents'] if e.get('cat') in allowed and e.get('ph')=='X' and 'dur' in e]
    if not activities:raise ValueError('No explicit CUDA kernel/memory trace activities.')
    totals={};kernels={}
    for e in activities:
        name=e['name'];dur=float(e['dur'])
        if dur<0:raise ValueError('Negative activity duration.')
        if e['cat']!='kernel':category='CUDA memory operations'
        elif 'forward_mode_differentiation' in name:category='NCC patch Jacobian kernel'
        elif 'sampleDepthCUDA' in name:category='Geometry target-depth sampling'
        elif 'renderCUDA' in name:category='Gaussian render forward/backward'
        elif 'preprocessCUDA' in name:category='Gaussian preprocess forward/backward'
        elif 'ssim' in name.lower():category='SSIM image loss'
        elif 'adam' in name.lower():category='Adam kernels'
        else:category='Other CUDA kernels'
        totals[category]=totals.get(category,0.)+dur
        k=kernels.setdefault(name,{'calls':0,'duration_us':0.,'trace_category':e['cat']})
        k['calls']+=1;k['duration_us']+=dur
    total=sum(totals.values())
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'active_training_steps':meta['active_steps'],
        'included_trace_categories':sorted(allowed),'all_trace_category_counts':dict(categories),
        'actual_cuda_activity_count':len(activities),'summed_actual_device_activity_us':total,
        'groups':[{'name':k,'duration_us':v,'fraction_of_summed_actual_device_activity':v/total} for k,v in sorted(totals.items(),key=lambda kv:-kv[1])],
        'top_kernels':[{'name':k,**v} for k,v in sorted(kernels.items(),key=lambda kv:-kv[1]['duration_us'])[:30]],
        'correction':'profile_summary.json v1 included synthetic CUDA user-annotation ranges such as ProfilerStep*/research.*, double-counting work. Use this explicit trace-category summary for device fractions.',
        'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest,trace,Path(__file__))},
        'limitations':['10 early regularized iterations, not all training stages or an end-to-end benchmark.',
            'Kernel duration sums can overlap across streams and are not the critical path or wall clock.',
            'Profiling instrumentation adds overhead; no acceleration is inferred without a controlled comparison.',
            'Name-based groups are descriptive; miscellaneous elementwise/reduction operations can belong to multiple algorithm stages.']}
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'activities':len(activities),'total_us':total,'groups':report['groups']},indent=2))


if __name__=='__main__':main()
