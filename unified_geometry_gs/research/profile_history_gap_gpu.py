from pathlib import Path
import hashlib,json,importlib.util
import numpy as np
import mpmath as mp
import torch
from research.history_gap_gpu import Engine,_Gap,metadata_array
from research.history_gap_extension import mp_extended
R=Path(__file__).resolve().parent;ROOT=R.parent;GPU=R/'experiments/history_gap_gpu_v1';WORLD=R/'experiments/world_root_vjp_v1';OUT=R/'results/history_gap_gpu_profile_v1.json'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert not OUT.exists();torch.set_num_threads(1);m=read(GPU/'manifest.json');binary=ROOT/m['binary']['path'];assert sha(binary)==m['binary']['sha256'];spec=importlib.util.spec_from_file_location(binary.stem,binary);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    raw=[];source=[];sections=[0];metadata=[]
    for f in sorted(WORLD.glob('scan*_gradients.npz')):
        meta=read(f.with_name(f.name.removesuffix('_gradients.npz')+'.json'))['meta']
        with np.load(f) as d:raw.append(d['raw']);source.append(d['source']);ss=d['sections']
        for n in np.diff(ss):sections.append(sections[-1]+int(n));metadata.append(metadata_array(meta))
    engine=Engine(np.concatenate(raw),np.concatenate(source),np.array(sections),np.array(metadata),native);timings=[]
    for _ in range(4):
        theta=engine.base.detach().clone().requires_grad_(True);events=[torch.cuda.Event(enable_timing=True) for _ in range(4)];events[0].record();fields=engine.fields(theta);events[1].record();loss,info=_Gap.apply(*fields,engine.sections,engine.ids,native);events[2].record();loss.sum().backward();events[3].record();torch.cuda.synchronize();timings.append(dict(fields_ms=float(events[0].elapsed_time(events[1])),loss_and_seed_ms=float(events[1].elapsed_time(events[2])),raw_backward_ms=float(events[2].elapsed_time(events[3]))))
    cases=[('single_singular',[.75],[0.],[1.]),('common_singular',[.5,.5],[0.,0.],[1.,2.]),('missing',[.4],[0.],[1.]),('mass_boundary',[.5],[0.],[1.]),('before',[.9],[0.],[1.]),('strict_cap',[.6,.999],[-.2,.2],[5.,5.])]
    cases += [(f'thin_{r}',[.5,.5],[-1.,1.],[r,r]) for r in [5.,10.,1/.075,20.]]
    special=[];mp.mp.dps=240
    for name,amp,p,rho in cases:
        inputs=[torch.tensor(v,device='cuda',dtype=torch.float64) for v in [p,rho,np.zeros(len(p)),amp]];loss,seeds,info=native.solve(*inputs,torch.tensor([0,len(p)],device='cuda',dtype=torch.int64));actual=seeds.cpu().numpy();alpha=[min(mp.mpf(float(np.float32(.99))),mp.mpf(float(a))) for a in amp];nodes=[[a,mp.mpf(float(pp)),mp.mpf(float(rr))] for a,pp,rr in zip(alpha,p,rho)];reference_loss,g,refinfo=mp_extended(nodes);expected=[]
        for j,row in enumerate(g):
            logseed=row[0]*alpha[j];capped=mp.mpf(float(amp[j]))>mp.mpf(float(np.float32(.99)));expected.append([float(row[1]),float(row[2]),0. if capped else float(-logseed/2),0. if capped else float(logseed/mp.mpf(float(amp[j])) )])
        expected=np.array(expected);rel=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-300));loss_error=abs(float(loss.cpu()[0])-float(reference_loss));special.append(dict(name=name,status=int(info.cpu()[0,0]),relative_seed_error=rel,loss_error=loss_error,gate_pass=rel<=1e-5 and loss_error<=1e-9))
    invalid=native.solve(torch.tensor([0.],device='cuda',dtype=torch.float64),torch.tensor([0.],device='cuda',dtype=torch.float64),torch.tensor([0.],device='cuda',dtype=torch.float64),torch.tensor([.5],device='cuda',dtype=torch.float64),torch.tensor([0,1],device='cuda',dtype=torch.int64));invalid_rejected=bool(torch.isnan(invalid[0]).all()) and int(invalid[2].cpu()[0,0])<0;assert invalid_rejected
    result=dict(status='complete',binary_sha256=sha(binary),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),R/'history_gap_gpu.py',R/'history_gap_gpu.cu',R/'history_gap_extension.py']},stage_timings_ms=timings,warm_medians_ms={k:float(np.median([r[k] for r in timings[1:]])) for k in timings[0]},coefficient_special_cases=special,invalid_input_explicitly_rejected=invalid_rejected,scope='CUDA-stage events including allocation/seed backward wrapper, excludes host transfer/setup; special cases are synthetic coefficient checks, no training performance claim.');OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
