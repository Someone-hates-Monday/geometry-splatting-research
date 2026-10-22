"""All nine captured rays, three fixed CUDA precision paths, full traces retained."""
from pathlib import Path
import argparse,hashlib,json,os,traceback,zipfile
from datetime import datetime,timezone
import numpy as np
import torch
from research.exact_outlier_solver_backend import load
ROOT=Path(__file__).resolve().parents[1]
NAMES=('fp32_factors_fp32_product','fp32_factors_fp64_product','fp64_logT')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def execute(out,state,capture):
    mod=load(out);state.update(status='running',binary_sha256=sha(Path(mod.__file__)),binary_path=mod.__file__);save(out/'manifest.json',state);rows=[]
    for i in range(9):
        row=read(capture/f'ray_{i:02d}.json');results={};arrays={}
        with np.load(capture/f'ray_{i:02d}.npz',allow_pickle=False) as d:
            old=d['copied_initialization'];init=np.asarray([old[1],old[2],old[3],old[5]],dtype=np.float32)
            for arm,original in (('packet','exact_packet'),('factor','exact_factor')):
                coeff=np.column_stack([d[arm+'_'+k] for k in ('peak','rho','alpha')]).astype(np.float32)
                trace=mod.forward(torch.tensor(coeff,device='cuda'),torch.tensor(init,device='cuda')).cpu().numpy();arrays[arm+'_trace']=trace
                z=d[original+'_output_z_depth'][0];slope=d[original+'_output_slope'][0]
                results[arm]={'original_z_depth':float(z),'original_slope':float(slope),
                    'fp32_replay_z_bitwise_equal':bool(np.float32(trace[0,1]).view(np.uint32)==z.view(np.uint32)),
                    'fp32_replay_slope_bitwise_equal':bool(np.float32(trace[0,2]).view(np.uint32)==slope.view(np.uint32)),
                    'modes':[{ 'name':name,'ray_depth':float(t[0]),'z_depth':float(t[1]),'slope':float(t[2]),'valid':bool(t[3]),
                        'final_bracket':[float(t[4]),float(t[5])],'final_T':[float(t[6]),float(t[7])],
                        'final_endpoint_T_exactly_equal':bool(t[6]==t[7]),'final_endpoint_T_exactly_half':[bool(t[6]==.5),bool(t[7]==.5)]}
                        for name,t in zip(NAMES,trace)]}
        np.savez_compressed(out/f'ray_{i:02d}.npz',**arrays)
        result={'case':i,'capture_identity':{k:row[k] for k in ('scan','camera','pixel')},'results':results}
        save(out/f'ray_{i:02d}.json',result);rows.append(result);save(out/'progress.json',{'utc':now(),'completed':len(rows)})
    summary={'status':'complete','rays':9,'arms':18,'fp32_z_bitwise_matches':sum(r['results'][a]['fp32_replay_z_bitwise_equal'] for r in rows for a in ('packet','factor')),
       'fp32_slope_bitwise_matches':sum(r['results'][a]['fp32_replay_slope_bitwise_equal'] for r in rows for a in ('packet','factor')),
       'rows':rows,'limits':['Independent standalone kernel with captured probe coefficients/initialization, not original instruction instrumentation.',
          'Modes1/2 change T and T-based interpolation precision; bracket remains FP32 and fixed5 subdivisions.',
          'No true-root existence assertion from finite outputs; compare separate exact mass/high-precision oracle.',
          'No GT, training, timing benefit or new universal method claim.']}
    save(out/'summary.json',summary);state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',state);print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','limits')},indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if out.exists():raise ValueError('Fresh output required')
    m=read(a.capture/'manifest.json');assert m['status']=='complete' and m['captured_rays']==9
    for name,digest in m['artifacts_sha256'].items():assert sha(a.capture/name)==digest
    sources=[Path(__file__),ROOT/'research/exact_outlier_solver_backend.py',ROOT/'research/EXACT_OUTLIER_SOLVER_REPLAY_PROTOCOL.md',a.capture/'manifest.json']
    sources+=[a.capture/f'ray_{i:02d}.{ext}' for i in range(9) for ext in ('json','npz')]
    out.mkdir(parents=True);state={'created_utc':now(),'status':'building','pid':os.getpid(),'torch':torch.__version__,
         'source_sha256':{p.resolve().relative_to(ROOT).as_posix():sha(p) for p in sources}}
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.resolve().relative_to(ROOT).as_posix())
    try:execute(out,state,a.capture)
    except BaseException as e:
        state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(out/'manifest.json',state)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
