"""Replay all captured inputs: original vs Gram-determinant forward only."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import sympy as sp
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.gram_forward_probe import load_probe,payloads
from research.world_preprocess_probe import load_probe as legacy_probe,RASTER
from research.audit_world_preprocess import oracle
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'research/gram_forward_probe.py',ROOT/'research/world_preprocess_probe.py',ROOT/'research/audit_world_preprocess.py',ROOT/'research/precise_preprocess_backend.py',RASTER/'cuda_rasterizer/render_forward.cu',RASTER/'cuda_rasterizer/render_backward.cu']
    sources+=list((RASTER/'cuda_rasterizer').glob('*.h'));sources+=[x for x in (RASTER/'third_party/glm/glm').rglob('*') if x.is_file()]
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'building','forward_scaled_gate':1e-4,'absolute_floor':1e-5,
        'source_sha256':{x.relative_to(ROOT).as_posix():sha(x) for x in sources},
        'selection':'Every 44 stress and 120 regular/synthetic captured input; no exclusions or new sampling.',
        'intervention':'Float32 Gram cross-product determinant and algebraically identical kernel dilation determinant; no training/backend change.',
        'limits':['Concurrent tiny CUDA diagnostic is not a timing experiment.','Null-scale covariance fallback is not tested.','No backward or quality claim.']}
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for x in sources:z.write(x,x.relative_to(ROOT).as_posix())
        for n,v in payloads().items():z.writestr('generated/'+n,v)
    state['source_archive_sha256']=sha(out/'sources.zip')
    def write():(out/'manifest.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    write();start=time.perf_counter()
    try:
        # Exact identity, not relying on a numerical determinant oracle.
        values=sp.symbols('a0:3 b0:3');a=sp.Matrix(values[:3]);b=sp.Matrix(values[3:]);k=sp.symbols('k',nonnegative=True)
        gram=sp.Matrix([[a.dot(a),a.dot(b)],[a.dot(b),b.dot(b)]])+k*sp.eye(2)
        identity=sp.expand(gram.det()-(a.cross(b).dot(a.cross(b))+k*(a.dot(a)+b.dot(b))+k*k))==0
        if not identity:raise ValueError('Gram identity failed.')
        module=load_probe();old=legacy_probe();state['binary_sha256']=sha(Path(module.__file__));state['legacy_binary_sha256']=sha(Path(old.__file__));state['status']='running';write()
        components={'conic':slice(0,3),'opacity_coef':slice(3,4),'plane_slope':slice(4,6),'center_distance':slice(6,7),'ray_precision':slice(7,8),'normal':slice(8,11)}
        rows=[];casecount=0
        for name in ('world_preprocess_stress_v1','factor_regular_inputs_v1'):
            d=ROOT/'research/experiments'/name;s=json.loads((d/'manifest.json').read_text());data=json.loads((d/'results.json').read_text())
            if s['status']!='complete' or sha(d/'results.json')!=s['results_sha256']:raise ValueError('Input changed.')
            state.setdefault('input_sha256',{})[str(d/'results.json')]=sha(d/'results.json')
            for i,c in enumerate(data['cases']):
                mg=torch.tensor([c['mean']],dtype=torch.float32,device='cuda');sg=torch.tensor([c['filtered_scales']],dtype=torch.float32,device='cuda')
                qg=torch.tensor([c['normalized_device_quaternion']],dtype=torch.float32,device='cuda');og=torch.tensor([c['opacity']],dtype=torch.float32,device='cuda')
                view=torch.tensor(c['view'],dtype=torch.float64);vg=view.float().cuda().contiguous();params=c['params']
                ref=oracle(mg[0].cpu().double(),sg[0].cpu().double().log(),torch.tensor(c['raw_quaternion'],dtype=torch.float64),og[0].cpu().double(),view,params)[0]
                results={v:mod.forward(mg,sg,qg,og,vg,*params)[0].cpu().double() for v,mod in (('legacy32',old),('gram32',module))}
                comparisons={}
                for v,actual in results.items():
                    comparisons[v]={}
                    for label,sl in components.items():
                        er=float(torch.linalg.vector_norm(actual[sl]-ref[sl]));norm=float(torch.linalg.vector_norm(ref[sl]));finite=bool(torch.isfinite(actual[sl]).all())
                        comparisons[v][label]={'reference_norm':norm,'absolute_l2':er if finite else None,'scaled_error':er/max(norm,1e-5) if finite else None,'finite':finite}
                rows.append({'input':name,'case':i,'model':c['model'],'id':c['id'],'axis_ratio':c['axis_ratio'],'comparison':comparisons,
                    'unchanged_ray_normal_outputs':torch.equal(results['legacy32'][4:],results['gram32'][4:])})
                casecount+=1
            print(f'{name}: all {len(data["cases"])} cases complete',flush=True)
        def summarize(v,label):
            vals=[r['comparison'][v][label] for r in rows]
            return {'max_scaled_error':max(x['scaled_error'] for x in vals if x['finite']),
                'failed_cases':sum(not x['finite'] or x['scaled_error']>1e-4 for x in vals),'total_cases':len(vals)}
        summary={v:{label:summarize(v,label) for label in components} for v in ('legacy32','gram32')}
        result={'exact_gram_identity_pass':identity,'case_count':casecount,'rows':rows,'summary':summary,
            'gram32_forward_gate_pass':all(v['failed_cases']==0 for v in summary['gram32'].values()),
            'unchanged_ray_normal_outputs_all':all(r['unchanged_ray_normal_outputs'] for r in rows),'limits':state['limits']}
        (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');state['results_sha256']=sha(out/'results.json');state['status']='complete'
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    except BaseException as e:state['status']='failed';state['error']=repr(e);raise
    finally:state['wall_seconds']=time.perf_counter()-start;write()
if __name__=='__main__':main()
