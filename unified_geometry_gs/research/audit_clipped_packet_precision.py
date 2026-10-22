"""FP32 packet rounding and FP64 raw-parameter independent-graph AD screening."""
from __future__ import annotations
import argparse,hashlib,json,os,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from research.clipped_packet import recover
from research.clipped_packet_torch import packet_from_raw,direct_world

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def tensor(x,grad=False):return torch.tensor(x,dtype=torch.float64,requires_grad=grad)

def fp32(packet,target):
    values=[np.asarray(packet[k],dtype=np.float32) for k in ('plane','width','conic','qmean','qbar','qray','focal','amplitude')]
    p,w,a,diag=recover(*values);assert p.dtype==w.dtype==a.dtype==np.float32
    candidate=np.column_stack((p,w,a));difference=abs(candidate-target)
    errors=np.column_stack((difference[:,0]/np.maximum(1e-8,abs(target[:,0])),difference[:,1]/np.maximum(1e-8,abs(target[:,1])),difference[:,2]))
    invalid=(~np.all(np.isfinite(candidate),axis=1))|(diag['energy']<0)|(diag['detK']<=0)|(diag['denominator']<=0)
    bad=invalid|np.any(errors>np.array([1e-5,1e-4,1e-4]),axis=1)
    return candidate,errors,bad

def raw_array(data,ids):
    row=data[ids]
    return np.column_stack([row[k] for k in ('x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','opacity','filter_3D')]).astype(float)

def execute(out,manifest,previous):
    start=time.perf_counter();torch.set_num_threads(1);rows=[]
    for scan in (37,55,63):
        path=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=path/'point_cloud/iteration_30000/point_cloud.ply'
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json')
        assert sha(ply)==next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
        data=PlyData.read(ply)['vertex'].data;cameras={c['img_name']:c for c in read(path/'cameras.json')}
        for _ in range(8):
            i=len(rows);prior=read(previous/f'packet_{i:02d}.json');point=read(ROOT/f'research/experiments/directional_field_world_v1/point_{i:02d}.json')
            assert prior['scan']==scan;camera=cameras[prior['camera']]
            with np.load(previous/f'packet_{i:02d}.npz',allow_pickle=False) as packet:
                ids=packet['selected_ids'];target=packet['expected'];candidate32,errors32,bad32=fp32(packet,target)
                raw=raw_array(data,ids);a=tensor(raw,True);b=tensor(raw,True)
                width,height=round(camera['width']/2),round(camera['height']/2)
                focal=np.array([camera['fx']*width/camera['width'],camera['fy']*height/camera['height']]);center=np.array([(width-1)/2,(height-1)/2])
                limit=1.3*np.array([width,height])/(2*focal)
                qray=tensor(np.broadcast_to(np.r_[(np.asarray(prior['pixel'])-center)/focal,1.],(len(ids),3)).copy())
                outputA,qmean=packet_from_raw(a,tensor(camera['position']),tensor(camera['rotation']),tensor(focal),tensor(center),tensor(limit),qray)
                outputB=direct_world(b,tensor(camera['position']),tensor(point['query_world_direction']))
                ga=[];gb=[]
                for component in range(3):
                    ga.append(torch.autograd.grad(outputA[:,component].sum(),a,retain_graph=component<2)[0].detach().numpy())
                    gb.append(torch.autograd.grad(outputB[:,component].sum(),b,retain_graph=component<2)[0].detach().numpy())
                ga=np.stack(ga,axis=1);gb=np.stack(gb,axis=1)
                graderrors=abs(ga-gb)/np.maximum(1e-8,abs(gb));gradbad=(~np.isfinite(graderrors))|(graderrors>1e-4)
                outputAn=outputA.detach().numpy();outputBn=outputB.detach().numpy()
                forwarderrors=abs(outputAn-outputBn)/np.maximum(1e-12,abs(outputBn));forwardbad=(~np.isfinite(forwarderrors))|(forwarderrors>1e-8)
                margin=np.min(abs(abs(qmean.detach().numpy()[:,:2])-limit),axis=1);boundary=margin<=1e-10
                np.savez_compressed(out/f'point_{i:02d}.npz',selected_ids=ids,clipped=packet['clipped'],raw=raw,
                    fp32_candidate=candidate32,fp32_errors=errors32,fp32_bad=bad32,
                    raw_packet_forward=outputAn,raw_direct_forward=outputBn,raw_forward_errors=forwarderrors,
                    raw_packet_gradients=ga,raw_direct_gradients=gb,raw_gradient_errors=graderrors,
                    raw_gradient_bad=gradbad,clip_boundary=boundary,clip_margin=margin)
                result={'scan':scan,'camera':prior['camera'],'pixel':prior['pixel'],'inputs':len(ids),
                    'fp32_bad_inputs':int(sum(bad32)),'fp32_bad_ids':ids[bad32].tolist(),'fp32_max_errors':np.max(errors32,axis=0).tolist(),
                    'raw_fp64_forward_bad_components':int(sum(forwardbad.ravel())),
                    'raw_fp64_forward_max_errors':np.max(forwarderrors,axis=0).tolist(),
                    'raw_fp64_gradient_bad_components':int(sum(gradbad.ravel())),
                    'raw_fp64_gradient_bad_inputs':int(sum(np.any(gradbad,axis=(1,2)))),
                    'raw_fp64_gradient_max_errors':np.max(graderrors,axis=(0,2)).tolist(),
                    'clip_boundary_inputs':int(sum(boundary)),'min_clip_margin':float(min(margin))}
            rows.append(result);save(out/f'point_{i:02d}.json',result);save(out/'progress.json',{'utc':now(),'completed':len(rows)})
            print(f'{len(rows)}/24: fp32 bad={result["fp32_bad_inputs"]}, raw gradient bad components={result["raw_fp64_gradient_bad_components"]}',flush=True)
        del data
    synthetic=[]
    for row in read(ROOT/'research/results/clipped_packet_theory_v1.json')['synthetic_cases']:
        packet={k:np.array([row[k]]) for k in ('plane','width','conic','qbar','qray')}
        packet['qmean']=np.array([row['qmean']]);packet['focal']=np.array([800.,850.]);packet['amplitude']=np.array([.75])
        candidate,errors,bad=fp32(packet,np.array([row['expected']]))
        synthetic.append({'case':row['case'],'clipped':row['clipped'],'candidate':candidate[0].tolist(),'errors':errors[0].tolist(),'pass':not bool(bad[0])})
    save(out/'synthetic_fp32.json',synthetic)
    summary={'inputs':sum(r['inputs'] for r in rows),'raw_gradient_components':sum(r['inputs'] for r in rows)*36,
      'fp32_packet_bad_inputs':sum(r['fp32_bad_inputs'] for r in rows),
      'fp32_packet_max_errors':np.max([r['fp32_max_errors'] for r in rows],axis=0).tolist(),
      'synthetic_fp32_bad_cases':sum(not r['pass'] for r in synthetic),'synthetic_fp32_max_errors':np.max([r['errors'] for r in synthetic],axis=0).tolist(),
      'raw_fp64_forward_bad_components':sum(r['raw_fp64_forward_bad_components'] for r in rows),
      'raw_fp64_gradient_bad_components':sum(r['raw_fp64_gradient_bad_components'] for r in rows),
      'raw_fp64_gradient_bad_inputs':sum(r['raw_fp64_gradient_bad_inputs'] for r in rows),
      'raw_fp64_gradient_max_errors':np.max([r['raw_fp64_gradient_max_errors'] for r in rows],axis=0).tolist(),
      'clip_boundary_inputs':sum(r['clip_boundary_inputs'] for r in rows),
      'limits':['FP32 inputs are rounded double packets, not actual float32/CUDA preprocessing.',
        'FP64 graphs share standard raw world-parameter transforms and the same torch AD library.',
        'Independent graph comparison is not an actual CUDA backward or full renderer/NCC check.',
        'No timing, quality, novelty or universal conditioning claim.']}
    summary['development_gate_pass']=not any(summary[k] for k in ('fp32_packet_bad_inputs','synthetic_fp32_bad_cases','raw_fp64_forward_bad_components','raw_fp64_gradient_bad_components','clip_boundary_inputs'))
    save(out/'summary.json',summary)
    manifest.update(status='complete',completed_utc=now(),wall_seconds=time.perf_counter()-start,
        development_gate_pass=summary['development_gate_pass'],
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',manifest);print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise ValueError('Fresh run required')
    previous=ROOT/'research/experiments/clipped_packet_v1';prior=read(previous/'manifest.json');assert prior['status']=='complete' and prior['numerical_gate_pass']
    for name,digest in prior['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in prior['artifacts_sha256'].items():assert sha(previous/name)==digest
    sources=set(ROOT/name for name in prior['source_sha256'])
    sources.update([Path(__file__),ROOT/'research/clipped_packet_torch.py',ROOT/'research/CLIPPED_PACKET_PRECISION_PROTOCOL.md',previous/'manifest.json',previous/'summary.json'])
    for i in range(24):sources.update([previous/f'packet_{i:02d}.json',previous/f'packet_{i:02d}.npz'])
    out.mkdir(parents=True);manifest={'created_utc':now(),'status':'running','pid':os.getpid(),'torch':torch.__version__,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}}
    save(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(sources):archive.writestr(str(source.relative_to(ROOT)),source.read_bytes())
    try:execute(out,manifest,previous)
    except BaseException as exc:
        manifest.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc))
        save(out/'manifest.json',manifest);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
