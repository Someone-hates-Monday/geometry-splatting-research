"""Replay production probe artifacts and inspect actual Torch CUDA diagnostics."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from research.clipped_packet_torch import recover

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def diagnostics(plane,width,conic,qmean,qbar,qray,focal,amplitude):
    z=plane[:,2]/torch.linalg.vector_norm(qmean,dim=1)
    rbar=torch.sum(qbar*qbar,dim=1);b=rbar/(width*width);scale=z*z*b
    c=(plane[:,:2]*focal*torch.sqrt(rbar)[:,None]/z[:,None]+qbar[:,:2])/rbar[:,None]
    kxx=conic[:,0]*focal[0]**2/scale;kxy=conic[:,1]*focal[0]*focal[1]/scale;kyy=conic[:,2]*focal[1]**2/scale
    a=qray[:,:2]-qbar[:,:2];m=qmean[:,:2]-qbar[:,:2]
    A=1+torch.sum(c*a,dim=1);B=1+torch.sum(c*m,dim=1)
    aa=kxx*a[:,0]**2+2*kxy*a[:,0]*a[:,1]+kyy*a[:,1]**2
    den=A*A+aa;v=A[:,None]*m-B[:,None]*a
    det=kxx*kyy-kxy*kxy;wedge=a[:,0]*m[:,1]-a[:,1]*m[:,0]
    gram=det*wedge*wedge+kxx*v[:,0]**2+2*kxy*v[:,0]*v[:,1]+kyy*v[:,1]**2
    return torch.stack((den,det,scale*gram/den),dim=1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    m=read(args.run/'manifest.json');s=read(args.run/'summary.json');assert m['status']=='complete'
    with zipfile.ZipFile(args.run/'sources.zip') as z:
        assert set(z.namelist())==set(m['source_sha256'])
        for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest and hashlib.sha256(z.read(name)).hexdigest()==digest
    for name,digest in m['artifacts_sha256'].items():assert sha(args.run/name)==digest
    for name,digest in m['generated_sha256'].items():assert sha(args.run/'generated'/name)==digest
    assert sha(Path(m['binary_path']))==m['binary_sha256']
    ply=[]
    for scan in (37,55,63):
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json')
        path=ROOT/f'output_compare/ggs_pipeline/scan{scan}/point_cloud/iteration_30000/point_cloud.ply'
        expected=next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
        assert sha(path)==expected;ply.append({'scan':scan,'sha256':expected})
    torch.set_num_threads(1);total=bad_cuda_diag=0;component_equal=0;counts={k:0 for k in s['arms']}
    for i in range(24):
        row=read(args.run/f'point_{i:02d}.json')
        with np.load(args.run/f'point_{i:02d}.npz',allow_pickle=False) as d:
            target=d['expected'];n=len(d['selected_ids']);assert n==row['inputs'];total+=n
            for name in s['arms']:
                candidate=d[name+'_candidate'];diff=abs(candidate-target)
                e=np.column_stack((diff[:,0]/np.maximum(1e-8,abs(target[:,0])),diff[:,1]/np.maximum(1e-8,abs(target[:,1])),diff[:,2]))
                assert np.array_equal(e,d[name+'_errors'],equal_nan=True)
                count=int(sum(d[name+'_bad']));assert count==row['arms'][name]['failed_inputs'];counts[name]+=count
            values=[d['plane'][:,:3],1/d['plane'][:,3],d['conic_opacity'][:,:3],d['qmean_uv'],d['qbar_uv'],d['qray'],d['focal'],d['conic_opacity'][:,3]]
            values=[torch.tensor(x,dtype=torch.float32,device='cuda') for x in values]
            candidate=recover(*values).cpu().numpy();diag=diagnostics(*values).cpu().numpy()
            assert np.array_equal(candidate,d['production_uv_cuda_fp32_decode_candidate']),i
            component_equal+=candidate.size
            invalid=(~np.all(np.isfinite(diag),axis=1))|(diag[:,0]<=0)|(diag[:,1]<=0)|(diag[:,2]<0)
            bad_cuda_diag+=int(sum(invalid))
            error=d['production_uv_cuda_fp32_decode_errors']
            bad=invalid|(d['radii']<=0)|(d['tiles_touched']<=0)|np.any(error>np.array([1e-5,1e-4,1e-4]),axis=1)
            assert np.array_equal(bad,d['production_uv_cuda_fp32_decode_bad'])
    assert total==s['inputs']==1992
    for name,count in counts.items():assert count==s['arms'][name]['failed_inputs']
    result={'status':'complete','checked_utc':datetime.now(timezone.utc).isoformat(),'verified_sources_and_zip':len(m['source_sha256']),
       'verified_artifacts':len(m['artifacts_sha256']),'verified_generated':len(m['generated_sha256']),
       'binary_sha256':m['binary_sha256'],'original_ply_verified':ply,'inputs':total,
       'cuda_replay_bitwise_equal_components':component_equal,'actual_cuda_invalid_diagnostics':bad_cuda_diag,
       'arm_failure_counts':counts,'development_gate_pass':s['development_gate_pass'],'source_sha256':sha(Path(__file__)),
       'limits':['Replays errors and CUDA decode; does not independently recompile or rerun production preprocessing.',
          'Original probe applied NumPy diagnostics to CUDA outputs; this replay additionally checks actual Torch CUDA den/det/energy.',
          'Copied production kernel remains a diagnostic extension, not the installed renderer binary.',
          'No synthetic guarantee, full renderer backward, runtime, GT quality or novelty conclusion.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','inputs','verified_sources_and_zip','verified_artifacts','cuda_replay_bitwise_equal_components','actual_cuda_invalid_diagnostics','development_gate_pass')},indent=2),flush=True)

if __name__=='__main__':main()
