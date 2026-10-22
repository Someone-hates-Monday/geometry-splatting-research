"""Independent complete147 image/root/TSDF-input and seeded-GT accounting."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import numpy as np
from research.benchmark_csr_decode import read,sha,save,now
from research.exact_perspective_backend import ROOT
R=ROOT/'research'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();base=a.base.resolve();out=a.out.resolve();assert not out.exists();m=read(base/'manifest.json');s=read(base/'summary.json');assert m['status']==s['status']=='complete' and len(s['scenes'])==3
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    with zipfile.ZipFile(base/'sources.zip') as z:
        for n,h in m['source_sha256'].items():
            if not n.startswith('dtu_eval/Offical_DTU_Dataset/'):assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    for b in m['backends'].values():assert sha(ROOT/b['binary_relative_path'])==b['binary_sha256']
    assert sha(Path(m['original_renderer_binary']['path']))==m['original_renderer_binary']['sha256']
    totals={k:0 for k in ['views','pixels','records','selected','corrected','no_root','unresolved','masked_control_valid','masked_certified_valid']};resultrows=[];artifacts={}
    for scene in s['scenes']:
        scan=scene['scan'];folder=base/f'scan{scan}';r=read(folder/'render_summary.json');align=read(folder/'alignment.json');assert r['status']=='complete' and len(r['rows'])==49 and align['camera_names']==[f'{i:04d}' for i in range(49)] and align['official_indices']==list(range(49));masked=[0,0]
        for j,row in enumerate(r['rows']):
            assert row['camera']==f'{j:04d}' and all(row['original_renderer_bits'].values()) and row['CSR_write_checker_zero'] and row['source_cache_all_valid'] and row['unchanged_pixels_preserved'];path=folder/(row['camera']+'_images.npz');artifacts[path.relative_to(base).as_posix()]=sha(path)
            with np.load(path) as d:rgb=d['rgb'];control=d['control_depth'];candidate=d['certified_depth'];mask=d['mask'][0];certificate=d['certificate'];selected=d['selected_pixels'];root=d['root_output'];K=d['intrinsic'];X=d['extrinsic']
            H,W=row['height'],row['width'];assert rgb.shape==(H,W,3) and control.shape==candidate.shape==mask.shape==(H,W) and certificate.shape==(H*W,12) and K.shape==(3,3) and X.shape==(4,4);assert np.isfinite(rgb).all() and np.isfinite(control).all() and np.isfinite(candidate).all();assert np.array_equal(selected,np.flatnonzero(certificate[:,0]==0))
            expected=certificate[:,11].astype(np.float32).reshape(H,W).copy();expected[~mask]=0;assert np.array_equal(expected.view(np.uint32),control.view(np.uint32));expected2=certificate[:,11].astype(np.float32).copy();expected2[selected]=0;good=root[:,0]==1;expected2[selected[good]]=root[good,2].astype(np.float32);expected2=expected2.reshape(H,W);expected2[~mask]=0;assert np.array_equal(expected2.view(np.uint32),candidate.view(np.uint32))
            assert np.all(root[good,2]>0) and np.isfinite(root[good]).all() and np.all(root[good,15]==0) and np.all(root[good,4]<root[good,5]);assert np.sum(good)==row['corrected'] and np.sum((root[:,0]==2)|(root[:,0]==3))==row['no_finite_root'] and np.sum(root[:,0]==0)==row['unresolved']
            for method in [1,2,3]:
                use=good&(root[:,12]==method)
                if method==1:assert np.all(root[use,6]>.5) and np.all(root[use,9]<.5)
                else:assert np.all(root[use,7]<0) and np.all(root[use,8]>0)
            cert=certificate[:,0]==1;assert np.all(certificate[cert,5]>.5) and np.all(certificate[cert,8]<.5) and np.all(certificate[cert,10]==0);assert np.sum(control>0)==row['masked_control_valid'] and np.sum(candidate>0)==row['masked_certified_valid']
            for k,v in [('views',1),('pixels',H*W),('records',row['records']),('selected',len(selected)),('corrected',row['corrected']),('no_root',row['no_finite_root']),('unresolved',row['unresolved']),('masked_control_valid',row['masked_control_valid']),('masked_certified_valid',row['masked_certified_valid'])]:totals[k]+=int(v)
            masked[0]+=int(np.sum(control>0));masked[1]+=int(np.sum(candidate>0))
        means={}
        for arm in ['control','certified']:
            for n,h in r['meshes'][arm]['files_sha256'].items():assert sha(folder/arm/n)==h,n
            metrics=[]
            for seed in [0,1,2]:
                e=folder/f'eval_{arm}_seed{seed}';em=read(e/'manifest.json');er=read(e/'results.json');assert em['status']=='complete' and em['shuffle_seed']==seed
                for p,h in em['source_sha256'].items():assert sha(Path(p))==h,p
                assert all(np.isfinite(er[k]) for k in ['overall','mean_d2s','mean_s2d']);assert abs(er['overall']-.5*(er['mean_d2s']+er['mean_s2d']))<1e-12;metrics.append(er)
            means[arm]={k:float(np.mean([q[k] for q in metrics])) for k in metrics[0]};assert all(abs(means[arm][k]-scene['means'][arm][k])<1e-12 for k in means[arm])
        resultrows.append(dict(scan=scan,means=means,coverage_ratio=masked[1]/masked[0],s2d_delta=means['certified']['mean_s2d']-means['control']['mean_s2d']))
    ratio=float(np.mean([r['means']['certified']['overall'] for r in resultrows])/np.mean([r['means']['control']['overall'] for r in resultrows]));gate=ratio<=.99 and all(r['s2d_delta']<=0 for r in resultrows);assert totals['views']==147 and ratio==s['mean_overall_ratio'] and gate==s['quality_gate_pass']
    result=dict(status='complete',checked_utc=now(),source_checks=len(m['source_sha256']),totals=totals,all147_original_output_bits_pass=True,all_masked_depth_arrays_and_saved_certificates_accounted=True,all18_seeded_evaluations_verified=True,mean_overall_ratio=ratio,quality_gate_pass=gate,audit_gate_pass=True,scenes=resultrows,images_sha256=artifacts,verifier_sha256=sha(Path(__file__)),scope='Complete saved data and provenance/accounting; not individual MP checks of all147-view source coefficients, training replications, CUDA backward or novelty.')
    save(out,result);print(json.dumps({k:v for k,v in result.items() if k not in ['images_sha256']},indent=2))
if __name__=='__main__':main()
