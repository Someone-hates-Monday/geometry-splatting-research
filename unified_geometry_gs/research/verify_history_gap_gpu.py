from pathlib import Path
import hashlib,json,zipfile
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metrics(a,b):
    rel=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300));blocks={k:float(np.linalg.norm(a[:,i:j]-b[:,i:j])/max(np.linalg.norm(b[:,i:j]),1e-30)) for k,i,j in [('xyz',0,3),('log_scale',3,6),('quaternion',6,10),('opacity_logit',10,11)]};return rel,blocks,rel<=1e-5 and max(blocks.values())<=1e-4 and np.isfinite(a).all()
def provenance(folder):
    m=read(folder/'manifest.json');assert m['status']=='complete'
    for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in m['artifacts_sha256'].items():assert sha(folder/name)==h,name
    with zipfile.ZipFile(folder/'sources.zip') as z:
        for name,h in m['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    if 'binary'in m:assert sha(ROOT/m['binary']['path'])==m['binary']['sha256']
    return dict(sources=len(m['source_sha256']),artifacts=len(m['artifacts_sha256']))
def main():
    out=R/'results/history_gap_gpu_audit_v1.json';assert not out.exists();folders=[R/'experiments'/n for n in ['history_gap_gpu_v1','history_gap_gpu_v2','history_gap_gpu_confirm_v3','history_gap_gpu_confirm_v4']];prov={p.name:provenance(p) for p in folders};real=[]
    ref=np.concatenate([np.load(f)['reference'] for f in sorted((R/'experiments/history_gap_raw_v1').glob('scan*_gradients.npz'))])
    for folder in folders[:2]:
        s=read(folder/'summary.json');passed=0;maxblock=0
        with np.load(folder/'real_arrays.npz') as d:
            assert d['actual'].shape==d['reference'].shape==(51524,11) and np.array_equal(d['reference'],ref) and np.isfinite(d['actual']).all()
            for j,row in enumerate(s['rows']):
                a,b=map(int,d['sections'][j:j+2]);relative,blocks,ok=metrics(d['actual'][a:b],ref[a:b]);assert relative==row['gradient']['relative_l2'] and blocks==row['gradient']['blocks'];error=abs(float(d['loss'][j])-row['reference_loss']);assert error==row['loss_error'] and row['status']==int(d['info'][j,0]);gate=bool(ok and error<=1e-9 and row['status']>0);assert gate==row['gate_pass'];passed+=gate;maxblock=max(maxblock,max(blocks.values()))
                if row['status']==3:assert d['loss'][j]==0 and np.count_nonzero(d['actual'][a:b])==0
            assert passed==s['pass_count']==891 and sum(r['old_anomaly'] is not None for r in s['rows'])==9
        real.append(dict(version=folder.name,pass_count=passed,max_relative_l2=s['max_relative_l2'],max_block_relative_l2=maxblock,status_counts=s['status_counts'],timed_median_ms=s['timed_median_ms'],peak_allocated_bytes=s['peak_allocated_bytes']))
    confirmation=[]
    for folder,expected in [(folders[2],28),(folders[3],64)]:
        s=read(folder/'summary.json');passed=0;maximum=0
        with np.load(folder/'new_arrays.npz') as d:
            for j,row in enumerate(s['rows']):
                a,b=map(int,d['sections'][j:j+2]);relative,blocks,ok=metrics(d['actual'][a:b],d['reference'][a:b]);assert relative==row['gradient']['relative_l2'] and blocks==row['gradient']['blocks'];loss_error=abs(float(d['loss'][j])-row['reference_loss']);gate=bool(ok and loss_error<=1e-9 and d['info'][j,0]>0);assert gate==row['gate_pass'];passed+=gate;maximum=max(maximum,relative)
        assert passed==s['new_pass']==expected and all(r['gate_pass'] for r in s['real_absent_updates'])
        confirmation.append(dict(version=folder.name,pass_count=passed,total=s['new_cases'],max_relative_l2=maximum,real_absent_updates=s['real_absent_updates']))
    s=read(folders[3]/'summary.json');assert sum(r['new_confirmation'] for r in s['rows'])==32 and all(r['gate_pass'] for r in s['rows'] if r['new_confirmation'])
    profile=read(R/'results/history_gap_gpu_profile_v1.json');assert all(r['gate_pass'] for r in profile['coefficient_special_cases']) and profile['invalid_input_explicitly_rejected']
    for name,h in profile['source_sha256'].items():assert sha(ROOT/name)==h
    result=dict(status='complete',audit_gate_pass=True,provenance=prov,real=real,confirmation=confirmation,new_seed13019_pass=32,old_failed_confirmation_preserved=True,special_coefficient_checks=len(profile['coefficient_special_cases']),invalid_input_explicitly_rejected=True,scope='All saved gradients/losses, exact prior MP reference mapping, source ZIPs and actual binaries audited. Float64 signs are numerical, not interval-certified. No training/GT/whole-pipeline speed/newness claim.')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
