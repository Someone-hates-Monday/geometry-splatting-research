from pathlib import Path
from fractions import Fraction
import hashlib,json,zipfile
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;BASE=R/'experiments/history_gap_warp_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metrics(a,b):
    relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300));blocks={k:float(np.linalg.norm(a[:,i:j]-b[:,i:j])/max(np.linalg.norm(b[:,i:j]),1e-30)) for k,i,j in [('xyz',0,3),('log_scale',3,6),('quaternion',6,10),('opacity_logit',10,11)]};return relative,blocks,relative<=1e-5 and max(blocks.values())<=1e-4
def main():
    out=R/'results/history_gap_warp_audit_v1.json';assert not out.exists();m=read(BASE/'manifest.json');s=read(BASE/'summary.json');assert m['status']==s['status']=='complete'
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in m['artifacts_sha256'].items():assert sha(BASE/n)==h,n
    assert sha(ROOT/m['binary']['path'])==m['binary']['sha256']
    with zipfile.ZipFile(BASE/'sources.zip') as z:
        for n,h in m['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    ref=np.concatenate([np.load(p)['reference'] for p in sorted((R/'experiments/history_gap_raw_v1').glob('scan*_gradients.npz'))]);counts={k:0 for k in ['serial','warp']};maxima={k:0. for k in counts};blockmax={k:0. for k in counts}
    with np.load(BASE/'real_arrays.npz') as d:
        assert np.array_equal(d['reference'],ref) and ref.shape==(51524,11) and len(s['real_rows'])==891
        for j,row in enumerate(s['real_rows']):
            a,b=map(int,d['sections'][j:j+2])
            for arm in counts:
                actual=d[arm+'_gradient'][a:b];assert np.isfinite(actual).all();relative,blocks,passed=metrics(actual,ref[a:b]);saved=row['branches'][arm];loss_error=abs(float(d[arm+'_loss'][j])-row['reference_loss']);assert relative==saved['relative_l2'] and blocks==saved['blocks'] and loss_error==saved['loss_error'];status=int(d[arm+'_info'][j,0]);assert status==saved['status'];gate=passed and loss_error<=1e-9 and status>0;assert gate==saved['gate_pass'];counts[arm]+=gate;maxima[arm]=max(maxima[arm],relative);blockmax[arm]=max(blockmax[arm],max(blocks.values()))
                if status==3:assert not np.count_nonzero(actual) and d[arm+'_loss'][j]==0
        assert sum(r['old_anomaly'] is not None for r in s['real_rows'])==9
    assert counts==s['real_pass_counts']==dict(serial=891,warp=891)
    original=read(R/'experiments/history_gap_gpu_confirm_v4/summary.json')['rows'];confirmed=0
    with np.load(BASE/'confirm_arrays.npz') as d:
        with np.load(R/'experiments/history_gap_gpu_confirm_v4/new_arrays.npz') as old:assert np.array_equal(d['reference'],old['reference']) and np.array_equal(d['sections'],old['sections'])
        for j,row in enumerate(s['confirm_rows']):
            a,b=map(int,d['sections'][j:j+2]);relative,blocks,passed=metrics(d['actual'][a:b],d['reference'][a:b]);error=abs(float(d['loss'][j])-original[j]['reference_loss']);assert row['relative_l2']==relative and row['blocks']==blocks and row['loss_error']==error;gate=passed and error<=1e-9 and d['info'][j,0]>0;assert gate==row['gate_pass'];confirmed+=gate
    assert confirmed==64 and all(r['gate_pass'] for r in s['absent_updates'])
    long=[]
    for row in s['special']:
        with np.load(BASE/(row['name']+'_coefficient.npz')) as d:
            actual=d['actual'];expected=d['expected'];assert np.isfinite(actual).all();relative=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-300));error=abs(float(d['loss'][0])-float(d['reference_loss'][0]));assert relative==row['relative_seed_error'] and error==row['loss_error'] and row['gate_pass']
            n=len(actual);assert n==row['nodes']==row['reported_nodes']==int(d['info'][0,5])
            if n>512:
                a=Fraction(float(d['amplitude'][0]));last=Fraction(float(d['amplitude'][-1]));prefix=(1-a)**(n-1);assert prefix>Fraction(1,2)>prefix*(1-last);long.append(n)
    assert long==[513,1025] and len(s['special'])==12 and s['invalid_input_rejected']
    medians={};km={}
    for name,key in [('full','timings'),('kernel','kernel_timings')]:
        rows=s[key];assert len(rows)==20
        for it in range(10):assert [r['arm'] for r in rows if r['iteration']==it]==(['serial','warp'] if it%2==0 else ['warp','serial'])
        med={}
        for arm in counts:
            values=[r['ms'] for r in rows if r['arm']==arm and not r['warmup']];assert len(values)==8;med[arm]=float(np.median(values))
        if name=='full':medians=med;assert med==s['full_median_ms']
        else:km=med;assert med==s['kernel_median_ms']
    ratio=medians['warp']/medians['serial'];assert ratio==s['full_ratio'] and ratio<=.95 and s['correctness_gate_pass'] and s['adoption_gate_pass']
    result=dict(status='complete',audit_gate_pass=True,source_checks=len(m['source_sha256']),artifact_checks=len(m['artifacts_sha256']),binary_sha256=m['binary']['sha256'],real_pass_counts=counts,max_relative_l2=maxima,max_block_relative_l2=blockmax,confirmed=64,special_cases=12,long_counts_exact_mass_witness=long,full_median_ms=medians,kernel_median_ms=km,full_ratio=ratio,component_speedup=medians['serial']/medians['warp'],adoption_gate_pass=True,scope='All frozen references, saved loss/gradient arrays, source ZIP, actual binary and alternating timing statistics audited; no whole-training speed, GT or novelty conclusion.')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
