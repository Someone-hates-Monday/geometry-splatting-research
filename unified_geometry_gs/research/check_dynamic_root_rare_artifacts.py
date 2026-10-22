"""Independent source/ZIP/full conditional-selection and mapping audit."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    out=R/'results/dynamic_root_rare_artifacts_v1.json';assert not out.exists();b=R/'experiments/dynamic_root_rare_v1';m=read(b/'manifest.json');a=read(R/'results/dynamic_root_rare_audit_v1.json');s=read(R/'experiments/dynamic_root_v1/summary.json');assert m['status']=='complete' and a['gate_pass']
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in m['artifacts_sha256'].items():assert sha(b/n)==h,n
    with zipfile.ZipFile(b/'sources.zip') as z:
        for n,h in m['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    seen=0;nodes=0
    for r in s['rows']:
        stem=f'scan{r["scan"]}_{r["camera"]}'
        with np.load(R/'experiments/dynamic_root_v1'/(stem+'_output.npz')) as d:pp=d['pixels'];oo=d['output']
        wanted=((oo[:,0]==2)|(oo[:,0]==3)|((oo[:,0]==1)&((oo[:,1]==1)|(oo[:,12]==2))))
        if not np.any(wanted):assert not (b/(stem+'_probes.npz')).exists();continue
        with np.load(b/(stem+'_probes.npz')) as d:p=d['pixels'];o=d['output'];src=d['source'];sections=d['sections'];indices=d['record_indices'];ids=d['primitive_ids'];q=d['query'];packets=d['packets']
        assert np.array_equal(p,pp[wanted]) and np.array_equal(o.view(np.uint64),oo[wanted].view(np.uint64))
        with np.load(R/'experiments/csr_capture_v1'/(stem+'.npz')) as d:offset=d['offsets'];records=d['records']
        with np.load(R/'experiments/csr_decode_v1'/(stem+'_query_amp.npz')) as d:query=d['query_xy'];amp=d['amplitude']
        expected=np.concatenate([np.arange(offset[i],offset[i+1],dtype=np.int64) for i in p]);assert np.array_equal(indices,expected) and np.array_equal(ids,records[indices,0]) and np.array_equal(np.diff(sections),np.diff(offset)[p]) and np.array_equal(q,query[p])
        assert np.array_equal(src[:,32:34],query[np.repeat(p,np.diff(sections))].astype(np.float64)) and np.all(src[:,34]==1) and np.array_equal(src[:,35],amp[ids].astype(np.float64)) and np.array_equal(packets[:,7],src[:,35]);seen+=len(p);nodes+=len(src)
    assert seen==a['conditional_subset_rays']==299 and nodes==a['world_nodes']==5920
    result=dict(status='complete',source_checks=len(m['source_sha256']),artifact_checks=len(m['artifacts_sha256']),all_conditional_selection_complete=True,all_output_bits_preserved=True,all_actual_query_amp_CSR_mappings_pass=True,rays=seen,nodes=nodes,gate_pass=True)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
