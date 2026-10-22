"""Independent complete gradient-array statistics, provenance and root-point audit."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;BASE=R/'experiments/world_root_vjp_v1';OUT=R/'results/world_root_vjp_audit_v1.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def errors(a,b):
    relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300));blocks={}
    for k,l,h in [('xyz',0,3),('log_scale',3,6),('quaternion',6,10),('opacity_logit',10,11)]:blocks[k]=float(np.linalg.norm(a[:,l:h]-b[:,l:h])/max(np.linalg.norm(b[:,l:h]),1e-30))
    return relative,blocks,relative<=1e-5 and max(blocks.values())<=1e-4
def main():
    assert not OUT.exists();m=read(BASE/'manifest.json');s=read(BASE/'summary.json');assert m['status']==s['status']=='complete'
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in m['artifacts_sha256'].items():assert sha(BASE/n)==h,n
    with zipfile.ZipFile(BASE/'sources.zip') as z:
        for n,h in m['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    rootmeta=read(R/'experiments/dynamic_root_v1/manifest.json');rowlookup={(r['view'],r['pixel']):r for r in s['rows']};rays=nodes=0;counts={k:0 for k in s['branch_counts']};maxima={k:0. for k in counts};kernel_values=[];norms=[]
    for f in sorted(BASE.glob('scan*_gradients.npz')):
        stem=f.name.removesuffix('_gradients.npz');old=R/'experiments/dynamic_root_v1'/(stem+'_probes.npz');assert sha(old)==rootmeta['artifacts_sha256'][old.name]
        with np.load(old) as d:oldpixels=d['pixels'];oldsource=d['source'];oldsections=d['sections'];oldids=d['primitive_ids'];oldoutput=d['output']
        with np.load(f) as d:pix=d['pixels'];sections=d['sections'];source=d['source'];raw=d['raw'];ids=d['primitive_ids'];arrays={k:d[k] for k in ['reference',*counts]}
        assert np.array_equal(pix,oldpixels) and np.array_equal(source.view(np.uint64),oldsource.view(np.uint64)) and np.array_equal(sections,oldsections) and np.array_equal(ids,oldids) and np.array_equal(raw[:,:3],source[:,:3])
        assert all(a.shape==(len(source),11) and np.isfinite(a).all() for a in arrays.values());norms.extend(np.linalg.norm(raw[:,6:10],axis=1).tolist());view=read(BASE/(stem+'.json'));meta=view['meta'];kernel_values.append(meta['kernel']);camera=read(R/'experiments/full_topology_v1'/(stem+'.json'));cs=camera['camera_scalar_inputs'];assert meta['kernel']==float(np.float32(cs['18']))
        tx=np.float32(cs['16']);ty=np.float32(cs['17']);assert meta['fx']==float(np.float32(np.float32(camera['width'])/(np.float32(2)*tx))) and meta['fy']==float(np.float32(np.float32(camera['height'])/(np.float32(2)*ty)))
        for j,p in enumerate(pix):
            r=rowlookup[(stem,int(p))];assert r['reference_status']=='regular';a,b=int(sections[j]),int(sections[j+1]);assert b-a==r['nodes'];ref=arrays['reference'][a:b]
            lo,hi=map(float,r['reference_bracket']);assert oldoutput[j,4]<=lo<=hi<=oldoutput[j,5]
            for branch in counts:
                v=r['branches'][branch];relative,blocks,passed=errors(arrays[branch][a:b],ref);assert v['status']=='finite' and abs(relative-v['relative_l2'])<=1e-15 and all(abs(blocks[k]-v['blocks'][k])<=1e-15 for k in blocks) and passed==v['gate_pass'];counts[branch]+=passed;maxima[branch]=max(maxima[branch],relative)
            rays+=1
        nodes+=len(source)
    assert rays==891 and nodes==51524 and all(counts[k]==s['branch_counts'][k]['pass_count'] for k in counts) and counts['factor_refined']==891
    result=dict(status='complete',gate_pass=True,source_checks=len(m['source_sha256']),artifact_checks=len(m['artifacts_sha256']),rays=rays,nodes=nodes,all_gradient_arrays_finite=True,branch_pass_counts=counts,max_relative_l2=maxima,raw_quaternion_min_norm=min(norms),all_real_screen_kernel_zero=all(k==0 for k in kernel_values),full_source_and_root_mapping_pass=True,all9_old_anomalies_present=s['all9_old_anomalies_present'],limits=['MP analytic reference and CPU graph only; not fullCUDA training backward or IEEE literal derivative.','Actual24 views have kernel0; nonzero screen compensation derivatives are not empirical evidence from this batch.','891 previously selected development rays; no population error-rate estimate or GT improvement.'])
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
