"""Independent all-array raw and shared statistics/provenance audit."""
from pathlib import Path
from decimal import Decimal,localcontext
import hashlib,json,zipfile
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify_manifest(folder):
    m=read(folder/'manifest.json');assert m['status']=='complete'
    for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in m['artifacts_sha256'].items():assert sha(folder/name)==h,name
    with zipfile.ZipFile(folder/'sources.zip') as z:
        for name,h in m['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    return dict(sources=len(m['source_sha256']),artifacts=len(m['artifacts_sha256']))
def relative(a,b,floor):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),floor))
def main():
    out=R/'results/history_gap_raw_audit_v1.json';assert not out.exists();RAW=R/'experiments/history_gap_raw_v1';MICRO=R/'experiments/history_gap_raw_micro_v1';SHARED=R/'experiments/history_gap_shared_v1';WORLD=R/'experiments/world_root_vjp_v1'
    provenance={p.name:verify_manifest(p) for p in [RAW,MICRO,SHARED]};s=read(RAW/'summary.json');lookup={(r['view'],r['pixel']):r for r in s['rows']};counts={'factor':0,'dense':0};maxima={k:0. for k in counts};blocksmax={k:{} for k in counts};rays=nodes=0;baselines=[];allscans=[];allkeys=[];allg=[];allraw=[];allscale=[]
    for f in sorted(RAW.glob('scan*_gradients.npz')):
        stem=f.name.removesuffix('_gradients.npz');scan=int(stem[4:6])
        with np.load(WORLD/f.name) as d:originalpixels=d['pixels'];osections=d['sections'];ids=d['primitive_ids'];raw=d['raw'];src=d['source']
        with np.load(f) as d:
            assert np.array_equal(d['pixels'],originalpixels) and np.array_equal(d['sections'],osections) and np.array_equal(d['primitive_ids'],ids)
            reference=d['reference'];assert reference.shape==(len(raw),11) and np.isfinite(reference).all()
            for j,p in enumerate(originalpixels):
                a,b=map(int,osections[j:j+2]);r=lookup[stem,int(p)];assert r['index']==rays and r['nodes']==b-a
                for arm in counts:
                    actual=d[arm][a:b];assert np.isfinite(actual).all();rel=relative(actual,reference[a:b],1e-300);blocks={key:relative(actual[:,i:k],reference[a:b,i:k],1e-30) for key,i,k in [('xyz',0,3),('log_scale',3,6),('quaternion',6,10),('opacity_logit',10,11)]};passed=rel<=1e-5 and max(blocks.values())<=1e-4
                    assert rel==r['branches'][arm]['relative_l2'] and blocks==r['branches'][arm]['blocks'] and passed==r['branches'][arm]['gate_pass'];counts[arm]+=passed;maxima[arm]=max(maxima[arm],rel)
                    for key,value in blocks.items():blocksmax[arm][key]=max(blocksmax[arm].get(key,0),value)
                baseline=r['branches']['factor']['loss'];baselines.append(baseline);allscans.append(scan);assert r['high_loss']==(r['reference_loss']>=.25)
                for step in r['descent']['steps']:
                    assert step['status']=='finite';delta=step['loss']-baseline;assert step['delta']==delta and step['armijo']==(step['loss']<=baseline-1e-4*step['h']*r['descent']['scaled_gradient_norm']) and step['measurable_decrease']==(delta< -max(1e-12,1e-8*baseline))
                rays+=1
            allg.append(d['factor']);nodes+=len(raw)
        scale=np.ones((len(raw),11));scale[:,:3]=np.min(src[:,3:6],axis=1)[:,None];scale[:,6:10]=np.linalg.norm(raw[:,6:10],axis=1)[:,None];allscale.append(scale);allraw.append(raw);allkeys.append(np.int64(scan)*np.int64(2**32)+ids.astype(np.int64))
    assert rays==891 and nodes==51524 and counts==s['branch_pass_counts']==dict(factor=891,dense=890)
    direction=[d for r in s['rows'] for d in r.get('directional',[])];assert len(direction)==42 and sum(d['gate_pass'] for d in direction)==35
    synthetic=read(RAW/'synthetic.json');assert [r['kernel'] for r in synthetic['rows']]==[0.,.3,1.] and all(r['gate_pass'] for r in synthetic['rows'])
    micro=read(MICRO/'summary.json');assert len(micro['rows'])==7;micro_pass=0
    with localcontext() as ctx:
        ctx.prec=80
        for row in micro['rows']:
            original=lookup[row['view'],row['pixel']];d=next(d for d in original['directional'] if d['direction']==row['direction']);assert not d['gate_pass'];last=row['steps'][-1];pred=Decimal(row['predicted_reference']);fd=Decimal(last['mp_derivative']);passed=abs(pred-fd)<=max(Decimal('1e-9'),Decimal('1e-5')*abs(pred));assert passed==row['micro_mp_gate_pass'];micro_pass+=passed;assert all(not st['cap_changed'] for st in row['steps'])
    assert micro_pass==7
    sh=read(SHARED/'summary.json');baseline=np.array(baselines);assert np.array_equal(np.load(SHARED/'baseline.npy'),baseline);keys=np.concatenate(allkeys);params=np.concatenate(allraw);scale=np.concatenate(allscale);g=np.concatenate(allg);unique,first,inverse,repeat=np.unique(keys,return_index=True,return_inverse=True,return_counts=True);aggregate=np.zeros((len(unique),11));np.add.at(aggregate,inverse,g/891)
    with np.load(SHARED/'shared_parameters.npz') as d:
        assert np.array_equal(d['keys'],unique) and np.array_equal(d['node_to_unique'],inverse) and np.array_equal(d['counts'],repeat)
        assert np.array_equal(d['raw'],params[first]) and np.array_equal(params,d['raw'][inverse]) and np.array_equal(d['scale'],scale[first]) and np.array_equal(scale,d['scale'][inverse]) and np.array_equal(d['mean_gradient'],aggregate)
        for arm in ['full','geometry','opacity']:
            active=aggregate.copy()
            if arm=='geometry':active[:,10]=0
            elif arm=='opacity':active[:,:10]=0
            norm=float(np.linalg.norm(d['scale']*active));stored_norm=sh['scaled_gradient_norms'][arm];assert abs(norm-stored_norm)<=1e-13*stored_norm
            expected=-d['scale']*d['scale']*active/stored_norm;assert np.array_equal(expected,d[arm])
    assert len(unique)==sh['unique_primitives'] and int((repeat>1).sum())==sh['repeated_primitives'];assert np.array_equal(np.load(SHARED/'zero_loss.npy').view(np.uint64),baseline.view(np.uint64))
    arrays={}
    for row in sh['rows']:
        label=row['branch'];values=np.load(SHARED/(label+'_loss.npy'));states=read(SHARED/(label+'_roots.json'))['rows'];assert len(values)==len(states)==891;finite=np.isfinite(values);assert row['finite_rays']==int(finite.sum()) and row['all_rays_finite']==bool(finite.all())
        for j,state in enumerate(states):assert finite[j]==('root' in state)
        if finite.all():
            assert row['mean_loss']==float(values.mean()) and row['mean_delta']==float(values.mean())-float(baseline.mean());assert row['increased_rays']==int((values>baseline).sum()) and row['decreased_rays']==int((values<baseline).sum()) and row['unchanged_rays']==int((values==baseline).sum())
            for scan in [37,55,63]:assert row['scene_means'][str(scan)]==float(values[np.array(allscans)==scan].mean())
        else:assert 'mean_loss' not in row and 'mean_delta' not in row
        arrays[label]=values
    for row in sh['directional']:
        h=row['h'];plus=arrays['full_0.0001' if h==1e-4 else 'full_plus_1e-05'];minus=arrays[f'full_minus_{h}'];fd=float((plus.mean()-minus.mean())/(2*h));assert fd==row['derivative'];passed=abs(fd-row['predicted'])<=max(1e-9,1e-5*abs(row['predicted']));assert passed==row['gate_pass']
    dep=R/'exact_outlier_world_reference.py';worldm=read(WORLD/'manifest.json');assert sha(dep)==worldm['source_sha256'][dep.relative_to(ROOT).as_posix()]
    result=dict(status='complete',audit_gate_pass=True,original_raw_gate_pass=False,original_direction_pass=35,original_direction_total=42,mp_followup_pass=7,rays=rays,nodes=nodes,branch_pass_counts=counts,max_relative_l2=maxima,max_block_relative_l2=blocksmax,high_loss_count=s['high_loss_count'],high_loss_weak_gradient_count=s['high_loss_weak_gradient_count'],provenance=provenance,shared_unique_primitives=sh['unique_primitives'],shared_repeated_primitives=sh['repeated_primitives'],shared_directional=sh['directional'],shared_zero_control_bits_equal=True,all_shared_branch_rays_finite=all(row['all_rays_finite'] for row in sh['rows']),shared_rows=sh['rows'],scope='Provenance and full-array/statistical verification; original numerical gates retained, no CUDA/training/GT/adoption claim.')
    save(out,result);print(json.dumps({k:v for k,v in result.items() if k!='shared_rows'}))
if __name__=='__main__':main()
