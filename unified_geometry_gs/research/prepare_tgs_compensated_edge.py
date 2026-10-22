"""Freeze arithmetic-only probes from sealed caches, no torch/native execution."""
import hashlib,json,struct
from fractions import Fraction as F
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_compensated_edge_v6'
OLD=ROOT/'experiments/tgs_phase_cost_v5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(x):return F.from_float(float(x))
def ceil32(x):
    assert x>=0
    f=np.float32(float(x))
    if q(f)<x:f=np.nextafter(f,np.float32(np.inf))
    assert q(f)>=x
    return f
def main():
    assert not OUT.exists();OUT.mkdir()
    manifest=OLD/'evidence_manifest.json';expected='c8ab7c32e93b5aca6a830d41fdc04b2bc1e8205b852a9a0a39f6bfdd8a9af08d';assert sha(manifest)==expected
    hashes={e['path']:e['sha256'] for e in json.loads(manifest.read_text())['files']};used={}
    def checked(p):
        rel=p.relative_to(ROOT).as_posix();value=sha(p);assert value==hashes[rel];used[rel]=value
        return p
    plan=json.loads(checked(OLD/'input_plan.json').read_text());rng=np.random.default_rng(202610208)
    coeff=[];xy=[];labels=[]
    def append(c,x,y,label):coeff.append(c.copy());xy.append([x,y]);labels.append(label)
    def plane(c,label):
        p=rng.integers(-1600,1601,2);append(c,*p,dict(label,point='local_random_1600'))
        p=rng.integers(-8192,8193,2);append(c,*p,dict(label,point='local_random_8192'))
        x=int(rng.integers(-1600,1601));y=int(np.clip(np.rint(-(c[0]*x+c[2])/c[1]),-8192,8192)) if c[1]!=0 else 0
        append(c,x,y,dict(label,point='near_line_integer'));append(c,x,min(8192,y+1),dict(label,point='near_line_integer_plus1'))
    for e in plan['cases']:
        p=checked(OLD/('oracle_'+e['name']+'.npz')) if 'name' in e else None
        # The canonical name is also the NPZ input scalar; keep original input provenance.
        inp=ROOT/e['input'];assert sha(inp)==e['sha256'];used[e['input']]=e['sha256']
        with np.load(inp) as z:name=str(z['name'])
        if p is None:p=checked(OLD/('oracle_'+name+'.npz'))
        with np.load(p) as z:c=z['double_coeff'];valid=np.flatnonzero(z['native_radii']>0)
        for t in valid:
            for k in range(3):plane(c[t,[2*k,2*k+1,6+k]],dict(family='retained_case',name=name,triangle=int(t),edge=k))
    real_selection=[]
    for ci in range(4):
        p=checked(OLD/f'real_{ci}_oracle.npz')
        with np.load(p) as z:c=z['double_coeff'];valid=np.flatnonzero(z['radii']>0)
        chosen=rng.choice(valid,size=512,replace=False);real_selection.append(chosen.tolist())
        for t in chosen:
            for k in range(3):plane(c[t,[2*k,2*k+1,6+k]],dict(family='real_cache_artificial_coordinates',camera=ci,triangle=int(t),edge=k))
    for exponent in [-20,-40,-60,-80,-126,-140]:
        for c,coords,kind in [(np.array([1+2.**-30,-1,2.**exponent]),[8192,8192],'cancellation'),(np.array([0.,0.,2.**exponent]),[1,1],'tiny_constant'),(np.array([2.**exponent,0,0]),[1,0],'tiny_product')]:
            append(c,*coords,dict(family='control',kind=kind,exponent=exponent))
    for _ in range(128):
        c=rng.uniform(-1,1,3);x,y=rng.integers(-8192,8193,2);c[2]=-c[0]*x-c[1]*y+2.**int(rng.integers(-48,-12));append(c,x,y,dict(family='control',kind='random_cancellation'))
    append(np.array([1.,-1.,0]),.5,.5,dict(family='control',kind='noninteger'))
    append(np.array([1.,-1.,0]),8193,8193,dict(family='control',kind='outside_coordinate_domain'))
    c=np.array(coeff,dtype='<f8');xy=np.array(xy,dtype='<f4');hi=c.astype(np.float32);lo=(c-hi.astype(float)).astype(np.float32);a=np.concatenate([hi,lo],axis=1).astype('<f4')
    cache=np.empty((len(c),4),dtype='<f4');gate=[]
    B=[F(8192),F(8192),F(1)]
    for i,(cc,hh,ll) in enumerate(zip(c,hi,lo)):
        cq=list(map(q,cc));hq=list(map(q,hh));lq=list(map(q,ll))
        cache[i]=[ceil32(sum((abs(h)+abs(l))*b for h,l,b in zip(hq,lq,B))),ceil32(sum(abs(z-h-l)*b for z,h,l,b in zip(cq,hq,lq,B))),ceil32(sum(abs(z)*b for z,b in zip(cq,B))),ceil32(sum(abs(z-h)*b for z,h,b in zip(cq,hq,B)))]
        values=np.r_[cc,hh,ll];gate.append(bool(np.isfinite(values).all() and np.all((values==0)|((np.abs(values)>=2.**-60)&(np.abs(values)<=2.**20))) and np.all(xy[i]==np.rint(xy[i])) and np.max(np.abs(xy[i]))<=8192))
    us=np.unique(np.r_[np.geomspace(float(np.float32(2.**-149)),1.,192).astype(np.float32),np.array([1/255,.5,.99,1.,np.nextafter(np.float32(1),np.float32(0))],np.float32)])
    power=np.array([(u,s) for u in us for s in np.array([.02,.2,.216805,1,3.532998,10],np.float32)],dtype='<f4')
    np.savez_compressed(OUT/'inputs.npz',coeff=c,split=a,xy=xy,cache=cache,gate=np.array(gate),power=power)
    with (OUT/'input.bin').open('wb') as f:
        f.write(struct.pack('<qq',len(c),len(power)))
        for arr in [c,a,xy,cache,power]:f.write(arr.tobytes())
    (OUT/'labels.json').write_text(json.dumps(labels,separators=(',',':'))+'\n')
    frozen=dict(seed=202610208,rows=len(c),power_rows=len(power),in_domain=sum(gate),cases=len(plan['cases']),real_triangles_per_camera=512,real_selection=real_selection,protocol_sha256=sha(ROOT/'TGS_COMPENSATED_EDGE_PROTOCOL_V6.md'),source_sha256=sha(ROOT/'tgs_compensated_edge_probe.cu'),prepare_sha256=sha(Path(__file__)),v5_manifest_sha256=expected,used_sources=used,files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},training=False,renderer=False,timing=False,new_holdout=False,gate_is_sufficient_not_necessary=True)
    (OUT/'input_plan.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps({k:v for k,v in frozen.items() if k not in ['used_sources','real_selection','files']}),flush=True)
if __name__=='__main__':main()
