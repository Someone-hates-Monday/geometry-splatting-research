from pathlib import Path
import hashlib,json,os,time,traceback,zipfile
import numpy as np
import mpmath as mp
from research.history_gap_extension import mp_extended,mp_path_nodes,torch_absent_gradient
from research.history_gap_math import mp_root
R=Path(__file__).resolve().parent;ROOT=R.parent;SH=R/'experiments/history_gap_shared_v1';WORLD=R/'experiments/world_root_vjp_v1';OUT=R/'experiments/history_gap_extension_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def independent_loss(nodes):
    omega=-mp.fsum(mp.log1p(-a) for a,p,rho in nodes)
    if omega<=mp.log(2):return (mp.log(4)+mp.log(mp.fprod(1-a for a,p,rho in nodes)))**2
    t=mp_root(nodes);ell=mp.fsum(mp.log1p(-a*mp.exp(-((t-p)*rho)**2/2))/2 for a,p,rho in nodes);return (ell+mp.log(2))**2
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);files=[Path(__file__),R/'history_gap_extension.py',R/'HISTORY_GAP_EXTENSION_PROTOCOL.md',R/'history_gap_math.py',R/'history_gap_raw.py',SH/'manifest.json',SH/'summary.json',SH/'shared_parameters.npz'];m=dict(status='running',created_epoch=time.time(),pid=os.getpid(),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files});save(OUT/'manifest.json',m)
    try:
        mp.mp.dps=240;rng=np.random.default_rng(13017);synthetic=[]
        for case in range(16):
            n=[1,2,4,8][case%4];target=mp.mpf(['.3','.65','.7','1','1.5','3'][case%6]);w=rng.uniform(.2,1,n);w=w/w.sum();nodes=[[1-mp.exp(-target*mp.mpf(float(ww))),mp.mpf(float(p)),mp.mpf(float(rho))] for ww,p,rho in zip(w,rng.uniform(-1,1,n),rng.uniform(.5,3,n))];loss,g,info=mp_extended(nodes);errors=[]
            for hh in ['1e-30','1e-35']:
                h=mp.mpf(hh);fd=[]
                for i in range(n):
                    row=[]
                    for k in range(3):
                        plus=[list(v) for v in nodes];minus=[list(v) for v in nodes];plus[i][k]+=h;minus[i][k]-=h;row.append((independent_loss(plus)-independent_loss(minus))/(2*h))
                    fd.append(row)
                den=max(mp.sqrt(mp.fsum(v*v for row in g for v in row)),mp.mpf('1e-300'));errors.append(mp.sqrt(mp.fsum((a-b)**2 for row,ref in zip(g,fd) for a,b in zip(row,ref)))/den)
            synthetic.append(dict(case=case,nodes=n,status=info['status'],loss=mp.nstr(loss,40),relative_errors=[mp.nstr(e,30) for e in errors],gate_pass=all(e<=mp.mpf('1e-30') for e in errors)))
        seam=[]
        for n in [1,2]:
            for k in [2,8,16,32]:
                for sign in [-1,1]:
                    nodes=[[mp.mpf('.5')+sign*mp.power(10,-k),mp.mpf(0),mp.mpf(1)]] if n==1 else [[mp.mpf('.2'),mp.mpf(-1),mp.mpf(2)],[mp.mpf('.375')+sign*mp.power(10,-k),mp.mpf(1),mp.mpf(3)]]
                    loss,g,info=mp_extended(nodes);D=mp.log(4)-info['omega'];expected=[[-2*D/(1-a),mp.mpf(0),mp.mpf(0)] for a,p,rho in nodes];error=max(abs(a-b) for aa,bb in zip(g,expected) for a,b in zip(aa,bb));assert abs(loss-D*D)<mp.mpf('1e-120') and error<mp.mpf('1e-120');seam.append(dict(nodes=n,k=k,sign=sign,status=info['status'],gradient_error=mp.nstr(error,20)))
        boundary=mp_extended([[mp.mpf('.5'),mp.mpf(0),mp.mpf(1)]]);assert boundary[2]['status']=='mass_extension' and abs(boundary[1][0][0]+4*mp.log(2))<mp.mpf('1e-200')
        z=np.load(SH/'shared_parameters.npz');inverse=z['node_to_unique'];directions={k:z[k] for k in ['full','geometry']};cursor=0;index=0;real=[]
        missing={arm:np.flatnonzero(~np.isfinite(np.load(SH/f'{arm}_0.01_loss.npy'))).tolist() for arm in directions};assert all(len(v)==1 for v in missing.values())
        for f in sorted(WORLD.glob('scan*_gradients.npz')):
            stem=f.name.removesuffix('_gradients.npz');metadata=read(WORLD/(stem+'.json'))['meta'];assert metadata['kernel']==0
            with np.load(f) as d:raw=d['raw'];source=d['source'];sections=d['sections'];pixels=d['pixels']
            for j,pixel in enumerate(pixels):
                a,b=map(int,sections[j:j+2])
                for arm in directions:
                    if index not in missing[arm]:continue
                    m['source_sha256'][f.relative_to(ROOT).as_posix()]=sha(f);delta=directions[arm][inverse[cursor+a:cursor+b]];nodes=mp_path_nodes(raw[a:b],source[a:b],delta,mp.mpf('.01'));loss,g,info=mp_extended(nodes);assert info['status']=='mass_extension';fl,fg,omega=torch_absent_gradient(raw[a:b,:11]+.01*delta,raw[a:b],source[a:b],metadata);pred=float(np.sum(fg*delta));steps=[]
                    for hh in ['1e-8','1e-10']:
                        h=mp.mpf(hh);lm=mp_extended(mp_path_nodes(raw[a:b],source[a:b],delta,mp.mpf('.01')-h))[0];lp=mp_extended(mp_path_nodes(raw[a:b],source[a:b],delta,mp.mpf('.01')+h))[0];fd=(lp-lm)/(2*h);error=abs(float(fd)-pred)/max(abs(pred),1e-30);steps.append(dict(h=hh,mp_derivative=mp.nstr(fd,40),relative_error=error,gate_pass=error<=1e-5))
                    real.append(dict(arm=arm,index=index,view=stem,pixel=int(pixel),omega=mp.nstr(info['omega'],45),log2_minus_omega=mp.nstr(mp.log(2)-info['omega'],35),extended_loss=mp.nstr(loss,45),float64_loss=fl,loss_error=abs(fl-float(loss)),predicted_raw_directional=pred,steps=steps,gate_pass=abs(fl-float(loss))<=1e-10 and all(st['gate_pass'] for st in steps)))
                index+=1
            cursor+=len(raw)
        assert len(real)==2;summary=dict(status='complete',synthetic=synthetic,seam_cases=seam,exact_single_boundary_gradient=mp.nstr(boundary[1][0][0],40),real_missing_updates=real,gate_pass=all(r['gate_pass'] for r in synthetic+real),scope='C1 extension mathematical/numerical checks; original loss unchanged on finite-root domain, original shared failures remain. No CUDA/training/GT/novelty evidence.');save(OUT/'summary.json',summary)
        for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
        with zipfile.ZipFile(OUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for n in m['source_sha256']:archive.write(ROOT/n,n)
        m.update(status='complete',artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'});save(OUT/'manifest.json',m);print(json.dumps({k:v for k,v in summary.items() if k not in ['synthetic','seam_cases']}),flush=True)
    except BaseException as exc:m.update(status='failed',failure=repr(exc));save(OUT/'manifest.json',m);(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
