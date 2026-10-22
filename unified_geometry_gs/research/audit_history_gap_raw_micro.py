"""High precision composed forward, independent of analytic adjoint."""
from pathlib import Path
import hashlib,json,os,time,traceback,zipfile
import numpy as np
import mpmath as mp
from research.history_gap_math import mp_root
from research.history_gap_raw import parameter_scale,torch_forward
from research.exact_outlier_world_reference import rotation,dot
R=Path(__file__).resolve().parent;ROOT=R.parent;BASE=R/'experiments/history_gap_raw_v1';WORLD=R/'experiments/world_root_vjp_v1';OUT=R/'experiments/history_gap_raw_micro_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
class MPForward:
    def __init__(self,raw,source):
        self.raw=[[mp.mpf(float(v)) for v in row] for row in raw];self.src=[[mp.mpf(float(v)) for v in row] for row in source]
        s=self.src[0];A=mp.matrix([s[20+3*i:23+3*i] for i in range(3)]);Ai=A**-1;q=mp.matrix(s[32:35]);self.origin=-Ai*mp.matrix(s[29:32]);self.direction=Ai*q/mp.sqrt(dot(q,q));self.cap=mp.mpf(float(np.float32(.99)))
    def evaluate(self,direction,h,root):
        nodes=[];caps=[]
        for raw,src,d in zip(self.raw,self.src,direction):
            theta=[v+h*mp.mpf(float(delta)) for v,delta in zip(raw,d)];mu=mp.matrix(theta[:3]);f2=raw[11]**2;v0=[mp.exp(2*v) for v in raw[3:6]];v=[mp.exp(2*a) for a in theta[3:6]];scales=[src[3+j]*mp.sqrt((v[j]+f2)/(v0[j]+f2)) for j in range(3)]
            q0=raw[6:10];q1=theta[6:10];n0=mp.sqrt(mp.fsum(a*a for a in q0));n1=mp.sqrt(mp.fsum(a*a for a in q1));q=[src[6+j]+q1[j]/n1-q0[j]/n0 for j in range(4)];rot=rotation(q);cov=rot*mp.diag([a*a for a in scales])*rot.T;P=cov**-1;diff=mu-self.origin;den=dot(self.direction,P*self.direction);peak=dot(self.direction,P*diff)/den;w=diff-peak*self.direction;energy=dot(w,P*w)
            sigmoid_ratio=(1+mp.exp(-raw[10]))/(1+mp.exp(-theta[10]));filter_ratio=mp.sqrt(mp.fprod(v[j]/(v[j]+f2)*(v0[j]+f2)/v0[j] for j in range(3)));amp=src[35]*sigmoid_ratio*filter_ratio;uncapped=amp*mp.exp(-energy/2);caps.append(uncapped>self.cap);nodes.append([min(uncapped,self.cap),peak,mp.sqrt(den)])
        t=mp_root(nodes,root);ell=mp.fsum(mp.log1p(-a*mp.exp(-((t-p)*rho)**2/2))/2 for a,p,rho in nodes)
        return (ell+mp.log(2))**2,t,caps
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);files=[Path(__file__),R/'HISTORY_GAP_RAW_MICRO_PROTOCOL.md',R/'history_gap_raw.py',R/'history_gap_math.py',BASE/'manifest.json',BASE/'summary.json'];m=dict(status='running',pid=os.getpid(),created_epoch=time.time(),source_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files});save(OUT/'manifest.json',m)
    try:
        rng=np.random.default_rng(13017);rows=[];directions_archive={};mp.mp.dps=240;summary=read(BASE/'summary.json');assert summary['directional_pass']==35 and summary['direction_count']==42
        for f in sorted(BASE.glob('scan*_gradients.npz')):
            stem=f.name.removesuffix('_gradients.npz');wf=WORLD/f.name;meta=WORLD/(stem+'.json');rf=BASE/(stem+'.json')
            for p in [f,wf,meta,rf]:m['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
            with np.load(f) as d:grad=d['factor'];ref=d['reference'];sections=d['sections']
            with np.load(wf) as d:raw=d['raw'];source=d['source']
            metadata=read(meta)['meta'];assert metadata['kernel']==0
            for j,row in enumerate(read(rf)['rows']):
                if 'directional' not in row:continue
                a,b=map(int,sections[j:j+2]);params=raw[a:b];src=source[a:b];g=grad[a:b];scale=parameter_scale(params,src);xi=rng.normal(size=g.shape);random=scale*xi/np.linalg.norm(xi);norm=np.linalg.norm(scale*g);directions={'random':random,'negative_gradient':-scale*scale*g/norm}
                for original in row['directional']:
                    direction=directions[original['direction']];pred=float(np.sum(direction*g));assert abs(pred-original['predicted'])<=1e-12*max(1,abs(pred))
                    if original['gate_pass']:continue
                    ctx=MPForward(params,src);root=mp.mpf(row['reference_root']);baseline,base_t,base_caps=ctx.evaluate(direction,mp.mpf(0),root);reference=mp.fsum(mp.mpf(float(v))*mp.mpf(float(d)) for rr,dd in zip(ref[a:b],direction) for v,d in zip(rr,dd));steps=[]
                    assert abs(baseline-mp.mpf(row['reference_loss']))<mp.mpf('1e-15')
                    for hh in ['1e-6','1e-8','1e-10','1e-12']:
                        h=mp.mpf(hh);minus,tm,cm=ctx.evaluate(direction,-h,base_t);plus,tp,cp=ctx.evaluate(direction,h,base_t);fd=(plus-minus)/(2*h);error=abs(fd-reference);fpminus,mi=torch_forward(params[:,:11]-float(h)*direction,params,src,metadata);fpplus,pi=torch_forward(params[:,:11]+float(h)*direction,params,src,metadata);ffd=(fpplus-fpminus)/(2*float(h)) if fpminus is not None and fpplus is not None else None
                        steps.append(dict(h=hh,mp_derivative=mp.nstr(fd,40),mp_absolute_error=mp.nstr(error,25),mp_relative_error=mp.nstr(error/max(abs(reference),mp.mpf('1e-300')),25),float64_derivative=ffd,float64_relative_error=abs(ffd-float(reference))/max(abs(float(reference)),1e-300) if ffd is not None else None,cap_changed=cm!=base_caps or cp!=base_caps))
                    last=steps[-1];passed=mp.mpf(last['mp_absolute_error'])<=max(mp.mpf('1e-9'),mp.mpf('1e-5')*abs(reference));key=f'{row["index"]}_{original["direction"]}';directions_archive[key]=direction
                    rows.append(dict(view=stem,pixel=row['pixel'],index=row['index'],direction=original['direction'],original_gate_pass=False,predicted_factor=pred,predicted_reference=mp.nstr(reference,45),steps=steps,micro_mp_gate_pass=passed));save(OUT/'progress.json',dict(pairs=len(rows),active=key));print(f'micro {len(rows)}/7 {stem} {original["direction"]}: MP pass={passed}',flush=True)
        assert len(rows)==7;save(OUT/'summary.json',dict(status='complete',original_direction_pass=35,original_direction_total=42,diagnostic_pairs=7,mp_micro_pass=sum(r['micro_mp_gate_pass'] for r in rows),rows=rows,scope='Follow-up diagnosis of original failures; original gate unchanged, no training/GT evidence.'));np.savez_compressed(OUT/'directions.npz',**directions_archive)
        for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
        with zipfile.ZipFile(OUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
            for n in m['source_sha256']:z.write(ROOT/n,n)
        m.update(status='complete',artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(OUT/'manifest.json',m)
    except BaseException as exc:m.update(status='failed',failure=repr(exc));save(OUT/'manifest.json',m);(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
