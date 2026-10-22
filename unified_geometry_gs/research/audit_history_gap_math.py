"""Predeclared mathematical checks; no training or quality claims."""
from pathlib import Path
import argparse, hashlib, json, os, time, traceback, zipfile
import numpy as np
import mpmath as mp
import sympy as sp
from research.history_gap_math import mp_root,mp_residual,mp_loss_gradient,float_loss_gradient
from research.world_root_vjp import mp_geometry

R=Path(__file__).resolve().parent; ROOT=R.parent
BASE=R/'experiments/world_root_vjp_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def direct_loss(nodes,t):
    # Independent expression, using V rather than the after-peak history sum.
    ell=mp.fsum(mp.log(1-a*mp.exp(-((t-p)*rho)**2/2))/2 for a,p,rho in nodes)
    return (ell+mp.log(2))**2
def err(a,b):return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-300))
def metrics(a,b):
    if not np.isfinite(a).all():return dict(relative_l2=None,blocks=None,gate_pass=False,nonfinite=True)
    blocks={k:err(a[:,i],b[:,i]) for i,k in enumerate(['alpha','peak','rho'])}
    relative=err(a,b)
    return dict(relative_l2=relative,blocks=blocks,gate_pass=bool(np.isfinite(a).all() and relative<=1e-5 and max(blocks.values())<=1e-4))
def symbolic():
    a,p,rho,t,r=sp.symbols('a p rho t r',real=True);x=(t-p)*rho
    u=a*sp.exp(-x*x/2);ell=sp.log(1-u)/2;b=u/(2*(1-u));checks={}
    for after in [False,True]:
        H=-sp.log(1-a)+ell if after else -ell;s=-1 if after else 1;c=b*(1+r*s)
        expected=[-c/a-r*int(after)/(1-a),-c*x*rho,c*x*x/rho]
        for var,e in zip([a,p,rho],expected):checks[f'{after}_{var}']=sp.simplify(sp.diff(ell,var)-r*sp.diff(H,var)-e)==0
    checks['ell_t']=sp.simplify(sp.diff(ell,t)-b*x*rho)==0
    checks['single_hinge_second_below']=sp.limit(sp.diff((sp.log(4)+sp.log(1-a))**2,a,2),a,sp.Rational(3,4),dir='-')==32
    assert all(checks.values());return dict(checks=checks,sympy_version=sp.__version__)
def finite_difference(nodes,h,root):
    result=[]
    for i in range(len(nodes)):
        row=[]
        for k in range(3):
            plus=[list(v) for v in nodes];minus=[list(v) for v in nodes];plus[i][k]+=h;minus[i][k]-=h
            tp=mp_root(plus,root);tm=mp_root(minus,root)
            row.append((direct_loss(plus,tp)-direct_loss(minus,tm))/(2*h))
        result.append(row)
    return result
def mp_relative(a,b):
    return mp.sqrt(mp.fsum((x-y)**2 for aa,bb in zip(a,b) for x,y in zip(aa,bb)))/max(mp.sqrt(mp.fsum(x*x for aa in b for x in aa)),mp.mpf('1e-400'))
def synthetic(out):
    mp.mp.dps=240;rng=np.random.default_rng(13017);rows=[]
    for case in range(16):
        n=[1,2,4,8][case%4]
        av=rng.uniform(.55,.9,n);pv=rng.uniform(-1,1,n);rv=rng.uniform(.5,3,n)
        nodes=[[mp.mpf(float(a)),mp.mpf(float(p)),mp.mpf(float(rho))] for a,p,rho in zip(av,pv,rv)]
        t=mp_root(nodes);loss,gradient,info=mp_loss_gradient(nodes,t);errors=[]
        assert abs(info['D']-info['ell']-mp.log(2))<mp.mpf('1e-180')
        for h in ['1e-30','1e-35']:errors.append(mp_relative(gradient,finite_difference(nodes,mp.mpf(h),t)))
        passed=all(e<=mp.mpf('1e-30') for e in errors)
        rows.append(dict(case=case,nodes=n,root=mp.nstr(t,45),D=mp.nstr(info['D'],45),relative_fd=[mp.nstr(e,25) for e in errors],gate_pass=passed))
        print(f'synthetic {case+1}/16 pass={passed}',flush=True)
    critical=[]
    for sign in [-1,1]:
        for k in [2,4,8,16,32]:
            a=mp.mpf('.75')+sign*mp.power(10,-k);nodes=[[a,mp.mpf(0),mp.mpf(1)]];t=mp_root(nodes);loss,gradient,info=mp_loss_gradient(nodes,t)
            u=a*mp.exp(-t*t/2);Halpha= (1/(1-a)-u/(2*a*(1-u))) if t>0 else u/(2*a*(1-u))
            root_jac=-Halpha/info['Ht'];exact=max(mp.log(4)+mp.log1p(-a),mp.mpf(0))
            assert abs(info['D']-exact)<mp.mpf('1e-180')
            critical.append(dict(side=sign,k=k,loss=mp.nstr(loss,30),loss_alpha_gradient=mp.nstr(gradient[0][0],30),root_alpha_gradient=mp.nstr(root_jac,30)))
    common=[]
    for alpha in [['.6'],['.75'],['.9'],['.5','.5'],['.4','.4'],['.8','.8']]:
        nodes=[[mp.mpf(a),mp.mpf('.3'),mp.mpf(i+1)] for i,a in enumerate(alpha)];t=mp_root(nodes);loss,gradient,info=mp_loss_gradient(nodes,t);omega=-mp.fsum(mp.log1p(-a) for a,p,rho in nodes);expected=max(mp.log(4)-omega,mp.mpf(0))
        assert abs(info['D']-expected)<mp.mpf('1e-160')
        common.append(dict(alpha=alpha,D=mp.nstr(info['D'],35),expected_D=mp.nstr(expected,35),singular=info['singular']))
    mp.mp.dps=420;adverse=[]
    for eps in ['.2','.1','.075','.05']:
        rho=1/mp.mpf(eps);nodes=[[mp.mpf('.5'),mp.mpf(-1),rho],[mp.mpf('.5'),mp.mpf(1),rho]];t=mp_root(nodes);loss,gradient,info=mp_loss_gradient(nodes,t);g=mp.exp(-rho*rho/2)
        assert t==0 and info['r']==0 and abs(info['D']-mp.log(2-g))<mp.mpf('1e-350')
        exponent=int(mp.ceil(-mp.log10(g)))+60;h=mp.power(10,-exponent);errors=[mp_relative(gradient,finite_difference(nodes,h*factor,t)) for factor in [1,mp.mpf('.01')]]
        adverse.append(dict(epsilon=eps,loss=mp.nstr(loss,40),D=mp.nstr(info['D'],40),gradient_norm=mp.nstr(mp.sqrt(mp.fsum(v*v for row in gradient for v in row)),40),h=mp.nstr(h,8),relative_fd=[mp.nstr(v,30) for v in errors],gate_pass=all(e<mp.mpf('1e-30') for e in errors)))
        print(f'adverse eps={eps} relative_fd={mp.nstr(max(errors),8)}',flush=True)
    result=dict(status='complete',regular=rows,critical=critical,common_peak=common,adverse=adverse,gate_pass=all(v['gate_pass'] for v in rows+adverse),precision_regular=240,precision_adverse=420)
    save(out/'synthetic.json',result);return result
def real(out,manifest):
    oldm=read(BASE/'manifest.json');rows=[];nodes_count=0
    for f in sorted(BASE.glob('scan*_gradients.npz')):
        assert sha(f)==oldm['artifacts_sha256'][f.name]
        meta=BASE/(f.name.removesuffix('_gradients.npz')+'.json');assert sha(meta)==oldm['artifacts_sha256'][meta.name]
        manifest['source_sha256'][f.relative_to(ROOT).as_posix()]=sha(f);manifest['source_sha256'][meta.relative_to(ROOT).as_posix()]=sha(meta);save(out/'manifest.json',manifest)
        with np.load(f) as d:source=d['source'];sections=d['sections'];pixels=d['pixels']
        oldrows=read(meta)['rows'];coef=np.empty((len(source),3));reference=np.empty_like(coef);actual=np.empty_like(coef);public=np.empty_like(coef);logalpha=np.empty(len(source));viewrows=[]
        for j,previous in enumerate(oldrows):
            begin,end=map(int,sections[j:j+2]);assert int(pixels[j])==previous['pixel'];mp.mp.dps=140
            geometry=[mp_geometry(row) for row in source[begin:end]];mp.mp.dps=240
            nodes=[[v['alpha'],v['p'],v['rho']] for v in geometry];t=mp_root(nodes,mp.mpf(previous['reference_root']));loss,gradient,info=mp_loss_gradient(nodes,t)
            identity=abs(info['D']-info['ell']-mp.log(2));assert identity<mp.mpf('1e-120')
            c=np.array([[float(v) for v in row] for row in nodes]);la=np.array([float(mp.log(v[0])) for v in nodes]);ref=np.array([[float(v) for v in row] for row in gradient]);fl,fg,fi=float_loss_gradient(c,float(t),la);center=previous['branches']['factor_public']['t'];pl,pg,pi=float_loss_gradient(c,center,la)
            coef[begin:end]=c;reference[begin:end]=ref;actual[begin:end]=fg;public[begin:end]=pg;logalpha[begin:end]=la
            row=dict(view=previous['view'],pixel=int(pixels[j]),nodes=end-begin,rounded_alpha_zero=int(np.count_nonzero(c[:,0]==0)),old_anomaly=previous['old_anomaly'],root=mp.nstr(t,55),D=float(info['D']),loss=float(loss),ratio=float(info['r']),identity_error=mp.nstr(identity,15),float64=metrics(fg,ref),public_center=metrics(pg,ref),float64_loss_error=abs(fl-float(loss)),public_loss_error=abs(pl-float(loss)))
            rows.append(row);viewrows.append(row);nodes_count+=end-begin
            if (j+1)%8==0 or j+1==len(oldrows):
                save(out/'progress.json',dict(phase='real',rays=len(rows),nodes=nodes_count,view=previous['view'],pid=os.getpid()));print(f"real {len(rows)}/891, nodes {nodes_count}, current D {row['D']:.8g}",flush=True)
        np.savez_compressed(out/f.name,pixels=pixels,sections=sections,coefficients=coef,log_alpha=logalpha,reference=reference,float64=actual,public_center=public)
        save(out/meta.name,dict(rows=viewrows))
    assert len(rows)==891 and nodes_count==51524 and sum(r['old_anomaly'] is not None for r in rows)==9
    result=dict(status='complete',rays=len(rows),nodes=nodes_count,rows=rows,all9_old_anomalies_present=True,pass_counts={k:sum(r[k]['gate_pass'] for r in rows) for k in ['float64','public_center']},max_relative_l2={k:max(r[k]['relative_l2'] for r in rows) for k in ['float64','public_center']},D_range=[min(r['D'] for r in rows),max(r['D'] for r in rows)],gate_pass=all(r['float64']['gate_pass'] for r in rows))
    save(out/'real.json',result);return result
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();out=args.out.resolve();assert not out.exists();out.mkdir(parents=True)
    sources=[Path(__file__),R/'history_gap_math.py',R/'HISTORY_GAP_MATH_PROTOCOL.md',R/'HISTORY_GAP_MATH_AMENDMENT_V2.md',R/'HISTORY_GAP_MATH_AMENDMENT_V3.md',R/'world_root_vjp.py',BASE/'manifest.json'];manifest=dict(status='running',created_epoch=time.time(),pid=os.getpid(),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources});save(out/'manifest.json',manifest)
    try:
        save(out/'symbolic.json',symbolic());s=synthetic(out);r=real(out,manifest)
        for name,h in manifest['source_sha256'].items():assert sha(ROOT/name)==h
        with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name in manifest['source_sha256']:z.write(ROOT/name,name)
        result=dict(status='complete',synthetic_gate=s['gate_pass'],real_gate=r['gate_pass'],scope='coefficient-space continuous layer model only; neither CUDA/full-raw chain nor training/GT benefit',limitations=['Separated thin surfaces have large loss but exponentially vanishing gradients.','Bounded ratio alone is not global boundedness at alpha=1 or rho=infinity.','C1 coefficient result excludes cap, visibility and IEEE input program transitions.']);save(out/'summary.json',result)
        manifest.update(status='complete',completed_epoch=time.time(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(out/'manifest.json',manifest);print(json.dumps(result),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',failure=repr(exc));save(out/'manifest.json',manifest);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
