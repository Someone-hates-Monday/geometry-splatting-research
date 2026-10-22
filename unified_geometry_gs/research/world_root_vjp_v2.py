"""Continuous source-preserving parameter graph and independent MP VJP."""
import numpy as np
import torch
import mpmath as mp

def torch_rotation(q):
    w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)

def torch_fields(theta,raw,source,meta,arm,_baseline=False):
    base=torch.tensor(raw[:,:11],dtype=torch.float64);src=torch.tensor(source,dtype=torch.float64);radius=torch.tensor(raw[:,11],dtype=torch.float64)
    variance=theta[:,3:6].exp().square();v0=base[:,3:6].exp().square();filtered=variance+radius[:,None].square();f0=v0+radius[:,None].square()
    scales=src[:,3:6]*torch.sqrt(filtered/f0);q=src[:,6:10]+(torch.nn.functional.normalize(theta[:,6:10],dim=1)-torch.nn.functional.normalize(base[:,6:10],dim=1))
    r=torch_rotation(q);mu=theta[:,:3];A=src[0,20:29].reshape(3,3);translation=src[0,29:32];query=src[0,32:35];qn=torch.linalg.vector_norm(query);origin=-torch.linalg.solve(A,translation);direction=torch.linalg.solve(A,query)/qn
    diff=mu-origin
    if arm=='dense':
        cov=r@torch.diag_embed(scales.square())@r.transpose(1,2);precision=torch.linalg.inv(cov);pd=precision@direction;den=(pd*direction).sum(1);peak=(pd*diff).sum(1)/den;res=diff-peak[:,None]*direction;energy=(res*(precision@res[:,:,None]).squeeze(-1)).sum(1);rho=torch.sqrt(den)
    else:
        invr=torch.linalg.inv(r);factor=torch.diag_embed(scales.reciprocal())@invr;u=factor@direction;v=(factor@diff[:,:,None]).squeeze(-1);den=u.square().sum(1);peak=(u*v).sum(1)/den;res=v-peak[:,None]*u;energy=res.square().sum(1);rho=torch.sqrt(den)
    camera=mu@A.T+translation;z=camera[:,2];ux=camera[:,0]/z;vy=camera[:,1]/z;uc=ux.clamp(-meta['limx'],meta['limx']);vc=vy.clamp(-meta['limy'],meta['limy']);zero=torch.zeros_like(z)
    j=torch.stack([meta['fx']/z,zero,-meta['fx']*uc/z,zero,meta['fy']/z,-meta['fy']*vc/z],1).reshape(-1,2,3)
    projected=torch.diag_embed(scales)@r.transpose(1,2)@A.T@j.transpose(1,2)
    c=projected.transpose(1,2)@projected
    if arm=='dense':det=c[:,0,0]*c[:,1,1]-c[:,0,1].square()
    else:det=torch.linalg.cross(projected[:,:,0],projected[:,:,1],dim=1).square().sum(1)
    det1=det+meta['kernel']*(c[:,0,0]+c[:,1,1])+meta['kernel']**2
    gamma=.5*(det.clamp_min(meta['floor']).log()-det1.clamp_min(meta['floor']).log())
    # Gamma0 recomputed at the same captured base arguments; its derivative is
    # stopped. This keeps actual captured AMP and all existing root semantics.
    if not _baseline:
        with torch.no_grad():gamma0=torch_fields(base,raw,source,meta,arm,_baseline=True)[4]
    else:gamma0=gamma.detach()
    logfilter=.5*(variance.log()-filtered.log()).sum(1);logfilter0=.5*(v0.log()-f0.log()).sum(1)
    opacity_ratio=torch.nn.functional.logsigmoid(theta[:,10])-torch.nn.functional.logsigmoid(base[:,10]);amplitude=src[:,35]*torch.exp(opacity_ratio+logfilter-logfilter0+gamma-gamma0)
    return peak,rho,energy,amplitude,gamma,qn

def torch_residual(fields,t):
    p,rho,E,A=fields[:4];cap=float(np.float32(.99));positive=A>0
    la=torch.full_like(A,-torch.inf);la[positive]=A[positive].log()-.5*E[positive];capped=la>np.log(cap)
    base=torch.where(capped,torch.full_like(A,cap),A);en=torch.where(capped,torch.zeros_like(E),E);after=t>p;active_after=after&positive;safe=active_after&(base<1)
    beta=torch.prod(1-base[safe]);shift=-(torch.log1p(base[safe]/(1-base[safe])*(-torch.expm1(-.5*en[safe])))).sum()
    other=active_after&~safe
    if other.any():beta=beta*torch.prod(1-torch.exp(la[other]))
    constant=torch.log1p((.5-beta)/beta)+shift
    lg=la[positive].clamp_max(np.log(cap))-.5*((t-p[positive])*rho[positive]).square();far=lg<-40;lt=torch.empty_like(lg);lt[far]=lg[far]-np.log(2);lt[~far]=torch.log(-.5*torch.log1p(-torch.exp(lg[~far])))
    scale=max(float(lt.detach().max()) if len(lt) else -np.inf,np.log(abs(float(constant.detach()))) if float(constant.detach()) else -np.inf)
    if not np.isfinite(scale) or scale<-700 or scale>700:raise FloatingPointError('Scaled residual outside F64 diagnostic representability')
    sign=torch.where(after[positive],-torch.ones_like(lt),torch.ones_like(lt));return constant*np.exp(-scale)+(sign*torch.exp(lt-scale)).sum(),scale

def polish(fields,lo,hi):
    # Independent of MP reference. All field tensors are constants at theta0.
    lo=float(lo);hi=float(hi)
    vl,_=torch_residual(fields,torch.tensor(lo,dtype=torch.float64));vh,_=torch_residual(fields,torch.tensor(hi,dtype=torch.float64))
    if not float(vl)<0<float(vh):return None,'unbracketed',0
    for count in range(200):
        mid=.5*lo+.5*hi
        if mid==lo or mid==hi:return mid,'machine_bracket',count
        value,_=torch_residual(fields,torch.tensor(mid,dtype=torch.float64));value=float(value)
        if value<0:lo=mid
        elif value>0:hi=mid
        else:return mid,'rounded_zero_candidate',count+1
    return .5*lo+.5*hi,'iteration_limit',200

def torch_vjp(raw,source,meta,t,arm):
    theta=torch.tensor(raw[:,:11],dtype=torch.float64,requires_grad=True);query_t=torch.tensor(float(t),dtype=torch.float64,requires_grad=True);fields=torch_fields(theta,raw,source,meta,arm);value,scale=torch_residual(fields,query_t);g,gt=torch.autograd.grad(value,(theta,query_t));qn=float(fields[5]);slope=float(gt)*np.exp(scale)
    if not float(gt)>0 or not torch.isfinite(g).all() or not torch.isfinite(gt):raise FloatingPointError('Undefined/nonfinite IFT VJP')
    return (-g/gt/qn).detach().numpy(),dict(scaled_slope=float(gt),log_H_slope=np.log(float(gt))+scale,H_slope=slope,scale=scale,residual_scaled=float(value.detach()))

def mp_rotation_derivatives(q):
    w,x,y,z=q
    return [mp.matrix([[0,-2*z,2*y],[2*z,0,-2*x],[-2*y,2*x,0]]),mp.matrix([[0,2*y,2*z],[2*y,-4*x,-2*w],[2*z,2*w,-4*x]]),mp.matrix([[-4*y,2*x,2*w],[2*x,0,2*z],[-2*w,2*z,-4*y]]),mp.matrix([[-4*z,-2*w,2*x],[2*w,-4*z,2*y],[2*x,2*y,0]])]

def inner(a,b):return mp.fsum(a[i,j]*b[i,j] for i in range(a.rows) for j in range(a.cols))

def mp_geometry(row):
    from research.exact_outlier_world_reference import matrix,vector,rotation,dot
    A=matrix(row[20:29].reshape(3,3));Ai=A**-1;origin=-Ai*vector(row[29:32]);query=vector(row[32:35]);qn=mp.sqrt(dot(query,query));d=Ai*query/qn;mu=vector(row[:3]);v=mu-origin;R=rotation([mp.mpf(float(x)) for x in row[6:10]]);S=mp.diag([mp.mpf(float(x)) for x in row[3:6]]);cov=R*S*S*R.T;P=cov**-1;a=dot(d,P*d);p=dot(d,P*v)/a;w=v-p*d;E=dot(w,P*w);amp=mp.mpf(float(row[35]));alpha=min(mp.mpf(float(np.float32(.99))),amp*mp.exp(-E/2))
    from fractions import Fraction
    exact_cam=[sum((Fraction(float(row[20+3*i+j]))*Fraction(float(row[j])) for j in range(3)),Fraction(float(row[29+i]))) for i in range(3)];qr=[Fraction(float(x)) for x in row[32:35]]
    if all(exact_cam[i]*qr[2]==exact_cam[2]*qr[i] for i in [0,1]):
        ratio=exact_cam[2]/qr[2];p=mp.mpf(ratio.numerator)/ratio.denominator*qn;w=mp.zeros(3,1);E=mp.mpf(0);alpha=min(mp.mpf(float(np.float32(.99))),amp)
    return dict(A=A,d=d,mu=mu,v=v,R=R,S=S,cov=cov,P=P,a=a,p=p,w=w,E=E,rho=mp.sqrt(a),alpha=alpha,amp=amp,qn=qn)

def mp_screen_derivatives(g,row,meta):
    from research.exact_outlier_world_reference import vector,dot
    A=g['A'];cam=A*g['mu']+vector(row[29:32]);z=cam[2];ux=cam[0]/z;vy=cam[1]/z;lx=mp.mpf(meta['limx']);ly=mp.mpf(meta['limy']);uc=min(lx,max(-lx,ux));vc=min(ly,max(-ly,vy));fx=mp.mpf(meta['fx']);fy=mp.mpf(meta['fy']);k=mp.mpf(meta['kernel']);floor=mp.mpf(meta['floor']);J=mp.matrix([[fx/z,0,-fx*uc/z],[0,fy/z,-fy*vc/z]]);C=J*A*g['cov']*A.T*J.T;D0=mp.det(C);C1=C+k*mp.eye(2);D1=mp.det(C1)
    if D0==floor or D1==floor or abs(ux)==lx or abs(vy)==ly:raise ValueError('Exact screen clamp boundary')
    B=mp.zeros(2)
    if D0>floor:B+=C**-1/2
    if D1>floor:B-=C1**-1/2
    U=A.T*J.T*B*J*A;GJ=2*B*J*A*g['cov']*A.T;gu=-fx*GJ[0,2]/z;gv=-fy*GJ[1,2]/z;gz=-inner(GJ,J)/z;camera_grad=mp.matrix([gu/z if abs(ux)<lx else 0,gv/z if abs(vy)<ly else 0,gz-(gu*ux/z if abs(ux)<lx else 0)-(gv*vy/z if abs(vy)<ly else 0)]);gm=A.T*camera_grad
    scale=[]
    for j in range(3):rj=g['R'][:,j];scale.append(2*g['S'][j,j]**2*dot(rj,U*rj))
    GR=2*U*g['R']*g['S']*g['S'];dq=[inner(GR,v) for v in mp_rotation_derivatives([mp.mpf(float(x)) for x in row[6:10]])]
    return gm,scale,dq,dict(det0=mp.nstr(D0,30),det1=mp.nstr(D1,30),floor0=bool(D0<floor),floor1=bool(D1<floor))

def mp_vjp(nodes,source,raw,meta,t):
    from research.exact_outlier_world_reference import dot,vector
    result=[];slope=mp.mpf(0);screen=[];cap=mp.mpf(float(np.float32(.99)))
    for g,row,r in zip(nodes,source,raw):
        if g['amp']==0:result.append([mp.mpf(0)]*11);screen.append({'zero_amp':True});continue
        x=(t-g['p'])*g['rho'];ag=g['alpha']*mp.exp(-x*x/2);tail=ag/(2*(1-ag));after=t>g['p'];signed=-tail if after else tail;hlog=(g['alpha']/(1-g['alpha']) if after else 0)+signed
        uncapped=g['amp']*mp.exp(-g['E']/2)
        if uncapped==cap:raise ValueError('Exact effective-alpha cap boundary')
        he=-hlog/2 if uncapped<cap else mp.mpf(0);ha=hlog if uncapped<cap else mp.mpf(0);hp=signed*x*g['rho'];hr=-signed*x*x/g['rho'];P=g['P'];d=g['d'];w=g['w'];a=g['a'];mu_grad=hp*(P*d)/a+2*he*(P*w);HP=hp*(d*w.T+w*d.T)/(2*a)+hr*(d*d.T)/(2*g['rho'])+he*(w*w.T);Ri=g['R']**-1;F=g['S']**-1*Ri;ss=[-2*dot(F[j,:].T,HP*F[j,:].T) for j in range(3)];GR=-2*P*HP*Ri.T;qq=[inner(GR,v) for v in mp_rotation_derivatives([mp.mpf(float(x)) for x in row[6:10]])]
        gm,gs,gq,diagnostic=mp_screen_derivatives(g,row,meta);screen.append(diagnostic);mu_grad+=ha*gm
        radius=mp.mpf(float(r[11]));kv=[mp.exp(2*mp.mpf(float(r[3+j])))/(mp.exp(2*mp.mpf(float(r[3+j])))+radius*radius) for j in range(3)];scale=[ss[j]*kv[j]+ha*(1-kv[j]+gs[j]*kv[j]) for j in range(3)];qraw=vector(r[6:10]);norm=mp.sqrt(dot(qraw,qraw));J=(mp.eye(4)-qraw*qraw.T/(norm*norm))/norm;quat=J.T*mp.matrix([qq[j]+ha*gq[j] for j in range(4)]);opacity=ha/(1+mp.exp(mp.mpf(float(r[10]))));result.append(list(mu_grad)+scale+list(quat)+[opacity]);slope+=ag*abs(x)*g['rho']/(2*(1-ag))
    if slope<=0:raise ValueError('Singular IFT denominator')
    return [[-v/(slope*nodes[0]['qn']) for v in row] for row in result],slope,screen
