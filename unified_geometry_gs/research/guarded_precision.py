"""Conditional world-coefficient enclosures and a source-preserving double solver."""
from fractions import Fraction
import bisect,math
import numpy as np
import mpmath as mp
from research.coherent_factor import construct,store,TRI

HALF=Fraction(1,2)


def bounds(v):return [mp.mpf(x) for x in v._mpi_]
def point(v):return mp.iv.mpf(float(v))
def box(lo,hi):return mp.iv.mpf([lo,hi])
def rational(v):return mp.iv.mpf(v.numerator)/mp.iv.mpf(v.denominator)
def exact_mpf(v):
    sign,mantissa,exponent,_=v._mpf_
    return Fraction((-1 if sign else 1)*int(mantissa))*Fraction(2)**int(exponent)
def dot(x,y):return sum((a*b for a,b in zip(x,y)),mp.iv.mpf(0))
def norm(x):return mp.iv.sqrt(sum((v**2 for v in x),mp.iv.mpf(0)))
def transpose(A):return [list(v) for v in zip(*A)]
def matmul(A,B):return [[dot(r,c) for c in transpose(B)] for r in A]
def mv(A,x):return [dot(r,x) for r in A]
def norm_upper(A):return bounds(norm([max(abs(a),abs(b)) for r in A for v in r for a,b in [bounds(v)]]))[1]


def inv3(A):
    a,b,c=A[0];d,e,f=A[1];g,h,i=A[2]
    det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    lo,hi=bounds(det)
    if lo<=0<=hi:raise ValueError('Unresolved matrix inverse')
    return [[v/det for v in row] for row in [[e*i-f*h,c*h-b*i,b*f-c*e],[f*g-d*i,a*i-c*g,c*d-a*f],[d*h-e*g,b*g-a*h,a*e-b*d]]]


def rotation_iv(q):
    w,x,y,z=map(point,q)
    R=[[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]
    k=w*w+x*x+y*y+z*z-1;v=[x,y,z];D=1+4*k*dot(v,v)
    if bounds(D)[0]<=0:raise ValueError('Unresolved quaternion inverse')
    return [[(R[j][i]+4*k*v[i]*v[j])/D for j in range(3)] for i in range(3)]


def camera_exact(item):
    A=np.asarray(item['view'])[:3,:3].T;t=np.asarray(item['view'])[3,:3]
    return [sum((Fraction(float(A[i,j]))*Fraction(float(item['mean'][j])) for j in range(3)),Fraction(float(t[i]))) for i in range(3)]


def upward32(v):
    f=np.float32(float(v))
    if mp.mpf(float(f))<v:f=np.nextafter(f,np.float32(np.inf))
    return float(f)


def log1p_iv(x):
    lo,hi=bounds(x);r=max(abs(lo),abs(hi))
    if r<mp.mpf('.125'):
        value=sum(((-1)**(k+1)*x**k/k for k in range(1,13)),mp.iv.mpf(0))
        radius=bounds(point(0)+box(r,r)**13/(13*(1-box(r,r))))[1]
        return value+box(-radius,radius)
    return mp.iv.log(1+x)


def expm1_iv(x):
    lo,hi=bounds(x);r=max(abs(lo),abs(hi))
    if r<mp.mpf('.125'):
        value=sum((x**k/math.factorial(k) for k in range(1,13)),mp.iv.mpf(0))
        radius=bounds(mp.iv.exp(box(r,r))*box(r,r)**13/math.factorial(13))[1]
        return value+box(-radius,radius)
    return mp.iv.exp(x)-1


def enclosure(item,high=False):
    mp.iv.prec=53
    U64,mu,F=construct(item['mean'],item['scales'],item['R'],item['view'],item['quat'])
    Q,U0=np.linalg.qr(F);signs=np.where(np.diag(U0)<0,-1.,1.);Q=Q*signs[None,:]
    cache,mean=store(U64,mu,'qr64' if high else 'qr_mixed');U=np.zeros((3,3));U[TRI]=cache
    A=[[point(v) for v in row] for row in np.asarray(item['view'])[:3,:3].T]
    Ai=inv3(A);Ri=rotation_iv(item['quat']) if item['quat'] is not None else inv3([[point(v) for v in r] for r in item['R']])
    M=matmul(Ri,Ai);M=[[v/point(s) for v in row] for row,s in zip(M,item['scales'])]
    Qv=[[point(v) for v in r] for r in Q];Uv=[[point(v) for v in r] for r in U]
    gram=matmul(transpose(Qv),Qv);defect=[[v-(1 if i==j else 0) for j,v in enumerate(row)] for i,row in enumerate(gram)]
    delta=norm_upper(defect)
    if delta>=1:raise ValueError('Polar factor budget unresolved')
    residual=[[a-b for a,b in zip(x,y)] for x,y in zip(M,matmul(Qv,Uv))]
    polar=box(delta,delta)/(1+mp.iv.sqrt(1-box(delta,delta)))
    eta=bounds(box(norm_upper(residual),norm_upper(residual))+polar*norm([v for r in Uv for v in r]))[1]
    eta=mp.mpf(upward32(eta));true_mu=camera_exact(item)
    muiv=[rational(v) for v in true_mu]
    mu_error=norm([v-point(m) for v,m in zip(muiv,mean)])
    eta_mu=mp.mpf(upward32(bounds(mu_error)[1]))
    q=[point(v) for v in item['qray']];qnorm=norm(q);d=[v/qnorm for v in q]
    qf=[Fraction(float(t)) for t in item['qray']]
    mf=[Fraction(float(t)) for t in mean]
    offset=[rational(mf[i]-qf[i]/qf[2]*mf[2]) for i in (0,1)]+[point(0)]
    u=mv(Uv,d);w=mv(Uv,offset);a=dot(u,u);correction=dot(u,w)/a
    p=rational(mf[2]/qf[2])*qnorm+correction;rho=mp.iv.sqrt(a)
    r=[ww-correction*uu for ww,uu in zip(w,u)];e=sum((vv**2 for vv in r),mp.iv.mpf(0))
    x=[cc-correction*dd for cc,dd in zip(offset,d)]
    rho_lo,rho_hi=bounds(rho)
    if rho_lo<=eta:raise ValueError('Ray denominator budget unresolved')
    e_hi=bounds(e)[1];unorm_hi=bounds(norm([t for row in Uv for t in row]))[1]
    rho_max=box(rho_hi,rho_hi)+box(eta,eta);rho_min=box(rho_lo,rho_lo)-box(eta,eta)
    zeta=box(eta,eta)*norm(x)+(box(unorm_hi,unorm_hi)+box(eta,eta))*box(eta_mu,eta_mu)
    dp=(box(eta,eta)*mp.iv.sqrt(box(e_hi,e_hi))+rho_max*zeta)/rho_min**2
    dp_hi=bounds(dp)[1];dr=zeta+box(dp_hi,dp_hi)*rho_max
    de=2*mp.iv.sqrt(box(e_hi,e_hi))*dr+dr**2;de_hi=bounds(de)[1]
    p=p+box(-dp_hi,dp_hi);rho=rho+box(-eta,eta)
    elo,ehi=bounds(e+box(-de_hi,de_hi));energy=box(max(mp.mpf(0),elo),ehi)
    aligned=all(true_mu[i]*qf[2]==true_mu[2]*qf[i] for i in (0,1))
    if aligned:
        p=rational(true_mu[2]/qf[2])*qnorm;energy=point(0)
    amp=point(item['amplitude']);raw=amp*mp.iv.exp(-energy/2);al,ah=bounds(raw);cap=mp.mpf(float(np.float32(.99)))
    alpha=box(min(cap,al),min(cap,ah))
    return {'p':p,'rho':rho,'alpha':alpha,'energy':energy,'amp':amp,'eta_U':float(eta),'eta_mean':float(eta_mu),'aligned':aligned}


def prefix_constant(nodes):
    beta0=Fraction(1);variable=[]
    for c in nodes:
        al,ah=bounds(c['alpha']);A=float(bounds(c['amp'])[0]);cap=float(np.float32(.99))
        if al==ah:
            beta0*=1-exact_mpf(al)
        elif ah<mp.mpf(cap) and 0<=A<1:
            beta0*=1-Fraction(A)
            h=(point(A)/(1-point(A)))*(-expm1_iv(-c['energy']/2))
            variable.append(-log1p_iv(h))
        else:variable.append(-log1p_iv(-c['alpha']))
    delta=(HALF-beta0)/beta0
    return log1p_iv(rational(delta))+sum(variable,mp.iv.mpf(0))


def residual_box(nodes,t):
    ti=point(t);left=[];near=[];ambiguous=False
    for c in nodes:
        lo,hi=bounds(c['p']);x=(ti-c['p'])*c['rho'];g=c['alpha']*mp.iv.exp(-x**2/2);b=-log1p_iv(-g)/2
        if mp.mpf(t)>hi:left.append(c);near.append(-b)
        elif mp.mpf(t)<=lo:near.append(b)
        else:ambiguous=True;break
    if not ambiguous:return prefix_constant(left)+sum(near,mp.iv.mpf(0))
    low=[];high=[]
    for c in nodes:
        xl,xh=bounds((ti-c['p'])*c['rho']);al,ah=bounds(c['alpha'])
        for dest,alpha,xx in ((low,al,xl),(high,ah,xh)):
            aa=box(alpha,alpha);x=box(xx,xx);b=-log1p_iv(-aa*mp.iv.exp(-x**2/2))/2
            dest.append(b if xx<=0 else -log1p_iv(-aa)-b)
    return box(bounds(sum(low,mp.iv.mpf(0)))[0],bounds(sum(high,mp.iv.mpf(0)))[1])-mp.iv.log(2)


def certify(nodes,z,tol=1e-3):
    lo=float(z)-tol;hi=float(z)+tol
    target=Fraction(1,1000) if tol==1e-3 else Fraction(tol)
    # Float64 addition can make a nominal 2*tol bracket wider than intended.
    if Fraction(float(z))-Fraction(lo)>target:lo=math.nextafter(lo,math.inf)
    if Fraction(hi)-Fraction(float(z))>target:hi=math.nextafter(hi,-math.inf)
    l=residual_box(nodes,lo);r=residual_box(nodes,hi)
    return bool(bounds(l)[1]<0 and bounds(r)[0]>0),{'lo':lo,'hi':hi,'width_upper':float(Fraction(hi)-Fraction(lo)),
        'left_residual_bounds':list(map(lambda v:mp.nstr(v,60),bounds(l))),
        'right_residual_bounds':list(map(lambda v:mp.nstr(v,60),bounds(r))),
        'left_sign_upper':mp.nstr(bounds(l)[1],20),'right_sign_lower':mp.nstr(bounds(r)[0],20)}


def source_coefficients(item):
    U,mu,F=construct(item['mean'],item['scales'],item['R'],item['view'],item['quat'])
    cam=camera_exact(item);qf=[Fraction(float(t)) for t in item['qray']]
    offset=np.array([float(cam[i]-qf[i]/qf[2]*cam[2]) for i in range(2)]+[0.])
    q=np.asarray(item['qray']);qn=np.linalg.norm(q);d=q/qn;u=U@d;w=U@offset;a=np.dot(u,u)
    correction=np.dot(u,w)/a;p=float(cam[2]/qf[2])*qn+correction
    cross=np.cross(w,u);E=np.dot(cross,cross)/a;rho=np.sqrt(a);A=float(item['amplitude'])
    if A*math.exp(-E/2)>=float(np.float32(.99)):A=float(np.float32(.99));E=0.
    return (A,float(E),float(p),float(rho))


def double_root(parameters):
    order=sorted(range(len(parameters)),key=lambda i:parameters[i][2]);peaks=[parameters[i][2] for i in order]
    def f(t):
        k=bisect.bisect_left(peaks,t);beta=Fraction(1);shift=[];tails=[]
        for index in order[:k]:
            A,E,p,r=parameters[index]
            if A<1:
                beta*=1-Fraction(A);shift.append(-math.log1p(A/(1-A)*(-math.expm1(-E/2))))
            else:shift.append(-math.log1p(-A*math.exp(-E/2)))
        delta=(HALF-beta)/beta
        base=math.log1p(float(delta))
        for A,E,p,r in parameters:
            g=A*math.exp(-.5*(E+((t-p)*r)**2));b=-math.log1p(-g)/2
            tails.append(-b if t>p else b)
        return math.fsum([base,*shift,*tails])
    if len(parameters)==2 and all(A==.5 and E==0 for A,E,p,r in parameters):
        x,y=parameters;return (x[3]*x[2]+y[3]*y[2])/(x[3]+y[3]),'analytic_exact_mass_two_layer'
    reach=max(1.,10/min(c[3] for c in parameters));lo=min(c[2] for c in parameters)-reach;hi=max(c[2] for c in parameters)+reach
    for _ in range(48):
        if f(lo)<0<f(hi):break
        lo-=reach;hi+=reach;reach*=2
    else:return None,'double_bracket_unresolved'
    for _ in range(240):
        if hi-lo<=1e-10:return (lo+hi)/2,'double_finite'
        mid=(lo+hi)/2;value=f(mid)
        if value<0:lo=mid
        elif value>0:hi=mid
        else:
            a=(lo+mid)/2;b=(mid+hi)/2
            if f(a)<0<f(b):lo,hi=a,b
            else:return None,'double_sign_unresolved'
    return None,'double_iteration_limit'
