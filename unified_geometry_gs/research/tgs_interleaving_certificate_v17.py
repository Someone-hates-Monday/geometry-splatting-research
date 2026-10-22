"""Exact recorded-alpha mathematics; no torch, renderer, or optimizer."""
from fractions import Fraction as F
import numpy as np
def q(x):return F.from_float(float(x))
ZERO=[F(0)]*3

def layers(packet,trace):
    assert len(packet)==len(trace)
    return [(int(p[16]),q(t[4]) if not t[5] and q(t[4])>=q(np.float32(1/255)) else F(0),[q(x) for x in p[13:16]]) for p,t in zip(packet,trace)]

def compose(a):
    T=F(1);C=ZERO.copy()
    for _,alpha,color in a:
        for ch in range(3):C[ch]+=T*alpha*color[ch]
        T*=1-alpha
    return C

def certificate(old,new,parent,N):
    old_parent=[x for x in old if x[0]==parent]
    assert len(old_parent)==1,'parent absent; this version cannot locate the parent slot'
    parent_layer=old_parent[0];c=parent_layer[2];P=parent_layer[1]
    ext=[x for x in old if x[0]!=parent];m=next(i for i,x in enumerate(old) if x[0]==parent)
    child=[x for x in new if x[0]>=N-1];assert all(x[2]==c for x in child)
    actual_ext=[(x[0] if x[0]<parent else x[0]+1,x[1],x[2]) for x in new if x[0]<N-1]
    assert actual_ext==ext,'external effective layers changed'
    R=F(1)
    for _,a,_ in child:R*=1-a
    Q=1-R;U=F(1)
    for _,a,_ in ext[:m]:U*=1-a
    B=compose(ext[m:]);window=[(Q-P)*U*(c[ch]-B[ch]) for ch in range(3)]
    wb=[abs(x) for x in window];order=ZERO.copy();ob=ZERO.copy();Uj=F(1);V=F(1);j=0;active=0
    for id,a,d in new:
        if id>=N-1:V*=1-a;continue
        vstar=F(1) if j<m else R;delta=V-vstar
        if a and delta:active+=1
        for ch in range(3):
            term=Uj*a*(d[ch]-c[ch])*delta;order[ch]+=term;ob[ch]+=abs(term)
        Uj*=1-a;j+=1
    assert j==len(ext)
    actual=[a-b for a,b in zip(compose(new),compose(old))]
    assert actual==[a+b for a,b in zip(window,order)]
    assert all(abs(v)<=w+o for v,w,o in zip(actual,wb,ob))
    return dict(actual=actual,window=window,order=order,window_bound=wb,order_bound=ob,
        real_bound=[w+o for w,o in zip(wb,ob)],active_crossed_external=active)

def endpoint_envelope(full,trace):
    # Native trace is a prefix of the full packet; stop excludes the current layer.
    accepted=[full[i] for i,r in enumerate(trace) if r[8]]
    for i,r in enumerate(trace):assert q(r[4])==full[i][1] or full[i][1]==0
    n=len(accepted);u=F(1,2**24);k=3*n+3;assert k*u<1
    gamma=k*u/(1-k*u);prefix=compose(accepted);total=compose(full)
    # A deliberately strong sufficient normal-range premise, checked per endpoint.
    positive=[v for _,a,c in accepted for v in c if a and v>0]
    if positive:
        lower=min(positive)*q(np.float32(1/255))*q(np.float32(1e-4))/(1+u)**(2*n+3)
        assert lower>F(1,2**126),'subnormal intermediates not certified'
        assert max(positive)*(1+gamma)<F(2)**120,'overflow margin not certified'
    tail=[a-b for a,b in zip(total,prefix)];assert all(x>=0 for x in tail)
    return dict(prefix=prefix,tail=tail,round_bound=[gamma*x for x in prefix],accepted=n,
        normal_range_premise=True,stop=bool(any(r[9] for r in trace)))
