// Shared host/device directed intervals. Standard exp/log functions are not used.
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#ifndef __CUDACC__
#include <cfenv>
using std::isfinite;
using std::isnan;
#pragma fenv_access(on)
#define GI_INLINE inline
#else
#include <cuda_runtime.h>
#define GI_INLINE __device__ __forceinline__
#endif

namespace gi {
struct I {double lo,hi;};
GI_INLINE I pt(double x){return {x,x};}
GI_INLINE double low(double a,double b,int op){
#ifdef __CUDACC__
 if(op==0)return __dadd_rd(a,b);if(op==1)return __dmul_rd(a,b);return __ddiv_rd(a,b);
#else
 std::fesetround(FE_DOWNWARD);volatile double x=a,y=b,z;
 if(op==0)z=x+y;else if(op==1)z=x*y;else z=x/y;
 double r=z;std::fesetround(FE_TONEAREST);return r;
#endif
}
GI_INLINE double high(double a,double b,int op){
#ifdef __CUDACC__
 if(op==0)return __dadd_ru(a,b);if(op==1)return __dmul_ru(a,b);return __ddiv_ru(a,b);
#else
 std::fesetround(FE_UPWARD);volatile double x=a,y=b,z;
 if(op==0)z=x+y;else if(op==1)z=x*y;else z=x/y;
 double r=z;std::fesetround(FE_TONEAREST);return r;
#endif
}
GI_INLINE I add(I a,I b){return {low(a.lo,b.lo,0),high(a.hi,b.hi,0)};}
GI_INLINE I neg(I x){return {-x.hi,-x.lo};}
GI_INLINE I sub(I a,I b){return add(a,neg(b));}
GI_INLINE I mul(I a,I b){
 double l[4]={low(a.lo,b.lo,1),low(a.lo,b.hi,1),low(a.hi,b.lo,1),low(a.hi,b.hi,1)};
 double h[4]={high(a.lo,b.lo,1),high(a.lo,b.hi,1),high(a.hi,b.lo,1),high(a.hi,b.hi,1)};
 double lo=l[0],hi=h[0];for(int i=1;i<4;i++){lo=fmin(lo,l[i]);hi=fmax(hi,h[i]);}return {lo,hi};
}
GI_INLINE I div(I a,I b){
 if(b.lo<=0&&b.hi>=0)return {-INFINITY,INFINITY};
 return mul(a,{low(1.,b.hi,2),high(1.,b.lo,2)});
}
GI_INLINE I sq(I a){
 double h=fmax(fabs(a.lo),fabs(a.hi));
 double l=a.lo<=0&&a.hi>=0?0.:fmin(fabs(a.lo),fabs(a.hi));return {low(l,l,1),high(h,h,1)};
}
GI_INLINE I ln2(){return {0x1.62e42fefa39efp-1,0x1.62e42fefa39f0p-1};}
GI_INLINE bool valid(I x){return !isnan(x.lo)&&!isnan(x.hi)&&x.lo<=x.hi;}
GI_INLINE I entire(){return {-INFINITY,INFINITY};}
GI_INLINE I power(I x,int n){I p=pt(1.);for(int i=0;i<n;i++)p=mul(p,x);return p;}

GI_INLINE I log_one(double x){
 if(x==0)return pt(-INFINITY);if(x<0||!isfinite(x))return entire();if(x==1)return pt(0);
 // Bit normalization is exact, including subnormal inputs after exact 2^54 scaling.
 int correction=0;if(x<0x1p-1022){x*=0x1p54;correction=-54;}
 union {double d;uint64_t u;} bits;bits.d=x;
 int k=int((bits.u>>52)&2047)-1023+correction;
 bits.u=(bits.u&0xfffffffffffffull)|(uint64_t(1023)<<52);I r=pt(bits.d);
 I z=div(sub(r,pt(1)),add(r,pt(1))),z2=sq(z),term=z,sum=pt(0);
 for(int j=0;j<25;j++){sum=add(sum,div(term,pt(2*j+1)));term=mul(term,z2);}
 I remainder=div(mul(pt(2),term),mul(pt(51),sub(pt(1),z2)));
 I answer=add(mul(pt(2),sum),{0.,remainder.hi});
 return add(answer,mul(pt(k),ln2()));
}
GI_INLINE I log(I x){if(x.lo<0||!valid(x))return entire();return {log_one(x.lo).lo,log_one(x.hi).hi};}

GI_INLINE I log1p(I x){
 if(!valid(x)||x.lo<=-1)return entire();double r=fmax(fabs(x.lo),fabs(x.hi));
 if(r>=.125)return log(add(pt(1),x));
 I sum=pt(0),term=x;
 for(int k=1;k<=24;k++){I part=div(term,pt(k));sum=add(sum,k&1?part:neg(part));term=mul(term,x);}
 I remainder=div(power(pt(r),25),mul(pt(25),sub(pt(1),pt(r))));
 return add(sum,{-remainder.hi,remainder.hi});
}

GI_INLINE I exp_one(double x){
 if(isnan(x)||x>700)return entire();if(x==0)return pt(1.);if(x==-INFINITY)return pt(0.);
 // exp(-700)<2^-1009, independently proved from the ln2 enclosure.
 if(x<-700)return {0.,0x1p-1009};
 int k=int(floor(x/0x1.62e42fefa39efp-1));
 I r=sub(pt(x),mul(pt(k),ln2()));double radius=fmax(fabs(r.lo),fabs(r.hi));
 if(radius>1||k<-1022||k>1023)return entire();
 I term=pt(1),sum=pt(1);for(int j=1;j<=24;j++){term=div(mul(term,r),pt(j));sum=add(sum,term);}
 I rem=pt(1);for(int j=1;j<=25;j++)rem=div(mul(rem,pt(radius)),pt(j));
 double bound=mul(pt(3),rem).hi;sum=add(sum,{-bound,bound});
 union {double d;uint64_t u;} scale;scale.u=uint64_t(k+1023)<<52;
 I out=mul(sum,pt(scale.d));out.lo=fmax(out.lo,0.);return out;
}
GI_INLINE I exp(I x){if(!valid(x))return entire();return {exp_one(x.lo).lo,exp_one(x.hi).hi};}
GI_INLINE I expm1(I x){
 double r=fmax(fabs(x.lo),fabs(x.hi));if(r>=.125)return sub(exp(x),pt(1));
 I sum=pt(0),term=pt(1);
 for(int j=1;j<=24;j++){term=div(mul(term,x),pt(j));sum=add(sum,term);}
 I rem=pt(1);for(int j=1;j<=25;j++)rem=div(mul(rem,pt(r)),pt(j));
 rem=div(rem,sub(pt(1),pt(r)));return add(sum,{-rem.hi,rem.hi});
}

struct Layer {I p,rho,energy;double amplitude;};
GI_INLINE I log_tail(Layer c,double t){
 I x=mul(sub(pt(t),c.p),c.rho);
 I la=sub(log(pt(c.amplitude)),mul(pt(.5),c.energy));I cap=log(pt(double(float(.99))));
 la={fmin(la.lo,cap.lo),fmin(la.hi,cap.hi)};
 I lg=sub(la,mul(pt(.5),sq(x)));
 if(lg.hi<-40)return add(sub(lg,ln2()),{0.,0x1p-56});
 I b=mul(pt(-.5),log1p(neg(exp(lg))));return log(b);
}
GI_INLINE I prefix(const Layer* nodes,int n,double t,bool& okay){
 I beta=pt(1),shift=pt(0);double cap=double(float(.99));okay=true;
 for(int i=0;i<n;i++){
  Layer c=nodes[i];if(c.amplitude==0)continue;
  if(t<=c.p.lo)continue;if(t<=c.p.hi){okay=false;return entire();}
  if(c.amplitude<=cap){
   I A=pt(c.amplitude);beta=mul(beta,sub(pt(1),A));
   I h=mul(div(A,sub(pt(1),A)),neg(expm1(mul(pt(-.5),c.energy))));
   // Energy is nonnegative, so 1-exp(-E/2) belongs to [0,1].
   h.lo=fmax(0.,h.lo);shift=sub(shift,log1p(h));
  }else{
   I raw=mul(pt(c.amplitude),exp(mul(pt(-.5),c.energy)));I a={fmin(cap,raw.lo),fmin(cap,raw.hi)};
   beta=mul(beta,sub(pt(1),a));
  }
 }
 if(beta.lo<=0){okay=false;return entire();}
 return add(log1p(div(sub(pt(.5),beta),beta)),shift);
}

GI_INLINE I scale_value(double x,double s){
 if(x==0)return pt(0);I v=exp(sub(log(pt(fabs(x))),pt(s)));return x<0?neg(v):v;
}
// Return R*exp(-s), with a positive scale. Its interval determines R's sign.
GI_INLINE I residual(const Layer* nodes,int n,double t,double& s,bool& okay){
 I c=prefix(nodes,n,t,okay);if(!okay||!valid(c)||!isfinite(c.lo)||!isfinite(c.hi))return entire();
 s=-INFINITY;double mag=fmax(fabs(c.lo),fabs(c.hi));if(mag>0)s=log(pt(mag)).hi;
 for(int i=0;i<n;i++)if(nodes[i].amplitude>0){I b=log_tail(nodes[i],t);if(!valid(b)){okay=false;return entire();}s=fmax(s,b.hi);}
 if(!isfinite(s)){okay=false;return entire();}
 I result={scale_value(c.lo,s).lo,scale_value(c.hi,s).hi};
 for(int i=0;i<n;i++)if(nodes[i].amplitude>0){I b=exp(sub(log_tail(nodes[i],t),pt(s)));result=add(result,t>nodes[i].p.hi?neg(b):b);}
 okay=valid(result);return result;
}
} // namespace gi
