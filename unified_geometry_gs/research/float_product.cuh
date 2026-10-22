#pragma once
#include <cuda_runtime.h>
namespace fp {
struct I{float lo,hi;};
__device__ __forceinline__ I pt(float x){return {x,x};}
__device__ __forceinline__ I full(){return {-INFINITY,INFINITY};}
__device__ __forceinline__ bool valid(I a){return !isnan(a.lo)&&!isnan(a.hi)&&a.lo<=a.hi;}
__device__ __forceinline__ I from(double lo,double hi){return {__double2float_rd(lo),__double2float_ru(hi)};}
__device__ __forceinline__ I add(I a,I b){return {__fadd_rd(a.lo,b.lo),__fadd_ru(a.hi,b.hi)};}
__device__ __forceinline__ I sub(I a,I b){return add(a,{-b.hi,-b.lo});}
__device__ __forceinline__ I mul(I a,I b){
 float l[4]={__fmul_rd(a.lo,b.lo),__fmul_rd(a.lo,b.hi),__fmul_rd(a.hi,b.lo),__fmul_rd(a.hi,b.hi)};
 float h[4]={__fmul_ru(a.lo,b.lo),__fmul_ru(a.lo,b.hi),__fmul_ru(a.hi,b.lo),__fmul_ru(a.hi,b.hi)};
 float lo=l[0],hi=h[0];for(int i=0;i<4;i++){if(isnan(l[i])||isnan(h[i]))return full();lo=fminf(lo,l[i]);hi=fmaxf(hi,h[i]);}return {lo,hi};
}
__device__ __forceinline__ I div(I a,I b){if(!valid(b)||b.lo<=0)return full();return mul(a,{__fdiv_rd(1.f,b.hi),__fdiv_ru(1.f,b.lo)});}
__device__ __forceinline__ I sq(I a){if(!valid(a))return full();float h=fmaxf(fabsf(a.lo),fabsf(a.hi)),l=a.lo<=0&&a.hi>=0?0.f:fminf(fabsf(a.lo),fabsf(a.hi));return {__fmul_rd(l,l),__fmul_ru(h,h)};}
__device__ __forceinline__ I root(I a){if(!valid(a)||a.lo<0)return full();return {__fsqrt_rd(a.lo),__fsqrt_ru(a.hi)};}
// Generated exact reciprocal enclosure table, not rounded division assumptions.
#include "float_reciprocals.cuh"
__device__ __forceinline__ I exp_one(float x){
 if(isnan(x)||x>0)return full();if(x==0)return pt(1);if(x==-INFINITY)return pt(0);if(x<-80)return {0,0x1p-115f};
 I ln2={0x1.62e42ep-1f,0x1.62e430p-1f};int k=int(floorf(x/ln2.lo));
 I r=sub(pt(x),mul(pt(float(k)),ln2));float radius=fmaxf(fabsf(r.lo),fabsf(r.hi));if(radius>1||k<-126||k>0)return full();
 I term=pt(1),sum=pt(1);for(int j=1;j<=8;j++){term=mul(mul(term,r),recip(j));sum=add(sum,term);}
 I rem=pt(1);for(int j=1;j<=9;j++)rem=mul(mul(rem,pt(radius)),recip(j));float bound=__fmul_ru(3.f,rem.hi);
 I answer=mul(add(sum,{-bound,bound}),pt(__int_as_float((k+127)<<23)));answer.lo=fmaxf(0,answer.lo);answer.hi=fminf(1,answer.hi);return answer;
}
__device__ __forceinline__ I exp(I x){if(!valid(x))return full();return {exp_one(x.lo).lo,exp_one(x.hi).hi};}
__device__ __forceinline__ I alpha(double amp,double elo,double ehi){
 if(!(amp>=0)||!isfinite(amp)||elo<0||elo>ehi)return full();
 I a=mul(from(amp,amp),exp(mul(pt(-.5f),from(elo,ehi))));if(!valid(a))return full();
 a.lo=fmaxf(0,fminf(.99f,a.lo));a.hi=fminf(fminf(.99f,__double2float_ru(amp)),a.hi);return a;
}
__device__ __forceinline__ I probability(I a){if(!valid(a))return {0,1};return {fmaxf(0,a.lo),fminf(1,a.hi)};}
__device__ __forceinline__ I factor(const double* v,double t,int &bad){
 if(v[0]!=0){bad=1;return {0,1};}I p=from(v[1],v[2]),rho=from(v[3],v[4]),a=from(v[24],v[25]),ti=from(t,t);
 if(!valid(p)||!valid(rho)||!valid(a)||!isfinite(p.lo)||!isfinite(p.hi)||!isfinite(rho.hi)||rho.lo<=0||a.lo<0||a.hi>=1){bad=1;return {0,1};}
 I x=mul(sub(ti,p),rho),g=exp(mul(pt(-.5f),sq(x))),b=root(sub(pt(1),mul(a,g)));
 if(!valid(b)||b.lo<=0){bad=1;return {0,1};}I before=probability(b),after=probability(div(sub(pt(1),a),b));
 if(ti.hi<=p.lo)return before;if(ti.lo>=p.hi)return after;return {fminf(before.lo,after.lo),fmaxf(before.hi,after.hi)};
}
}
