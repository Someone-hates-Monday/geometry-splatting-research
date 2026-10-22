#pragma once
#include "shared_cache.cuh"
#include "short_interval.cuh"
#include "float_product.cuh"
namespace dr {
using gi::I;
constexpr double TOL=0x1.0624dd2f1a9fbp-10;
__device__ __forceinline__ double sum(double x){int lane=threadIdx.x%32;for(int s=16;s;s/=2){double v=__shfl_down_sync(0xffffffff,x,s);if(lane<s)x+=v;}return __shfl_sync(0xffffffff,x,0);}
__device__ __forceinline__ double prod(double x){int lane=threadIdx.x%32;for(int s=16;s;s/=2){double v=__shfl_down_sync(0xffffffff,x,s);if(lane<s)x*=v;}return __shfl_sync(0xffffffff,x,0);}
__device__ __forceinline__ double maximum(double x){int lane=threadIdx.x%32;for(int s=16;s;s/=2){double v=__shfl_down_sync(0xffffffff,x,s);if(lane<s)x=fmax(x,v);}return __shfl_sync(0xffffffff,x,0);}
__device__ __forceinline__ double minimum(double x){return -maximum(-x);}
__device__ __forceinline__ int any(int x){return __any_sync(0xffffffff,x);}
__device__ __forceinline__ I reduce(I x,bool product){int lane=threadIdx.x%32;for(int s=16;s;s/=2){I v={__shfl_down_sync(0xffffffff,x.lo,s),__shfl_down_sync(0xffffffff,x.hi,s)};if(lane<s)x=product?gi::mul(x,v):gi::add(x,v);}return {__shfl_sync(0xffffffff,x.lo,0),__shfl_sync(0xffffffff,x.hi,0)};}
__device__ __forceinline__ gi::Layer layer(const double* v){return {{v[1],v[2]},{v[3],v[4]},{v[5],v[6]},v[7]};}
struct C{double A,E,p,rho,logA,shift;};
__device__ __forceinline__ C nominal(const double* v){
 double A=v[7],E=v[10],cap=double(float(.99));double la=A>0?log(A)-.5*E:-INFINITY;
 if(A>cap||la>=log(cap)){A=exp(fmin(log(cap),la));E=0;}
 return {A,E,v[8],v[9],A>0?log(A):-INFINITY,A>0?-log1p(A/(1-A)*(-expm1(-.5*E))):0};
}
__device__ __forceinline__ double tail(C c,double t){double x=(t-c.p)*c.rho,l=c.logA-.5*(c.E+x*x);return l<-40?l-0x1.62e42fefa39efp-1:log(-.5*log1p(-exp(l)));}
__device__ __forceinline__ double proposal_residual(const double* p,int64_t b,int64_t e,double t){
 int lane=threadIdx.x%32;double beta=1,shift=0,scale=-INFINITY;int bad=0;
 for(int64_t i=b+lane;i<e;i+=32){C c=nominal(p+26*i);bad|=!isfinite(c.p)||!(c.rho>0)||!isfinite(c.rho)||c.A<0||c.A>=1||c.E<0;if(t>c.p&&c.A>0){beta*=1-c.A;shift+=c.shift;}if(c.A>0)scale=fmax(scale,tail(c,t));}
 if(any(bad))return NAN;beta=prod(beta);shift=sum(shift);if(beta==0)return 1;
 double constant=log1p((.5-beta)/beta)+shift;scale=maximum(scale);scale=fmax(scale,constant!=0?log(fabs(constant)):-INFINITY);if(!isfinite(scale))return NAN;
 double value=0;for(int64_t i=b+lane;i<e;i+=32){C c=nominal(p+26*i);if(c.A>0){double v=exp(tail(c,t)-scale);value+=t>c.p?-v:v;}}
 return sum(value)+(constant==0?0:copysign(exp(log(fabs(constant))-scale),constant));
}
__device__ __forceinline__ bool propose(const double* p,int64_t b,int64_t e,double& z,int& iterations){
 int lane=threadIdx.x%32;double pmin=INFINITY,pmax=-INFINITY,rmin=INFINITY;
 for(int64_t i=b+lane;i<e;i+=32){const double* v=p+26*i;pmin=fmin(pmin,v[8]);pmax=fmax(pmax,v[8]);rmin=fmin(rmin,v[9]);}pmin=minimum(pmin);pmax=maximum(pmax);rmin=minimum(rmin);if(!(rmin>0)||!isfinite(pmin)||!isfinite(pmax))return false;
 double reach=fmax(1.,10/rmin),lo=pmin-reach,hi=pmax+reach;bool bracket=false;
 for(int j=0;j<48;j++){double l=proposal_residual(p,b,e,lo),h=proposal_residual(p,b,e,hi);if(l<0&&h>0){bracket=true;break;}lo-=reach;hi+=reach;reach*=2;}
 if(!bracket)return false;
 for(int j=0;j<200;j++){iterations=j+1;double mid=.5*lo+.5*hi;if(hi-lo<=1e-10||mid==lo||mid==hi){z=mid;return isfinite(z);}double v=proposal_residual(p,b,e,mid);if(!isfinite(v))return false;if(v<0)lo=mid;else if(v>0)hi=mid;else{z=mid;return true;}}
 return false;
}
__device__ __forceinline__ I product_endpoint(const double* p,int64_t b,int64_t e,double t,int& bad){
 int lane=threadIdx.x%32;fp::I value=fp::pt(1);for(int64_t i=b+lane;i<e;i+=32)value=fp::mul(value,fp::factor(p+26*i,t,bad));
 for(int s=16;s;s/=2){fp::I v={__shfl_down_sync(0xffffffff,value.lo,s),__shfl_down_sync(0xffffffff,value.hi,s)};if(lane<s)value=fp::mul(value,v);}bad=any(bad);return {double(__shfl_sync(0xffffffff,value.lo,0)),double(__shfl_sync(0xffffffff,value.hi,0))};
}
template<bool SHORT> __device__ __forceinline__ I interval_residual(const double* p,int64_t b,int64_t e,double t,double& scale,int& bad){
 int lane=threadIdx.x%32;I beta=gi::pt(1),shift=gi::pt(0);double cap=double(float(.99));
 for(int64_t i=b+lane;i<e;i+=32){const double* v=p+26*i;gi::Layer c=layer(v);if(v[0]!=0){bad=1;continue;}if(c.amplitude==0||t<=c.p.lo)continue;if(t<=c.p.hi){bad=1;continue;}
  if(c.amplitude<=cap){I A=gi::pt(c.amplitude);beta=gi::mul(beta,gi::sub(gi::pt(1),A));I h=gi::mul(gi::div(A,gi::sub(gi::pt(1),A)),gi::neg(SHORT?gf::expm1(gi::mul(gi::pt(-.5),c.energy)):gi::expm1(gi::mul(gi::pt(-.5),c.energy))));h.lo=fmax(0.,h.lo);shift=gi::sub(shift,SHORT?gf::log1p(h):gi::log1p(h));}
  else{I raw=gi::mul(gi::pt(c.amplitude),SHORT?gf::exp(gi::mul(gi::pt(-.5),c.energy)):gi::exp(gi::mul(gi::pt(-.5),c.energy)));I a={fmin(cap,raw.lo),fmin(cap,raw.hi)};beta=gi::mul(beta,gi::sub(gi::pt(1),a));}
 }
 beta=reduce(beta,true);shift=reduce(shift,false);bad=any(bad||!gi::valid(beta)||beta.lo<=0);if(bad)return gi::entire();
 I fraction=gi::div(gi::sub(gi::pt(.5),beta),beta),constant=gi::add(SHORT?gf::log1p(fraction):gi::log1p(fraction),shift);
 bad=any(!gi::valid(constant)||!isfinite(constant.lo)||!isfinite(constant.hi));if(bad)return gi::entire();
 scale=-INFINITY;double mag=fmax(fabs(constant.lo),fabs(constant.hi));if(mag>0)scale=(SHORT?gf::log(gi::pt(mag)):gi::log(gi::pt(mag))).hi;
 for(int64_t i=b+lane;i<e;i+=32){gi::Layer c=layer(p+26*i);if(c.amplitude>0){I v=SHORT?gf::log_tail(c,t):gi::log_tail(c,t);if(!gi::valid(v))bad=1;scale=fmax(scale,v.hi);}}
 scale=maximum(scale);bad=any(bad||!isfinite(scale));if(bad)return gi::entire();I value=gi::pt(0);
 for(int64_t i=b+lane;i<e;i+=32){gi::Layer c=layer(p+26*i);if(c.amplitude>0){I v=SHORT?gf::log_tail(c,t):gi::log_tail(c,t);I a=SHORT?gf::exp(gi::sub(v,gi::pt(scale))):gi::exp(gi::sub(v,gi::pt(scale)));value=gi::add(value,t>c.p.hi?gi::neg(a):a);}}
 value=reduce(value,false);I cl=SHORT?gf::scale_value(constant.lo,scale):gi::scale_value(constant.lo,scale),ch=SHORT?gf::scale_value(constant.hi,scale):gi::scale_value(constant.hi,scale);value=gi::add({cl.lo,ch.hi},value);bad=any(!gi::valid(value));return value;
}
__global__ void decode(const double* cache,const int* ids,const int64_t* sections,const double* queries,const float* amp,double* packets,int64_t n,int models,int mode,int P){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;int lo=0,hi=models+1;while(lo<hi){int m=lo+(hi-lo)/2;if(sections[m]<=i)lo=m+1;else hi=m;}int pixel=lo-1,id=ids[i];double* v=packets+26*i;if(pixel<0||pixel>=models||id<0||id>=P){v[0]=6;return;}double src[36];for(int j=0;j<3;j++)src[32+j]=queries[4*pixel+j];src[35]=amp[id];gc::decode_one(src,cache+28*id,mode,v);
}
__global__ void solve(const double* packets,const int64_t* sections,const double* query,double* output,int count,int mode){
 int lane=threadIdx.x%32,model=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(model>=count)return;int64_t b=sections[model],e=sections[model+1];double* o=output+20*model;int bad=0;I beta=gi::pt(1);
 for(int64_t i=b+lane;i<e;i+=32){const double* v=packets+26*i;bad|=v[0]!=0;beta=gi::mul(beta,gi::sub(gi::pt(1),{v[24],v[25]}));}beta=reduce(beta,true);bad=any(bad||!gi::valid(beta));
 if(lane==0){o[0]=0;o[1]=mode;o[12]=0;o[13]=0;o[14]=double(e-b);o[15]=bad;o[16]=beta.lo;o[17]=beta.hi;o[18]=0;o[19]=mode+1;}
 if(bad||e<=b)return;if(beta.lo>=.5){if(lane==0)o[0]=beta.lo==.5&&beta.hi==.5?2:3;return;}if(!(beta.hi<.5))return;
 double t=NAN;int iterations=0;if(!propose(packets,b,e,t,iterations))return;
 const double* q=query+4*model;double norm=__dsqrt_rn(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);double z=double(__double2float_rn(t/norm));I raw[3]={gi::pt(q[0]),gi::pt(q[1]),gi::pt(q[2])};I center=gi::mul(gi::pt(z),gp::norm(raw,3));double lo=__dsub_ru(center.hi,TOL),hi=__dadd_rd(center.lo,TOL);
 if(lane==0){o[2]=z;o[3]=t;o[4]=lo;o[5]=hi;o[13]=iterations;o[18]=2;}
 if(!isfinite(z)||!isfinite(lo)||!isfinite(hi)||!(lo<hi))return;
 int lb=0,rb=0;I l=product_endpoint(packets,b,e,lo,lb),r=product_endpoint(packets,b,e,hi,rb);double ls=0,rs=0;int method=1;bool accepted=!lb&&!rb&&l.lo>.5&&r.hi<.5;
 if(!accepted){lb=rb=0;method=2;l=interval_residual<true>(packets,b,e,lo,ls,lb);r=interval_residual<true>(packets,b,e,hi,rs,rb);accepted=!lb&&!rb&&l.hi<0&&r.lo>0;}
 if(!accepted){lb=rb=0;method=3;l=interval_residual<false>(packets,b,e,lo,ls,lb);r=interval_residual<false>(packets,b,e,hi,rs,rb);accepted=!lb&&!rb&&l.hi<0&&r.lo>0;}
 if(lane==0){o[0]=accepted?1:0;o[6]=l.lo;o[7]=l.hi;o[8]=r.lo;o[9]=r.hi;o[10]=ls;o[11]=rs;o[12]=method;o[15]=lb||rb;}
}
}
