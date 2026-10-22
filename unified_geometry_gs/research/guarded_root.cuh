#pragma once
#include "guarded_preprocess.cuh"
namespace gr {
struct Candidate {double A,E,p,rho,logA,shift;};
constexpr double TOL=0x1.0624dd2f1a9fbp-10;
__device__ __forceinline__ int mass(const double* packets,int start,int n,int mode){
 gi::I beta=gi::pt(1);for(int i=0;i<n;i++){
  const double* p=packets+(2*(start+i)+mode)*gp::WIDTH;if(p[0]!=0)return 0;
  beta=gi::mul(beta,gi::sub(gi::pt(1),{p[24],p[25]}));
 }return beta.hi<.5?1:beta.lo>.5?3:beta.lo==.5&&beta.hi==.5?2:0;
}
__device__ __forceinline__ bool parameters(const double* packets,int start,int n,int mode,Candidate* c,gi::Layer* bounds){
 double cap=double(float(.99));for(int i=0;i<n;i++){
  const double* p=packets+(2*(start+i)+mode)*gp::WIDTH;if(p[0]!=0)return false;
  bounds[i]={{p[1],p[2]},{p[3],p[4]},{p[5],p[6]},p[7]};
  double A=p[7],E=p[10],peak=p[8],rho=p[9];double alpha=fmin(cap,A*exp(-E/2));
  if(mode==0){A=double(float(alpha));E=0.;peak=double(float(peak));rho=double(float(rho));}
  else if(alpha>=cap){A=cap;E=0.;}
  if(!isfinite(A)||!isfinite(E)||!isfinite(peak)||!isfinite(rho)||A<0||A>=1||E<0||rho<=0)return false;
  c[i]={A,E,peak,rho,A>0?log(A):-INFINITY,-log1p(A/(1-A)*(-expm1(-E/2)))};
 }return true;
}
__device__ __forceinline__ double log_tail(Candidate c,double t){
 double x=(t-c.p)*c.rho;double l=c.logA-.5*(c.E+x*x);
 return l<-40?l-0x1.62e42fefa39efp-1:log(-.5*log1p(-exp(l)));
}
__device__ __forceinline__ double residual(const Candidate* c,int n,double t){
 double beta=1.,shift=0.;for(int i=0;i<n;i++)if(t>c[i].p){beta*=1-c[i].A;shift+=c[i].shift;}
 if(!(beta>0))return NAN;double constant=log1p((.5-beta)/beta)+shift;
 double scale=constant!=0?log(fabs(constant)):-INFINITY;
 for(int i=0;i<n;i++)if(c[i].A>0)scale=fmax(scale,log_tail(c[i],t));
 if(!isfinite(scale))return NAN;
 double value=constant==0?0:copysign(exp(log(fabs(constant))-scale),constant);
 for(int i=0;i<n;i++)if(c[i].A>0){double v=exp(log_tail(c[i],t)-scale);value+=t>c[i].p?-v:v;}
 return value;
}
__device__ __forceinline__ bool propose(const Candidate* c,int n,double& z,int& method){
 if(n==2&&c[0].A==.5&&c[1].A==.5&&c[0].E==0&&c[1].E==0){z=(c[0].rho*c[0].p+c[1].rho*c[1].p)/(c[0].rho+c[1].rho);method=1;return isfinite(z);}
 method=2;double pmin=c[0].p,pmax=pmin,rmin=c[0].rho;
 for(int i=1;i<n;i++){pmin=fmin(pmin,c[i].p);pmax=fmax(pmax,c[i].p);rmin=fmin(rmin,c[i].rho);}
 double reach=fmax(1.,10/rmin),lo=pmin-reach,hi=pmax+reach;
 bool okay=false;for(int i=0;i<48;i++){double l=residual(c,n,lo),h=residual(c,n,hi);if(l<0&&h>0){okay=true;break;}lo-=reach;hi+=reach;reach*=2;}
 if(!okay)return false;
 for(int i=0;i<200;i++){
  double mid=.5*lo+.5*hi;if(hi-lo<=1e-10||mid==lo||mid==hi){z=mid;return true;}
  double v=residual(c,n,mid);if(!isfinite(v))return false;
  if(v<0)lo=mid;else if(v>0)hi=mid;else{z=mid;return true;} // Candidate only; strict certificate is still mandatory.
 }return false;
}
__device__ __forceinline__ bool certify(const gi::Layer* b,int n,double z,double* out){
 double lo=__dsub_ru(z,TOL),hi=__dadd_rd(z,TOL),ls=0,rs=0;bool lok=false,rok=false;
 gi::I l=gi::residual(b,n,lo,ls,lok),r=gi::residual(b,n,hi,rs,rok);
 out[2]=z;out[3]=lo;out[4]=hi;out[5]=l.lo;out[6]=l.hi;out[7]=r.lo;out[8]=r.hi;out[9]=ls;out[10]=rs;
 return hi>lo&&lok&&rok&&l.hi<0&&r.lo>0&&isfinite(z)&&gi::valid(l)&&gi::valid(r);
}
__global__ void solve(int count,const double* packets,const int* sections,double* output){
 int model=blockIdx.x*blockDim.x+threadIdx.x;if(model>=count)return;int start=sections[2*model],n=sections[2*model+1];double* out=output+12*model;
 out[0]=0;if(n<=0||n>192)return;Candidate candidate[192];gi::Layer bounds[192];
 for(int mode=0;mode<2;mode++){
  out[1]=mode;int kind=mass(packets,start,n,mode);
  if(kind==2||kind==3){out[0]=kind;return;}if(kind!=1)continue;
  if(!parameters(packets,start,n,mode,candidate,bounds))continue;
  double z;int method;if(!propose(candidate,n,z,method))continue;out[11]=method;
  if(certify(bounds,n,z,out)){out[0]=1;return;}
 }
}
} // namespace gr
