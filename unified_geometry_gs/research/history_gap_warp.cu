#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

constexpr double LN2=0x1.62e42fefa39efp-1;
constexpr double CAP=0.9900000095367431640625;
struct Sum{double s=0,c=0;__device__ void add(double x){double y=x-c,t=s+y;c=(t-s)-y;s=t;}};
struct DD{double h=1,l=0;__device__ void times(double a){
 double dh=1-a,bb=dh-1,dl=(1-(dh-bb))+(-a-bb);
 double p=h*dh,e=fma(h,dh,-p)+h*dl+l*dh+l*dl;
 double s=p+e;l=e-(s-p);h=s;
}};
struct Inputs{const double*p,*rho,*E,*A;};
__device__ double logalpha(Inputs v,int j){return fmin(log(v.A[j])-.5*v.E[j],log(CAP));}
__device__ double logtail(double lg){return lg< -40 ? lg-LN2:log(-.5*log1p(-exp(lg)));}
// H(t)-log2, scaled to avoid underflow of the Gaussian tails.
__device__ double residual(Inputs v,int b,int e,double t){
 DD beta;Sum shift,logbeta;double scale=-INFINITY;
 for(int j=b;j<e;++j){double la=logalpha(v,j),x=(t-v.p[j])*v.rho[j];
  scale=fmax(scale,logtail(la-.5*x*x));
  if(t>v.p[j]){double A=v.A[j],E=v.E[j];
   if(log(A)-.5*E>log(CAP)){A=CAP;E=0;}
   if(A<1){beta.times(A);logbeta.add(log1p(-A));shift.add(-log1p(A/(1-A)*(-expm1(-.5*E))));}
   else{double a=exp(la);beta.times(a);logbeta.add(log1p(-a));}
  }
 }
 double constant;
 if(beta.h>DBL_MIN){double diff=(.5-beta.h)-beta.l;constant=log1p(diff/(beta.h+beta.l))+shift.s;}
 else constant=-LN2-logbeta.s+shift.s;
 double lc=constant==0 ? -INFINITY:log(fabs(constant));scale=fmax(scale,lc);
 if(!isfinite(scale))return NAN;
 Sum value;if(constant!=0)value.add(copysign(exp(lc-scale),constant));
 for(int j=b;j<e;++j){double x=(t-v.p[j])*v.rho[j],lt=logtail(logalpha(v,j)-.5*x*x);value.add((t>v.p[j] ? -1:1)*exp(lt-scale));}
 return value.s;
}
constexpr unsigned MASK=0xffffffffu;
__device__ DD dd_add(DD a,DD b){
 double s=a.h+b.h,v=s-a.h,e=(a.h-(s-v))+(b.h-v)+a.l+b.l;
 DD out;out.h=s+e;out.l=e-(out.h-s);return out;
}
__device__ DD dd_mul(DD a,DD b){
 double p=a.h*b.h,e=fma(a.h,b.h,-p)+a.h*b.l+a.l*b.h+a.l*b.l;
 DD out;out.h=p+e;out.l=e-(out.h-p);return out;
}
__device__ DD warp_dd(DD a,bool product){
 int lane=threadIdx.x&31;
 for(int step=16;step;step>>=1){DD b;b.h=__shfl_down_sync(MASK,a.h,step);b.l=__shfl_down_sync(MASK,a.l,step);if(lane<step)a=product?dd_mul(a,b):dd_add(a,b);}
 a.h=__shfl_sync(MASK,a.h,0);a.l=__shfl_sync(MASK,a.l,0);return a;
}
__device__ double warp_sum(Sum a){DD d;d.h=a.s;d.l=-a.c;d=warp_dd(d,false);return d.h+d.l;}
__device__ double warp_max(double a){for(int s=16;s;s>>=1)a=fmax(a,__shfl_down_sync(MASK,a,s));return __shfl_sync(MASK,a,0);}
__device__ double warp_min(double a){for(int s=16;s;s>>=1)a=fmin(a,__shfl_down_sync(MASK,a,s));return __shfl_sync(MASK,a,0);}
__device__ double warp_residual(Inputs v,int b,int e,double t){
 int lane=threadIdx.x&31;DD beta;Sum shift,logbeta;double scale=-INFINITY;
 for(int j=b+lane;j<e;j+=32){double la=logalpha(v,j),x=(t-v.p[j])*v.rho[j];scale=fmax(scale,logtail(la-.5*x*x));
  if(t>v.p[j]){double A=v.A[j],E=v.E[j];if(log(A)-.5*E>log(CAP)){A=CAP;E=0;}
   if(A<1){beta.times(A);logbeta.add(log1p(-A));shift.add(-log1p(A/(1-A)*(-expm1(-.5*E))));}
   else{double a=exp(la);beta.times(a);logbeta.add(log1p(-a));}
  }
 }
 beta=warp_dd(beta,true);double shift_sum=warp_sum(shift),lb=warp_sum(logbeta);double constant;
 if(beta.h>DBL_MIN){double diff=(.5-beta.h)-beta.l;constant=log1p(diff/(beta.h+beta.l))+shift_sum;}
 else constant=-LN2-lb+shift_sum;
 double lc=constant==0 ? -INFINITY:log(fabs(constant));scale=fmax(warp_max(scale),lc);if(!isfinite(scale))return NAN;
 Sum value;if(lane==0&&constant!=0)value.add(copysign(exp(lc-scale),constant));
 for(int j=b+lane;j<e;j+=32){double x=(t-v.p[j])*v.rho[j];value.add((t>v.p[j]?-1:1)*exp(logtail(logalpha(v,j)-.5*x*x)-scale));}
 return warp_sum(value);
}
__global__ void gap_kernel(Inputs v,const int64_t*sections,int M,double*loss,double*seed,double*info){
 int ray=blockIdx.x,lane=threadIdx.x&31;if(ray>=M)return;int b=sections[ray],e=sections[ray+1];double*o=info+ray*8;
 if(lane==0){o[0]=-1;o[1]=NAN;o[2]=NAN;o[3]=NAN;o[4]=0;o[5]=e-b;o[6]=NAN;o[7]=NAN;loss[ray]=NAN;}
 for(int j=b+lane;j<e;j+=32)for(int k=0;k<4;++k)seed[j*4+k]=NAN;
 if(e<=b)return;bool bad=false;double lo=INFINITY,hi=-INFINITY;Sum omega;
 for(int j=b+lane;j<e;j+=32){bad|=!(v.rho[j]>0&&v.A[j]>0&&v.E[j]>=0&&isfinite(v.p[j])&&isfinite(v.rho[j])&&isfinite(v.A[j])&&isfinite(v.E[j]));lo=fmin(lo,v.p[j]);hi=fmax(hi,v.p[j]);omega.add(-log1p(-exp(logalpha(v,j))));}
 if(__any_sync(MASK,bad))return;lo=warp_min(lo);hi=warp_max(hi);double mass=warp_sum(omega);if(lane==0)o[3]=mass;
 double left=warp_residual(v,b,e,lo),right=warp_residual(v,b,e,hi);if(lane==0){o[6]=left;o[7]=right;}if(!isfinite(left)||!isfinite(right))return;
 if(left>=0){if(lane==0){o[0]=3;o[1]=lo;o[2]=0;loss[ray]=0;}for(int j=b+lane;j<e;j+=32)for(int k=0;k<4;++k)seed[j*4+k]=0;return;}
 if(right<=0){double D=2*LN2-mass;if(D< -1e-12)return;if(lane==0){o[0]=2;o[1]=hi;o[2]=D;loss[ray]=D*D;}
  for(int j=b+lane;j<e;j+=32){double alpha=exp(logalpha(v,j)),g=-2*D*alpha/(1-alpha);bool cap=log(v.A[j])-.5*v.E[j]>log(CAP);seed[j*4]=0;seed[j*4+1]=0;seed[j*4+2]=cap?0:-.5*g;seed[j*4+3]=cap?0:g/v.A[j];}return;}
 if(!(left<0&&right>0))return;double t=lo;
 for(int it=0;it<100;++it){double mid=.5*lo+.5*hi;if(mid==lo||mid==hi){t=mid;break;}double value=warp_residual(v,b,e,mid);if(!isfinite(value))return;t=mid;if(lane==0)o[4]=it+1;if(value<0)lo=mid;else if(value>0)hi=mid;else break;}
 Sum history;double scale=-INFINITY;
 for(int j=b+lane;j<e;j+=32){double x=(t-v.p[j])*v.rho[j],la=logalpha(v,j),alpha=exp(la),lg=la-.5*x*x,u=exp(lg);if(t>v.p[j])history.add(log1p(alpha/(1-alpha)*(-expm1(-.5*x*x))));if(x!=0)scale=fmax(scale,lg+log(fabs(x))+log(v.rho[j])-LN2-log1p(-u));}
 double D=warp_sum(history);scale=warp_max(scale);if(lane==0){o[0]=1;o[1]=t;o[2]=D;loss[ray]=D*D;}
 if(!isfinite(scale)){if(D==0){for(int j=b+lane;j<e;j+=32)for(int k=0;k<4;++k)seed[j*4+k]=0;if(lane==0)o[0]=4;}else if(lane==0){o[0]=-2;loss[ray]=NAN;}return;}
 Sum after,before;
 for(int j=b+lane;j<e;j+=32){double x=(t-v.p[j])*v.rho[j];if(x==0)continue;double lg=logalpha(v,j)-.5*x*x,u=exp(lg),w=exp(lg+log(fabs(x))+log(v.rho[j])-LN2-log1p(-u)-scale);if(t>v.p[j])after.add(w);else before.add(w);}
 double A=warp_sum(after),B=warp_sum(before),den=A+B,r=(A-B)/den;
 for(int j=b+lane;j<e;j+=32){double x=(t-v.p[j])*v.rho[j],la=logalpha(v,j),alpha=exp(la),u=exp(la-.5*x*x),bb=u/(2*(1-u));bool passed=t>v.p[j],cap=log(v.A[j])-.5*v.E[j]>log(CAP);double factor=2*(passed?B:A)/den,c=bb*factor,g=2*D*(-c-r*(passed?alpha/(1-alpha):0));seed[j*4]=-2*D*c*x*v.rho[j];seed[j*4+1]=2*D*c*x*x/v.rho[j];seed[j*4+2]=cap?0:-.5*g;seed[j*4+3]=cap?0:g/v.A[j];}
}

std::vector<torch::Tensor> solve(torch::Tensor p,torch::Tensor rho,torch::Tensor E,torch::Tensor A,torch::Tensor sections){
 TORCH_CHECK(p.is_cuda()&&p.scalar_type()==torch::kFloat64&&p.is_contiguous(),"p CUDA contiguous float64 required");
 for(auto x:{rho,E,A})TORCH_CHECK(x.is_cuda()&&x.device()==p.device()&&x.scalar_type()==torch::kFloat64&&x.is_contiguous()&&x.numel()==p.numel(),"field mismatch");
 TORCH_CHECK(sections.is_cuda()&&sections.device()==p.device()&&sections.scalar_type()==torch::kInt64&&sections.is_contiguous(),"sections mismatch");
 TORCH_CHECK(p.numel()<=2147483647 && sections.numel()>=2,"index or empty batch"); c10::cuda::CUDAGuard guard(p.device());int M=sections.numel()-1;auto loss=torch::empty({M},p.options()),seed=torch::empty({p.numel(),4},p.options()),info=torch::empty({M,8},p.options());
 gap_kernel<<<M,32,0,at::cuda::getCurrentCUDAStream()>>>({p.data_ptr<double>(),rho.data_ptr<double>(),E.data_ptr<double>(),A.data_ptr<double>()},sections.data_ptr<int64_t>(),M,loss.data_ptr<double>(),seed.data_ptr<double>(),info.data_ptr<double>());C10_CUDA_KERNEL_LAUNCH_CHECK();return{loss,seed,info};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("solve",&solve);}
