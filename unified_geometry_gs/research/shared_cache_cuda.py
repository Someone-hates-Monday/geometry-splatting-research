"""Split the frozen preprocessing code at its query-independent boundary."""
from pathlib import Path
import hashlib
def once(t,a,b):
    assert t.count(a)==1,(a,t.count(a))
    return t.replace(a,b)

def cache_header(base):
    start=base.index('__device__ __forceinline__ void decode(')
    cut=base.index(' I rawq[3]=',start)
    end=base.index('\n__global__ void preprocess(',cut)
    prefix=base[start:cut]
    prefix=once(prefix,'void decode(','void budget(')
    prefix+=' out[0]=0;for(int i=0;i<6;i++)out[14+i]=u[upper[i]];for(int i=0;i<3;i++)out[20+i]=mean[i];\n}\n'
    suffix=base[cut:end]  # Original ray decode statements and closing brace.
    imports='using gp::root;using gp::dot;using gp::norm;using gp::norm_upper;using gp::mm;using gp::mv;using gp::inverse;using gp::midpoint;using gp::ndot;using gp::qr;using gp::rotation;\n'
    head='''#pragma once
#include "guarded_preprocess.cuh"
namespace gc {
using gi::I;using gi::pt;using gi::add;using gi::sub;using gi::mul;using gi::div;using gi::sq;
constexpr int CACHE_WIDTH=28;
constexpr int WIDTH=26;
'''+imports
    build=r'''
__device__ __forceinline__ void build_one(const double* src,double* c){
 I A[9],R[9],Ai[9],Ri[9],f[9],mu[3],m[3]={pt(src[0]),pt(src[1]),pt(src[2])};
 for(int i=0;i<9;i++)A[i]=pt(src[20+i]);rotation(src,R);
 if(!inverse(A,Ai)||!inverse(R,Ri)){c[0]=c[1]=1;return;}
 mm(Ri,Ai,f);for(int i=0;i<3;i++)for(int j=0;j<3;j++)f[3*i+j]=div(f[3*i+j],pt(src[3+i]));
 mv(A,m,mu);for(int i=0;i<3;i++)mu[i]=add(mu[i],pt(src[29+i]));
 double fnom[9],q[9],u[9];for(int i=0;i<9;i++)fnom[i]=midpoint(f[i]);
 if(!qr(fnom,q,u)){c[0]=c[1]=2;return;}
 for(int i=0;i<3;i++){c[9+2*i]=mu[i].lo;c[10+2*i]=mu[i].hi;}
 for(int mode=0;mode<2;mode++){
   double out[26];for(int i=0;i<26;i++)out[i]=NAN;budget(src,f,mu,q,u,mode,out);
   c[mode]=out[0];c[2]=out[23];if(out[0]!=0)continue;
   c[3+mode]=out[11];c[5]=out[12];
   for(int i=0;i<3;i++)c[6+i]=out[20+i];
   for(int i=0;i<6;i++)c[15+6*mode+i]=out[14+i];
 }
 c[27]=0;
}
__global__ void build(int n,const double* source,double* cache){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;build_one(source+36*i,cache+CACHE_WIDTH*i);
}
'''
    decode='''__device__ __forceinline__ void decode_one(const double* src,const double* c,int mode,double* out){
 out[0]=5;out[23]=c[2];if(c[mode]!=0){out[0]=c[mode];return;}
 I uv[9],mu[3];double u[9]={0},mean[3];int upper[6]={0,1,2,4,5,8};
 for(int i=0;i<6;i++)u[upper[i]]=c[15+6*mode+i];for(int i=0;i<9;i++)uv[i]=pt(u[i]);
 for(int i=0;i<3;i++){mean[i]=c[6+i];mu[i]={c[9+2*i],c[10+2*i]};}
 double eta=c[3+mode],nu=c[5];out[11]=eta;out[12]=nu;
'''+suffix
    query=r'''
__global__ void decode(int n,const double* cache,const int* ids,const double* queries,double* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 double src[36];for(int j=0;j<4;j++)src[32+j]=queries[4*i+j];
 const double* c=cache+CACHE_WIDTH*ids[i];for(int mode=0;mode<2;mode++)decode_one(src,c,mode,output+(2*i+mode)*WIDTH);
}
} // namespace gc
'''
    return head+prefix+build+decode+query

def load(out,old):
    import torch
    from research.guarded_interval_cuda import FLAGS
    from torch.utils.cpp_extension import load as compile_extension
    b=out/'generated';b.mkdir()
    base={n:(old/'variant_generated'/n).read_text(encoding='utf-8') for n in ['guarded_interval.cuh','guarded_preprocess.cuh','guarded_root.cuh']}
    base['shared_cache.cuh']=cache_header(base['guarded_preprocess.cuh'])
    base['cache.cu']=r'''
#include "shared_cache.cuh"
#include "guarded_root.cuh"
void original(const double* a,double* b,int n,cudaStream_t s){gp::preprocess<<<(n+63)/64,64,0,s>>>(n,a,b);}
void build(const double* a,double* b,int n,cudaStream_t s){gc::build<<<(n+63)/64,64,0,s>>>(n,a,b);}
void decode(const double* a,const int* b,const double* c,double* d,int n,cudaStream_t s){gc::decode<<<(n+63)/64,64,0,s>>>(n,a,b,c,d);}
void solve(const double* a,const int* b,double* c,int n,cudaStream_t s){gr::solve<<<(n+31)/32,32,0,s>>>(n,a,b,c);}
'''
    base['binding.cpp']=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void original(const double*,double*,int,cudaStream_t);void build(const double*,double*,int,cudaStream_t);
void decode(const double*,const int*,const double*,double*,int,cudaStream_t);void solve(const double*,const int*,double*,int,cudaStream_t);
void floating(torch::Tensor a){TORCH_CHECK(a.is_cuda()&&a.is_contiguous()&&a.scalar_type()==torch::kFloat64,"CUDA contiguous double required");}
torch::Tensor pre(torch::Tensor a){floating(a);TORCH_CHECK(a.dim()==2&&a.size(1)==36,"Source rows36 required");c10::cuda::CUDAGuard guard(a.device());auto o=torch::full({a.size(0),2,26},NAN,a.options());original(a.data_ptr<double>(),o.data_ptr<double>(),a.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
torch::Tensor make(torch::Tensor a){floating(a);TORCH_CHECK(a.dim()==2&&a.size(1)==36,"Source rows36 required");c10::cuda::CUDAGuard guard(a.device());auto o=torch::full({a.size(0),28},NAN,a.options());build(a.data_ptr<double>(),o.data_ptr<double>(),a.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
torch::Tensor query(torch::Tensor cache,torch::Tensor ids,torch::Tensor q){
 floating(cache);floating(q);TORCH_CHECK(cache.dim()==2&&cache.size(1)==28&&q.dim()==2&&q.size(1)==4&&ids.is_cuda()&&ids.device()==cache.device()&&q.device()==cache.device()&&ids.is_contiguous()&&ids.scalar_type()==torch::kInt32&&ids.dim()==1&&ids.size(0)==q.size(0),"Cache/query/IDs invalid");
 TORCH_CHECK(ids.numel()==0||(ids.min().item<int>()>=0&&ids.max().item<int>()<cache.size(0)),"IDs out of cache range");
 c10::cuda::CUDAGuard guard(cache.device());auto o=torch::full({q.size(0),2,26},NAN,q.options());decode(cache.data_ptr<double>(),ids.data_ptr<int>(),q.data_ptr<double>(),o.data_ptr<double>(),q.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return o;
}
torch::Tensor root(torch::Tensor a,torch::Tensor b){floating(a);TORCH_CHECK(a.dim()==3&&a.size(1)==2&&a.size(2)==26&&b.is_cuda()&&b.device()==a.device()&&b.is_contiguous()&&b.scalar_type()==torch::kInt32&&b.dim()==2&&b.size(1)==2,"Root inputs invalid");c10::cuda::CUDAGuard guard(a.device());auto o=torch::full({b.size(0),12},NAN,a.options());solve(a.data_ptr<double>(),b.data_ptr<int>(),o.data_ptr<double>(),b.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("preprocess",&pre);m.def("build",&make);m.def("decode",&query);m.def("solve",&root);}
'''
    h=hashlib.sha256((torch.__version__+str(FLAGS)).encode())
    for n,t in sorted(base.items()):h.update(n.encode());h.update(t.encode());(b/n).write_text(t,encoding='utf-8')
    return compile_extension(name='shared_cache_'+h.hexdigest()[:12],sources=[str(b/'binding.cpp'),str(b/'cache.cu')],extra_cflags=['-O3'],extra_cuda_cflags=FLAGS,build_directory=str(b),verbose=True)
