"""Isolated F32 product endpoint diagnostic; frozen directed geometry."""
from pathlib import Path
from fractions import Fraction
import numpy as np
from research.csr_decode_cuda import compile_module,once

def recip_table():
    rows=['__device__ __forceinline__ I recip(int j){switch(j){']
    for j in range(1,10):
        x=np.float32(1/j);exact=Fraction(1,j);lo=hi=x
        if Fraction(float(x))>exact:lo=np.nextafter(x,np.float32(-np.inf))
        if Fraction(float(x))<exact:hi=np.nextafter(x,np.float32(np.inf))
        assert Fraction(float(lo))<=exact<=Fraction(float(hi))
        rows.append(f'case {j}:return {{{float(lo).hex()}f,{float(hi).hex()}f}};')
    return '\n'.join(rows+['default:return full();}}'])

def load(out,old):
    from research.guarded_interval_cuda import FLAGS
    r=Path(__file__).parent
    files={n:(old/'generated'/n).read_text(encoding='utf-8') for n in ['guarded_interval.cuh','guarded_preprocess.cuh','shared_cache.cuh']}
    body=files.pop('shared_cache.cuh').replace('namespace gc {','namespace fc {').replace('} // namespace gc','} // namespace fc')
    a='I alpha=mul(pt(src[35]),gi::exp(mul(pt(-.5),ebox)));double cap=double(float(.99));alpha.lo=fmax(0.,fmin(cap,alpha.lo));alpha.hi=fmin(fmin(src[35],cap),alpha.hi);out[24]=alpha.lo;out[25]=alpha.hi;'
    b='fp::I af=fp::alpha(src[35],ebox.lo,ebox.hi);I alpha={double(af.lo),double(af.hi)};out[24]=alpha.lo;out[25]=alpha.hi;'
    files['float_cache.cuh']='#include "float_product.cuh"\n'+once(body,a,b)
    files['float_product.cuh']=(r/'float_product.cuh').read_text(encoding='utf-8')
    files['float_reciprocals.cuh']=recip_table()
    files['kernel.cu']=r'''
#include "float_cache.cuh"
__global__ void certificate(const double* cache,const int* records,const int64_t* offsets,const float* query,const float* amp,const float* z,const float* eligible,double* out,int pixels,int P){
 int lane=threadIdx.x%32,pixel=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(pixel>=pixels)return;
 double* o=out+12*pixel;double reported=double(z[pixel]);int64_t begin=offsets[pixel],end=offsets[pixel+1];
 if(lane==0){o[0]=2;o[9]=double(end-begin);o[10]=0;o[11]=reported;}
 if(eligible[pixel]!=1||!(reported>0)||!isfinite(reported)||begin>=end)return;
 gi::I q[3]={gi::pt(double(query[2*pixel])),gi::pt(double(query[2*pixel+1])),gi::pt(1)};
 gi::I center=gi::mul(gi::pt(reported),gp::norm(q,3));const double tol=0x1.0624dd2f1a9fbp-10;
 double lo=__dsub_ru(center.hi,tol),hi=__dadd_rd(center.lo,tol);int bad=!(isfinite(lo)&&isfinite(hi)&&lo<hi);fp::I left=fp::pt(1),right=fp::pt(1);
 for(int64_t i=begin+lane;i<end;i+=32){
  int id=records[2*i];if(id<0||id>=P){bad=1;continue;}
  double src[36];src[32]=query[2*pixel];src[33]=query[2*pixel+1];src[34]=1;src[35]=amp[id];double v[26];for(int j=0;j<26;j++)v[j]=NAN;
  fc::decode_one(src,cache+28*id,0,v);left=fp::mul(left,fp::factor(v,lo,bad));right=fp::mul(right,fp::factor(v,hi,bad));
 }
 for(int stride=16;stride>0;stride/=2){fp::I l={__shfl_down_sync(0xffffffff,left.lo,stride),__shfl_down_sync(0xffffffff,left.hi,stride)},r={__shfl_down_sync(0xffffffff,right.lo,stride),__shfl_down_sync(0xffffffff,right.hi,stride)};int other=__shfl_down_sync(0xffffffff,bad,stride);if(lane<stride){left=fp::mul(left,l);right=fp::mul(right,r);bad|=other;}}
 if(lane==0){o[0]=!bad&&left.lo>.5f&&right.hi<.5f?1:0;o[1]=lo;o[2]=hi;o[3]=center.lo;o[4]=center.hi;o[5]=left.lo;o[6]=left.hi;o[7]=right.lo;o[8]=right.hi;o[10]=bad;}
}
__global__ void scalar(const float* x,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){fp::I a=fp::exp_one(x[i]);y[2*i]=a.lo;y[2*i+1]=a.hi;}}
__global__ void probe(const double* src,const double* cache,const int* ids,double* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)fc::decode_one(src+36*i,cache+28*ids[i],0,out+26*i);}
void launch(const double* a,const int* b,const int64_t* c,const float* d,const float* e,const float* z,const float* el,double* o,int n,int P,cudaStream_t s){certificate<<<(n*32+127)/128,128,0,s>>>(a,b,c,d,e,z,el,o,n,P);}
void scalar_launch(const float* x,float* y,int n,cudaStream_t s){scalar<<<(n+127)/128,128,0,s>>>(x,y,n);}
void probe_launch(const double* a,const double* c,const int* ids,double* o,int n,cudaStream_t s){probe<<<(n+63)/64,64,0,s>>>(a,c,ids,o,n);}
'''
    files['binding.cpp']=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void launch(const double*,const int*,const int64_t*,const float*,const float*,const float*,const float*,double*,int,int,cudaStream_t);
void scalar_launch(const float*,float*,int,cudaStream_t);
void probe_launch(const double*,const double*,const int*,double*,int,cudaStream_t);
torch::Tensor run(torch::Tensor c,torch::Tensor r,torch::Tensor o,torch::Tensor q,torch::Tensor a,torch::Tensor z,torch::Tensor e){
 for(auto t:{c,r,o,q,a,z,e})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==c.device(),"device/contiguous");
 TORCH_CHECK(c.scalar_type()==torch::kFloat64&&c.dim()==2&&c.size(1)==28&&r.scalar_type()==torch::kInt32&&r.dim()==2&&r.size(1)==2&&o.scalar_type()==torch::kInt64&&o.dim()==1&&q.scalar_type()==torch::kFloat32&&q.dim()==2&&q.size(1)==2,"cache/CSR/query");
 TORCH_CHECK(a.scalar_type()==torch::kFloat32&&a.numel()==c.size(0)&&z.scalar_type()==torch::kFloat32&&e.scalar_type()==torch::kFloat32&&z.numel()==q.size(0)&&e.numel()==q.size(0)&&o.numel()==q.size(0)+1,"shape/type");
 c10::cuda::CUDAGuard g(c.device());auto v=torch::full({q.size(0),12},NAN,c.options());launch(c.data_ptr<double>(),r.data_ptr<int>(),o.data_ptr<int64_t>(),q.data_ptr<float>(),a.data_ptr<float>(),z.data_ptr<float>(),e.data_ptr<float>(),v.data_ptr<double>(),q.size(0),c.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return v;
}
torch::Tensor scalars(torch::Tensor x){TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==torch::kFloat32&&x.dim()==1,"scalar input");c10::cuda::CUDAGuard g(x.device());auto y=torch::empty({x.numel(),2},x.options());scalar_launch(x.data_ptr<float>(),y.data_ptr<float>(),x.numel(),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
torch::Tensor probes(torch::Tensor src,torch::Tensor c,torch::Tensor ids){for(auto t:{src,c,ids})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==c.device(),"probe device");TORCH_CHECK(src.scalar_type()==torch::kFloat64&&src.dim()==2&&src.size(1)==36&&c.scalar_type()==torch::kFloat64&&c.dim()==2&&c.size(1)==28&&ids.scalar_type()==torch::kInt32&&ids.dim()==1&&ids.numel()==src.size(0),"probe type/shape");TORCH_CHECK(ids.numel()>0&&ids.min().item<int>()>=0&&ids.max().item<int>()<c.size(0),"probe IDs");c10::cuda::CUDAGuard g(c.device());auto v=torch::full({src.size(0),26},NAN,c.options());probe_launch(src.data_ptr<double>(),c.data_ptr<double>(),ids.data_ptr<int>(),v.data_ptr<double>(),src.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return v;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("scalars",&scalars);m.def("probes",&probes);}
'''
    return compile_module(out,'generated',files,FLAGS,'float_product_')
