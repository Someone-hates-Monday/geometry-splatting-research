// Derivative of v22 replay. Pinned TGS window/compositing provenance/licenses:
// sources/tgs_native_diagnostic_v1/native/{LICENSE.md,LICENSE_GS.md}.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <climits>
struct G{const float *depth,*normal,*offset,*phi,*conic,*sigma;const int *radii,*lo,*hi;};
G geom(const std::vector<torch::Tensor>& v){TORCH_CHECK(v.size()==9);return G{v[0].data_ptr<float>(),v[1].data_ptr<float>(),v[2].data_ptr<float>(),v[3].data_ptr<float>(),v[4].data_ptr<float>(),v[5].data_ptr<float>(),v[6].data_ptr<int>(),v[7].data_ptr<int>(),v[8].data_ptr<int>()};}
__device__ bool inside(G g,int id,int x,int y){return g.radii[id]>0&&x>=g.lo[2*id]&&x<g.hi[2*id]&&y>=g.lo[2*id+1]&&y<g.hi[2*id+1];}
__noinline__ __device__ bool alpha_at(G g,int id,int X,int Y,float& a){
 float m=-INFINITY;for(int k=0;k<3;++k){float d=g.normal[6*id+2*k]*(float)X+g.normal[6*id+2*k+1]*(float)Y+g.offset[3*id+k];if(d>0)return false;m=fmaxf(m,d);}
 float cx=fmaxf(0.0f,__powf(m*g.phi[2*id],g.sigma[id]));a=fminf(.99f,g.conic[4*id+3]*cx);return a>=1.0f/255.0f;
}
__device__ void add64(int64_t* p,int64_t x){atomicAdd((unsigned long long*)p,(unsigned long long)x);}
__device__ int child_list(G c,int event,int x,int y,int* ids){
 int n=0;for(int k=0;k<4;++k){int id=4*event+k;if(!inside(c,id,x,y))continue;int at=n;
  while(at>0&&__float_as_uint(c.depth[ids[at-1]])>__float_as_uint(c.depth[id])){ids[at]=ids[at-1];--at;}ids[at]=id;++n;}
 return n;
}
__global__ void emit_kernel(G b,G c,const int* br,const int* bl,const float* bc,const float* cc,const int* parents,
 const int* box,const int64_t* offsets,int K,const int* pixel,const int64_t* patch_offsets,const float* original,int W,int H,float* output,int64_t* stats){
 __shared__ int event,tile,nchild,ids[4],X0,Y0,active;int rank=threadIdx.y*16+threadIdx.x;
 if(rank==0){int lo=0,hi=K;while(lo+1<hi){int m=(lo+hi)/2;if(offsets[m]<=(int64_t)blockIdx.x)lo=m;else hi=m;}event=lo;
  int width=box[4*event+2]-box[4*event];int at=(int)((int64_t)blockIdx.x-offsets[event]);int x=box[4*event]+at%width,y=box[4*event+1]+at/width;
  tile=y*((W+15)/16)+x;X0=x*16;Y0=y*16;nchild=child_list(c,event,x,y,ids);active=inside(b,parents[event],x,y)||nchild>0;
  for(int j=0;j<nchild;++j)if(!isfinite(c.depth[ids[j]])||c.depth[ids[j]]<=0)add64(stats+1,1);
 }
 __syncthreads();int X=X0+threadIdx.x,Y=Y0+threadIdx.y;
 if(X<W&&Y<H){float C[3]={0,0,0};
  if(active){int i=br[2*tile],be=br[2*tile+1],j=0,parent=parents[event];float T=1;
   while(i<be||j<nchild){while(i<be&&bl[i]==parent)++i;if(i>=be&&j>=nchild)break;
    bool from_base=j>=nchild||(i<be&&__float_as_uint(b.depth[bl[i]])<=__float_as_uint(c.depth[ids[j]]));int id=from_base?bl[i++]:ids[j++];
    float a;if(!alpha_at(from_base?b:c,id,X,Y,a))continue;float test=T*(1-a);if(test<.0001f)break;
    const float* color=(from_base?bc:cc)+3*id;for(int ch=0;ch<3;++ch)C[ch]+=color[ch]*a*T;T=test;
   }
  }else for(int ch=0;ch<3;++ch)C[ch]=original[(int64_t)ch*W*H+Y*W+X];
  const int* r=pixel+10*event;int rw=r[2]-r[0],rh=r[3]-r[1];
  for(int ch=0;ch<3;++ch){if(!isfinite(C[ch]))add64(stats,1);output[patch_offsets[event]+(int64_t)ch*rw*rh+(Y-r[1])*rw+X-r[0]]=C[ch];}
 }
}
std::vector<torch::Tensor> emit(std::vector<torch::Tensor>b,torch::Tensor br,torch::Tensor bl,torch::Tensor bc,std::vector<torch::Tensor>c,torch::Tensor cc,
 torch::Tensor parents,torch::Tensor boxes,torch::Tensor offsets,torch::Tensor pixel,torch::Tensor patch_offsets,torch::Tensor original,int W,int H,int64_t total_values){
 TORCH_CHECK(W>0&&H>0&&(int64_t)W*H<=INT_MAX/3&&parents.numel()>0&&total_values>=0);
 c10::cuda::CUDAGuard guard(original.device());int64_t total=offsets[-1].item<int64_t>();TORCH_CHECK(total>=0&&total<=INT_MAX&&pixel.size(0)==parents.numel()&&pixel.size(1)==10);
 auto output=torch::empty({total_values},original.options()),stats=torch::zeros({2},parents.options().dtype(torch::kInt64));cudaStream_t stream=c10::cuda::getCurrentCUDAStream();
 if(total)emit_kernel<<<(unsigned)total,dim3(16,16),0,stream>>>(geom(b),geom(c),br.data_ptr<int>(),bl.data_ptr<int>(),bc.data_ptr<float>(),cc.data_ptr<float>(),parents.data_ptr<int>(),boxes.data_ptr<int>(),offsets.data_ptr<int64_t>(),parents.numel(),pixel.data_ptr<int>(),patch_offsets.data_ptr<int64_t>(),original.data_ptr<float>(),W,H,output.data_ptr<float>(),stats.data_ptr<int64_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,stats};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("emit",&emit);}
