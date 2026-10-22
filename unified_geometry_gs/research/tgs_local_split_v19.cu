// Native TGS window/compositing provenance: sources/tgs_native_diagnostic_v1/native/
// LICENSE.md and LICENSE_GS.md. Cached counterfactual reference, no backward.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
// Geometry vector: depth,normal,offset,phi,conic,sigma,radii,rect_min,rect_max.
struct G{const float *depth,*normal,*offset,*phi,*conic,*sigma;const int *radii,*lo,*hi;};
G geometry(const std::vector<torch::Tensor>& v){
 TORCH_CHECK(v.size()==9);return G{v[0].data_ptr<float>(),v[1].data_ptr<float>(),v[2].data_ptr<float>(),
 v[3].data_ptr<float>(),v[4].data_ptr<float>(),v[5].data_ptr<float>(),v[6].data_ptr<int>(),v[7].data_ptr<int>(),v[8].data_ptr<int>()};
}
__device__ bool inside(G g,int id,int x,int y){return g.radii[id]>0&&x>=g.lo[2*id]&&x<g.hi[2*id]&&y>=g.lo[2*id+1]&&y<g.hi[2*id+1];}
__noinline__ __device__ bool alpha_at(G g,int id,int X,int Y,float& a){
 float m=-INFINITY;
 for(int k=0;k<3;++k){float d=g.normal[6*id+2*k]*(float)X+g.normal[6*id+2*k+1]*(float)Y+g.offset[3*id+k];
  if(d>0)return false;m=fmaxf(m,d);}
 float cx=fmaxf(0.0f,__powf(m*g.phi[2*id],g.sigma[id]));a=fminf(.99f,g.conic[4*id+3]*cx);
 return a>=1.0f/255.0f;
}
__global__ void mask_kernel(G base,G child,int parent,int tx,int ty,int* mask){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=tx*ty)return;int x=t%tx,y=t/tx;bool hit=inside(base,parent,x,y);
 for(int j=0;j<4;++j)hit=hit||inside(child,j,x,y);mask[t]=hit;
}
__device__ bool take_base(G base,G child,const int* bl,const int* cl,int i,int be,int j,int ce){
 if(j>=ce)return true;if(i>=be)return false;
 return __float_as_uint(base.depth[bl[i]])<=__float_as_uint(child.depth[cl[j]]);
}
__global__ void query_kernel(G base,G child,const int* br,const int* bl,const int* cr,const int* cl,
 const float* bc,const float* cc,int parent,int W,int H,const int* mask,float* rgb,float* alpha,int* paths){
 int tile=blockIdx.x;if(!mask[tile])return;
 int X=(tile%((W+15)/16))*16+threadIdx.x,Y=(tile/((W+15)/16))*16+threadIdx.y;if(X>=W||Y>=H)return;
 int i=br[2*tile],be=br[2*tile+1],j=cr[2*tile],ce=cr[2*tile+1];float T=1,C[3]={0,0,0};int ordinal=0,last=0,median=-1;
 while(i<be||j<ce){
  while(i<be&&bl[i]==parent)++i;if(i>=be&&j>=ce)break;
  bool from_base=take_base(base,child,bl,cl,i,be,j,ce);int id=from_base?bl[i++]:cl[j++];++ordinal;
  float a;if(!alpha_at(from_base?base:child,id,X,Y,a))continue;
  float test=T*(1-a);if(test<.0001f)break;
  if(T>.5f)median=ordinal;const float* color=(from_base?bc:cc)+3*id;
  for(int ch=0;ch<3;++ch)C[ch]+=color[ch]*a*T;T=test;last=ordinal;
 }
 int pix=Y*W+X,HW=W*H;for(int ch=0;ch<3;++ch)rgb[ch*HW+pix]=C[ch];alpha[pix]=1-T;paths[pix]=last;paths[HW+pix]=median;
}
__device__ void inc(int64_t* s,int k){atomicAdd((unsigned long long*)(s+k),1ULL);}
__global__ void audit_kernel(G base,G child,const int* br,const int* bl,const int* cr,const int* cl,
 const int* reference_range,const int* reference_list,int parent,int N,int tiles,int64_t* stats){
 int tile=blockIdx.x*blockDim.x+threadIdx.x;if(tile>=tiles)return;
 int i=br[2*tile],be=br[2*tile+1],j=cr[2*tile],ce=cr[2*tile+1],at=reference_range[2*tile],end=reference_range[2*tile+1];
 unsigned prev=0;bool first=true;
 while(i<be||j<ce){while(i<be&&bl[i]==parent)++i;if(i>=be&&j>=ce)break;
  bool from_base=take_base(base,child,bl,cl,i,be,j,ce);int id=from_base?bl[i++]:cl[j++];
  float d=(from_base?base:child).depth[id];unsigned key=__float_as_uint(d);
  if(!isfinite(d)||d<=0)inc(stats,0);if(!first&&key<prev)inc(stats,1);prev=key;first=false;
  int target=from_base?(id<parent?id:id-1):N-1+id;
  if(at>=end||reference_list[at]!=target)inc(stats,2);++at;inc(stats,4);
 }
 if(at!=end)inc(stats,3);
}
std::vector<torch::Tensor> query(std::vector<torch::Tensor> b,torch::Tensor br,torch::Tensor bl,torch::Tensor bc,
 std::vector<torch::Tensor> c,torch::Tensor cr,torch::Tensor cl,torch::Tensor cc,int parent,
 torch::Tensor oldrgb,torch::Tensor oldalpha,torch::Tensor oldpaths,int W,int H){
 auto rgb=oldrgb.clone(),alpha=oldalpha.clone(),paths=oldpaths.clone();int tx=(W+15)/16,ty=(H+15)/16;
 auto mask=torch::zeros({tx*ty},br.options());G bg=geometry(b),cg=geometry(c);
 mask_kernel<<<(tx*ty+255)/256,256>>>(bg,cg,parent,tx,ty,mask.data_ptr<int>());
 query_kernel<<<tx*ty,dim3(16,16)>>>(bg,cg,br.data_ptr<int>(),bl.data_ptr<int>(),cr.data_ptr<int>(),cl.data_ptr<int>(),
  bc.data_ptr<float>(),cc.data_ptr<float>(),parent,W,H,mask.data_ptr<int>(),rgb.data_ptr<float>(),alpha.data_ptr<float>(),paths.data_ptr<int>());
 return {rgb,alpha,paths,mask};
}
torch::Tensor audit(std::vector<torch::Tensor> b,torch::Tensor br,torch::Tensor bl,
 std::vector<torch::Tensor> c,torch::Tensor cr,torch::Tensor cl,torch::Tensor rr,torch::Tensor rl,int parent,int N,int tiles){
 auto stats=torch::zeros({8},br.options().dtype(torch::kInt64));
 audit_kernel<<<(tiles+127)/128,128>>>(geometry(b),geometry(c),br.data_ptr<int>(),bl.data_ptr<int>(),cr.data_ptr<int>(),cl.data_ptr<int>(),
  rr.data_ptr<int>(),rl.data_ptr<int>(),parent,N,tiles,stats.data_ptr<int64_t>());return stats;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("query",&query);m.def("audit",&audit);}
