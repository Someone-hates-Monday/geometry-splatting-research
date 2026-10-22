// Pinned TGS window/compositing provenance and licenses:
// sources/tgs_native_diagnostic_v1/native/{LICENSE.md,LICENSE_GS.md}.
#include <torch/extension.h>
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
__device__ int png(float x){float y=__fadd_rn(__fmul_rn(x,255.0f),.5f);return __float2int_rz(fminf(255.0f,fmaxf(0.0f,y)));}
__device__ void add64(int64_t* p,int64_t x){atomicAdd((unsigned long long*)p,(unsigned long long)x);}
__device__ int child_list(G c,int event,int x,int y,int* ids){
 int n=0;for(int k=0;k<4;++k){int id=4*event+k;if(!inside(c,id,x,y))continue;int at=n;
  while(at>0&&__float_as_uint(c.depth[ids[at-1]])>__float_as_uint(c.depth[id])){ids[at]=ids[at-1];--at;}ids[at]=id;++n;}
 return n;
}
__global__ void plan_kernel(G b,G c,const int* parents,int K,int tx,int ty,int* box,int64_t* counts){
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=K)return;int x0=tx,y0=ty,x1=0,y1=0;
 for(int j=-1;j<4;++j){G g=j<0?b:c;int id=j<0?parents[e]:4*e+j;if(g.radii[id]<=0)continue;
  x0=min(x0,g.lo[2*id]);y0=min(y0,g.lo[2*id+1]);x1=max(x1,g.hi[2*id]);y1=max(y1,g.hi[2*id+1]);}
 if(x1<=x0||y1<=y0)x0=y0=x1=y1=0;
 box[4*e]=x0;box[4*e+1]=y0;box[4*e+2]=x1;box[4*e+3]=y1;counts[e]=(int64_t)(x1-x0)*(y1-y0);
}
__global__ void score_kernel(G b,G c,const int* br,const int* bl,const float* bc,const float* cc,const int* parents,
 const int* box,const int64_t* offsets,int K,int W,int H,const unsigned char* base8,const unsigned char* gt,int64_t* score,int64_t* stats,
 int debug,float* rgb,float* alpha,int* paths){
 __shared__ int event,tile,nchild,ids[4],X0,Y0,active,sum[256];int rank=threadIdx.y*16+threadIdx.x;
 if(rank==0){int lo=0,hi=K;while(lo+1<hi){int m=(lo+hi)/2;if(offsets[m]<=(int64_t)blockIdx.x)lo=m;else hi=m;}event=lo;
  int width=box[4*event+2]-box[4*event];int at=(int)((int64_t)blockIdx.x-offsets[event]);int x=box[4*event]+at%width,y=box[4*event+1]+at/width;
  tile=y*((W+15)/16)+x;X0=x*16;Y0=y*16;nchild=child_list(c,event,x,y,ids);active=inside(b,parents[event],x,y)||nchild>0;
  for(int j=0;j<nchild;++j)if(!isfinite(c.depth[ids[j]])||c.depth[ids[j]]<=0)add64(stats+1,1);
 }
 __syncthreads();if(!active)return;int X=X0+threadIdx.x,Y=Y0+threadIdx.y;int value=0;
 if(X<W&&Y<H){
  int i=br[2*tile],be=br[2*tile+1],j=0,parent=parents[event],ordinal=0,last=0,median=-1;float T=1,C[3]={0,0,0};
  while(i<be||j<nchild){while(i<be&&bl[i]==parent)++i;if(i>=be&&j>=nchild)break;
   bool from_base=j>=nchild||(i<be&&__float_as_uint(b.depth[bl[i]])<=__float_as_uint(c.depth[ids[j]]));int id=from_base?bl[i++]:ids[j++];++ordinal;
   float a;if(!alpha_at(from_base?b:c,id,X,Y,a))continue;float test=T*(1-a);if(test<.0001f)break;
   if(T>.5f)median=ordinal;const float* color=(from_base?bc:cc)+3*id;for(int ch=0;ch<3;++ch)C[ch]+=color[ch]*a*T;T=test;last=ordinal;
  }
  int pix=Y*W+X,HW=W*H;
  for(int ch=0;ch<3;++ch){if(!isfinite(C[ch]))add64(stats,1);int q=png(C[ch]),g=gt[ch*HW+pix],old=base8[ch*HW+pix];value+=(q-g)*(q-g)-(old-g)*(old-g);
   if(event==debug)rgb[ch*HW+pix]=C[ch];}
  if(event==debug){alpha[pix]=1-T;paths[pix]=last;paths[HW+pix]=median;}
 }
 sum[rank]=value;__syncthreads();for(int stride=128;stride>0;stride/=2){if(rank<stride)sum[rank]+=sum[rank+stride];__syncthreads();}
 if(rank==0&&sum[0])add64(score+event,(int64_t)sum[0]);
}
__global__ void quant_kernel(const float* rgb,int N,unsigned char* out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)out[i]=(unsigned char)png(rgb[i]);}
__global__ void delta_kernel(const float* rgb,const unsigned char* base,const unsigned char* gt,int HW,int64_t* out){
 __shared__ int sum[256];int t=threadIdx.x,p=blockIdx.x*blockDim.x+t,v=0;
 if(p<HW)for(int ch=0;ch<3;++ch){int q=png(rgb[ch*HW+p]),g=gt[ch*HW+p],b=base[ch*HW+p];v+=(q-g)*(q-g)-(b-g)*(b-g);}
 sum[t]=v;__syncthreads();for(int s=128;s>0;s/=2){if(t<s)sum[t]+=sum[t+s];__syncthreads();}if(t==0&&sum[0])add64(out,(int64_t)sum[0]);
}
__global__ void list_audit_kernel(G b,G c,const int* br,const int* bl,const int* parents,int event,const int* rr,const int* rl,int N,int tiles,int tx,int64_t* stats){
 int tile=blockIdx.x*blockDim.x+threadIdx.x;if(tile>=tiles)return;int ids[4],nc=child_list(c,event,tile%tx,tile/tx,ids);
 int i=br[2*tile],be=br[2*tile+1],j=0,at=rr[2*tile],end=rr[2*tile+1],parent=parents[event];unsigned previous=0;bool first=true;
 while(i<be||j<nc){while(i<be&&bl[i]==parent)++i;if(i>=be&&j>=nc)break;
  bool from_base=j>=nc||(i<be&&__float_as_uint(b.depth[bl[i]])<=__float_as_uint(c.depth[ids[j]]));int id=from_base?bl[i++]:ids[j++];
  float d=(from_base?b:c).depth[id];unsigned key=__float_as_uint(d);if(!isfinite(d)||d<=0)add64(stats,1);if(!first&&key<previous)add64(stats+1,1);previous=key;first=false;
  int target=from_base?(id<parent?id:id-1):N-1+(id-4*event);if(at>=end||rl[at]!=target)add64(stats+2,1);++at;add64(stats+4,1);
 }if(at!=end)add64(stats+3,1);
}
std::vector<torch::Tensor> plan(std::vector<torch::Tensor>b,std::vector<torch::Tensor>c,torch::Tensor parents,int W,int H){
 TORCH_CHECK(parents.numel()>0&&W>0&&H>0&&c[0].numel()==4*parents.numel());
 auto boxes=torch::zeros({parents.numel(),4},parents.options()),counts=torch::zeros({parents.numel()},parents.options().dtype(torch::kInt64));
 plan_kernel<<<(parents.numel()+127)/128,128>>>(geom(b),geom(c),parents.data_ptr<int>(),parents.numel(),(W+15)/16,(H+15)/16,boxes.data_ptr<int>(),counts.data_ptr<int64_t>());return {boxes,counts};
}
std::vector<torch::Tensor> score(std::vector<torch::Tensor>b,torch::Tensor br,torch::Tensor bl,torch::Tensor bc,std::vector<torch::Tensor>c,torch::Tensor cc,
 torch::Tensor parents,torch::Tensor boxes,torch::Tensor offsets,torch::Tensor base8,torch::Tensor gt,torch::Tensor oldrgb,torch::Tensor oldalpha,torch::Tensor oldpaths,int W,int H,int debug){
 TORCH_CHECK(W>0&&H>0&&(int64_t)W*H<=INT_MAX/3);TORCH_CHECK(base8.scalar_type()==torch::kUInt8&&gt.scalar_type()==torch::kUInt8);
 int64_t total=offsets[-1].item<int64_t>();TORCH_CHECK(total<=INT_MAX&&debug<(int)parents.numel());
 auto sums=torch::zeros({parents.numel()},parents.options().dtype(torch::kInt64)),stats=torch::zeros({4},sums.options());
 auto rgb=debug>=0?oldrgb.clone():torch::empty({0},oldrgb.options()),alpha=debug>=0?oldalpha.clone():torch::empty({0},oldalpha.options()),paths=debug>=0?oldpaths.clone():torch::empty({0},oldpaths.options());
 if(total)score_kernel<<<(unsigned)total,dim3(16,16)>>>(geom(b),geom(c),br.data_ptr<int>(),bl.data_ptr<int>(),bc.data_ptr<float>(),cc.data_ptr<float>(),parents.data_ptr<int>(),boxes.data_ptr<int>(),offsets.data_ptr<int64_t>(),parents.numel(),W,H,
  base8.data_ptr<unsigned char>(),gt.data_ptr<unsigned char>(),sums.data_ptr<int64_t>(),stats.data_ptr<int64_t>(),debug,rgb.data_ptr<float>(),alpha.data_ptr<float>(),paths.data_ptr<int>());
 return {sums,stats,rgb,alpha,paths};
}
torch::Tensor quantize(torch::Tensor rgb){auto out=torch::empty_like(rgb,rgb.options().dtype(torch::kUInt8));quant_kernel<<<(rgb.numel()+255)/256,256>>>(rgb.data_ptr<float>(),rgb.numel(),out.data_ptr<unsigned char>());return out;}
torch::Tensor delta(torch::Tensor rgb,torch::Tensor base,torch::Tensor gt){int HW=rgb.numel()/3;auto out=torch::zeros({1},rgb.options().dtype(torch::kInt64));delta_kernel<<<(HW+255)/256,256>>>(rgb.data_ptr<float>(),base.data_ptr<unsigned char>(),gt.data_ptr<unsigned char>(),HW,out.data_ptr<int64_t>());return out;}
torch::Tensor audit(std::vector<torch::Tensor>b,torch::Tensor br,torch::Tensor bl,std::vector<torch::Tensor>c,torch::Tensor parents,int event,torch::Tensor rr,torch::Tensor rl,int N,int W,int H){
 int tx=(W+15)/16,tiles=tx*((H+15)/16);auto stats=torch::zeros({8},parents.options().dtype(torch::kInt64));
 list_audit_kernel<<<(tiles+127)/128,128>>>(geom(b),geom(c),br.data_ptr<int>(),bl.data_ptr<int>(),parents.data_ptr<int>(),event,rr.data_ptr<int>(),rl.data_ptr<int>(),N,tiles,tx,stats.data_ptr<int64_t>());return stats;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("plan",&plan);m.def("score",&score);m.def("quantize",&quantize);m.def("delta",&delta);m.def("audit",&audit);}
