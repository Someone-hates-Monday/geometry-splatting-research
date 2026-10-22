// Local research baseline. Native-window/compositing provenance:
// sources/tgs_native_diagnostic_v1/native/{LICENSE.md,LICENSE_GS.md}.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>

__noinline__ __device__ bool alpha_at(int id,int X,int Y,const float* normal,const float* offset,
 const float* phi,const float* sigma,const float* conic,float& alpha){
 float max_val=-INFINITY;
 for(int k=0;k<3;++k){
  float dist=normal[6*id+2*k]*(float)X+normal[6*id+2*k+1]*(float)Y+offset[3*id+k];
  if(dist>0)return false;max_val=fmaxf(max_val,dist);
 }
 float cx=fmaxf(0.0f,__powf(max_val*phi[2*id],sigma[id]));
 alpha=fminf(.99f,conic[4*id+3]*cx);
 if(alpha<1.0f/255.0f)return false;return true;
}
__device__ double ray_z(int id,int X,int Y,int W,int H,double tx,double ty,const double* plane){
 double rx=((2.0*X+1.0)/(double)W-1.0)*tx;
 double ry=((2.0*Y+1.0)/(double)H-1.0)*ty;
 const double* p=plane+4*id;return p[3]/(p[0]*rx+p[1]*ry+p[2]);
}
__device__ void add(long long* s,int k){atomicAdd((unsigned long long*)(s+k),1ULL);}

__global__ void count_kernel(const float* no,const float* off,const float* ph,const float* sig,const float* co,
 const int* ranges,const int* list,const double* plane,int W,int H,double tx,double ty,int* counts,long long* stats){
 int pix=blockIdx.x*blockDim.x+threadIdx.x;if(pix>=W*H)return;
 int X=pix%W,Y=pix/W,tile=(Y/16)*((W+15)/16)+X/16;int start=ranges[2*tile],end=ranges[2*tile+1],n=0;
 for(int j=start;j<end;++j){int id=list[j];float a;if(!alpha_at(id,X,Y,no,off,ph,sig,co,a))continue;
  ++n;double z=ray_z(id,X,Y,W,H,tx,ty,plane);
  if(!isfinite(a))add(stats,0);if(!isfinite(z)||z<=0)add(stats,1);
 }
 counts[pix]=n;
}
__global__ void fill_kernel(const float* no,const float* off,const float* ph,const float* sig,const float* co,
 const int* ranges,const int* list,const double* plane,int W,int H,double tx,double ty,const int* counts,
 const long long* offsets,int* ids,float* alpha,double* z,int* ordinal,int* rank,int* original,long long* stats){
 int pix=blockIdx.x*blockDim.x+threadIdx.x;if(pix>=W*H)return;
 int X=pix%W,Y=pix/W,tile=(Y/16)*((W+15)/16)+X/16,start=ranges[2*tile],end=ranges[2*tile+1],n=0;
 long long base=offsets[pix];
 for(int j=start;j<end;++j){int id=list[j];float a;if(!alpha_at(id,X,Y,no,off,ph,sig,co,a))continue;
  if(n>=counts[pix]){add(stats,2);continue;}long long at=base+n;
  ids[at]=id;alpha[at]=a;z[at]=ray_z(id,X,Y,W,H,tx,ty,plane);ordinal[at]=j-start+1;rank[at]=n;original[at]=j-start+1;++n;
 }
 if(n!=counts[pix])add(stats,2);
}
struct Hit{double z;float a;int id;int ord;int rank;};
__device__ Hit get(long long at,const int* ids,const float* a,const double* z,const int* ord,const int* rank){
 return Hit{z[at],a[at],ids[at],ord[at],rank[at]};
}
__device__ void put(long long at,Hit h,int* ids,float* a,double* z,int* ord,int* rank){
 z[at]=h.z;a[at]=h.a;ids[at]=h.id;ord[at]=h.ord;rank[at]=h.rank;
}
__device__ bool greater(Hit a,Hit b){return a.z>b.z||(a.z==b.z&&a.ord>b.ord);}
__device__ void sift(long long base,int n,int node,int* ids,float* a,double* z,int* ord,int* rank){
 while(true){int child=2*node+1;if(child>=n)return;
  if(child+1<n&&greater(get(base+child+1,ids,a,z,ord,rank),get(base+child,ids,a,z,ord,rank)))++child;
  Hit parent=get(base+node,ids,a,z,ord,rank),other=get(base+child,ids,a,z,ord,rank);
  if(!greater(other,parent))return;put(base+node,other,ids,a,z,ord,rank);put(base+child,parent,ids,a,z,ord,rank);node=child;
 }
}
__global__ void sort_kernel(const int* counts,const long long* offsets,int HW,int* ids,float* a,double* z,int* ord,int* rank){
 int pix=blockIdx.x*blockDim.x+threadIdx.x;if(pix>=HW)return;int n=counts[pix];long long base=offsets[pix];
 for(int node=n/2-1;node>=0;--node)sift(base,n,node,ids,a,z,ord,rank);
 for(int end=n-1;end>0;--end){Hit x=get(base,ids,a,z,ord,rank),y=get(base+end,ids,a,z,ord,rank);
  put(base,y,ids,a,z,ord,rank);put(base+end,x,ids,a,z,ord,rank);sift(base,end,0,ids,a,z,ord,rank);}
}
__global__ void verify_kernel(const float* no,const float* off,const float* ph,const float* sig,const float* co,
 const int* ranges,const int* list,int W,int H,const int* counts,const long long* offsets,
 const int* ids,const float* a,const double* z,const int* ord,const int* rank,const int* original,int* seen,long long* stats){
 int pix=blockIdx.x*blockDim.x+threadIdx.x;if(pix>=W*H)return;int n=counts[pix],X=pix%W,Y=pix/W;
 int tile=(Y/16)*((W+15)/16)+X/16,start=ranges[2*tile],end=ranges[2*tile+1];long long base=offsets[pix];
 for(int j=0;j<n;++j){long long at=base+j;int r=rank[at],o=ord[at];
  if(r<0||r>=n){add(stats,3);continue;}
  if(atomicCAS(seen+base+r,0,1)!=0)add(stats,4);
  if(o!=original[base+r]||o<=0||o>end-start){add(stats,5);continue;}
  if(ids[at]!=list[start+o-1])add(stats,6);
  float alpha;if(!alpha_at(ids[at],X,Y,no,off,ph,sig,co,alpha)||__float_as_uint(alpha)!=__float_as_uint(a[at]))add(stats,7);
  if(j>0&&greater(get(at-1,ids,a,z,ord,rank),get(at,ids,a,z,ord,rank)))add(stats,8);
 }
}
__global__ void replay_kernel(const int* counts,const long long* offsets,const int* ids,const float* a,const int* ord,
 const float* colors,int HW,float* rgb,float* alpha,int* paths){
 int pix=blockIdx.x*blockDim.x+threadIdx.x;if(pix>=HW)return;float T=1.0f,C[3]={0,0,0};int last=0,median=-1;
 long long base=offsets[pix];
 for(int j=0;j<counts[pix];++j){long long at=base+j;float al=a[at];int id=ids[at];
  float test=T*(1-al);if(test<.0001f)break;
  if(T>.5f)median=ord[at];
  for(int ch=0;ch<3;++ch)C[ch]+=colors[3*id+ch]*al*T;
  T=test;last=ord[at];
 }
 for(int ch=0;ch<3;++ch)rgb[ch*HW+pix]=C[ch];alpha[pix]=1-T;paths[pix]=last;paths[HW+pix]=median;
}

void basic(torch::Tensor t){TORCH_CHECK(t.is_cuda()&&t.is_contiguous());}
std::vector<torch::Tensor> counts(torch::Tensor no,torch::Tensor off,torch::Tensor ph,torch::Tensor sig,torch::Tensor co,
 torch::Tensor ranges,torch::Tensor list,torch::Tensor plane,int W,int H,double tx,double ty){
 basic(no);basic(off);basic(ph);basic(sig);basic(co);basic(ranges);basic(list);basic(plane);
 auto c=torch::zeros({W*H},ranges.options()),s=torch::zeros({16},ranges.options().dtype(torch::kInt64));
 count_kernel<<<(W*H+255)/256,256>>>(no.data_ptr<float>(),off.data_ptr<float>(),ph.data_ptr<float>(),sig.data_ptr<float>(),co.data_ptr<float>(),
  ranges.data_ptr<int>(),list.data_ptr<int>(),plane.data_ptr<double>(),W,H,tx,ty,c.data_ptr<int>(),s.data_ptr<long long>());return {c,s};
}
std::vector<torch::Tensor> fill(torch::Tensor no,torch::Tensor off,torch::Tensor ph,torch::Tensor sig,torch::Tensor co,
 torch::Tensor ranges,torch::Tensor list,torch::Tensor plane,int W,int H,double tx,double ty,torch::Tensor count,torch::Tensor offsets,torch::Tensor stats){
 long long N=offsets[-1].item<long long>();auto ids=torch::empty({N},ranges.options()),a=torch::empty({N},no.options()),
 z=torch::empty({N},plane.options()),ord=torch::empty_like(ids),rank=torch::empty_like(ids),orig=torch::empty_like(ids);
 fill_kernel<<<(W*H+255)/256,256>>>(no.data_ptr<float>(),off.data_ptr<float>(),ph.data_ptr<float>(),sig.data_ptr<float>(),co.data_ptr<float>(),
 ranges.data_ptr<int>(),list.data_ptr<int>(),plane.data_ptr<double>(),W,H,tx,ty,count.data_ptr<int>(),offsets.data_ptr<long long>(),
 ids.data_ptr<int>(),a.data_ptr<float>(),z.data_ptr<double>(),ord.data_ptr<int>(),rank.data_ptr<int>(),orig.data_ptr<int>(),stats.data_ptr<long long>());
 return {ids,a,z,ord,rank,orig};
}
void sort(torch::Tensor c,torch::Tensor offsets,torch::Tensor ids,torch::Tensor a,torch::Tensor z,torch::Tensor ord,torch::Tensor rank){
 sort_kernel<<<(c.numel()+127)/128,128>>>(c.data_ptr<int>(),offsets.data_ptr<long long>(),c.numel(),ids.data_ptr<int>(),a.data_ptr<float>(),z.data_ptr<double>(),ord.data_ptr<int>(),rank.data_ptr<int>());
}
torch::Tensor verify(torch::Tensor no,torch::Tensor off,torch::Tensor ph,torch::Tensor sig,torch::Tensor co,
 torch::Tensor ranges,torch::Tensor list,int W,int H,torch::Tensor c,torch::Tensor offsets,
 torch::Tensor ids,torch::Tensor a,torch::Tensor z,torch::Tensor ord,torch::Tensor rank,torch::Tensor original,torch::Tensor stats){
 auto seen=torch::zeros_like(ids);
 verify_kernel<<<(W*H+127)/128,128>>>(no.data_ptr<float>(),off.data_ptr<float>(),ph.data_ptr<float>(),sig.data_ptr<float>(),co.data_ptr<float>(),
 ranges.data_ptr<int>(),list.data_ptr<int>(),W,H,c.data_ptr<int>(),offsets.data_ptr<long long>(),ids.data_ptr<int>(),a.data_ptr<float>(),z.data_ptr<double>(),ord.data_ptr<int>(),rank.data_ptr<int>(),original.data_ptr<int>(),seen.data_ptr<int>(),stats.data_ptr<long long>());
 return seen;
}
std::vector<torch::Tensor> replay(torch::Tensor c,torch::Tensor offsets,torch::Tensor ids,torch::Tensor a,torch::Tensor ord,torch::Tensor colors){
 int HW=c.numel();auto rgb=torch::zeros({3,HW},colors.options()),alpha=torch::zeros({HW},colors.options()),paths=torch::zeros({2,HW},c.options());
 replay_kernel<<<(HW+255)/256,256>>>(c.data_ptr<int>(),offsets.data_ptr<long long>(),ids.data_ptr<int>(),a.data_ptr<float>(),ord.data_ptr<int>(),colors.data_ptr<float>(),HW,rgb.data_ptr<float>(),alpha.data_ptr<float>(),paths.data_ptr<int>());
 return {rgb,alpha,paths};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("counts",&counts);m.def("fill",&fill);m.def("sort",&sort);m.def("verify",&verify);m.def("replay",&replay);}
