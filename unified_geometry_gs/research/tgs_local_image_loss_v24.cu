// Research prototype: fixed-input image-loss stage, no rasterization or backward.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>
#include <vector>

__device__ int event_at(const int64_t* prefix,int K,int64_t block){
 int lo=0,hi=K;while(lo+1<hi){int mid=(lo+hi)/2;if(prefix[mid]<=block)lo=mid;else hi=mid;}return lo;
}
__device__ double get_trial(const float* base,const float* patches,const int* b,int64_t offset,int ch,int x,int y,int W,int H){
 if(x<0||x>=W||y<0||y>=H)return 0.;
 if(x>=b[0]&&x<b[2]&&y>=b[1]&&y<b[3])return (double)patches[offset+(int64_t)ch*(b[2]-b[0])*(b[3]-b[1])+(y-b[1])*(b[2]-b[0])+x-b[0]];
 return (double)base[(int64_t)ch*W*H+y*W+x];
}
__global__ void vertical_kernel(const float* base,const unsigned char* gt,const float* patches,const int* boxes,
 const int64_t* ro,const int64_t* vo,const int64_t* vb,int K,int W,int H,const double* weights,double* moments,int64_t* stats){
 __shared__ int e;__shared__ int64_t start;
 if(threadIdx.x==0){e=event_at(vb,K,blockIdx.x);start=vo[e]+((int64_t)blockIdx.x-vb[e])*256;}__syncthreads();
 int64_t at=start+threadIdx.x;if(at>=vo[e+1])return;const int* b=boxes+10*e;
 int jw=b[9]-b[8],ah=b[7]-b[5];int64_t p=at-vo[e];int ch=p/((int64_t)jw*ah);int x=b[8]+p%jw,y=b[5]+(p/jw)%ah;double s[8]={0,0,0,0,0,0,0,0};
 for(int d=-5;d<=5;++d){int yy=y+d;double B=0,X=0,G=0;
  if(yy>=0&&yy<H){int64_t idx=(int64_t)ch*W*H+yy*W+x;B=(double)base[idx];X=get_trial(base,patches,b,ro[e],ch,x,yy,W,H);G=__ddiv_rn((double)gt[idx],255.);}
  if(!isfinite(B)||!isfinite(X))atomicAdd((unsigned long long*)stats,1ULL);
  double vals[8]={B,X,G,B*B,X*X,G*G,B*G,X*G};double w=weights[d+5];
  for(int k=0;k<8;++k)s[k]+=w*vals[k];
 }
 for(int k=0;k<8;++k)moments[8*at+k]=s[k];
}
__device__ double ssim(double mx,double mg,double exx,double egg,double exg){
 double vx=exx-mx*mx,vg=egg-mg*mg,cov=exg-mx*mg;
 const double C1=.01*.01,C2=.03*.03;
 return ((2.*mx*mg+C1)*(2.*cov+C2))/((mx*mx+mg*mg+C1)*(vx+vg+C2));
}
__global__ void horizontal_kernel(const float* base,const unsigned char* gt,const float* patches,const int* boxes,
 const int64_t* ro,const int64_t* vo,const int64_t* ao,const int64_t* ab,int K,int W,int H,const double* weights,
 const double* moments,double* partial,double* maps,bool debug,int64_t* stats){
 __shared__ int e;__shared__ int64_t start;__shared__ double sums[3][256];int t=threadIdx.x;
 if(t==0){e=event_at(ab,K,blockIdx.x);start=ao[e]+((int64_t)blockIdx.x-ab[e])*256;}__syncthreads();
 int64_t at=start+t;double mse=0,l1=0,ds=0;
 if(at<ao[e+1]){const int* b=boxes+10*e;int aw=b[6]-b[4],ah=b[7]-b[5],jw=b[9]-b[8];int64_t p=at-ao[e];int ch=p/((int64_t)aw*ah),x=b[4]+p%aw,y=b[5]+(p/aw)%ah;double s[8]={0,0,0,0,0,0,0,0};
  for(int d=-5;d<=5;++d){int xx=x+d;if(xx<0||xx>=W)continue;
   int64_t idx=vo[e]+(int64_t)ch*jw*ah+(y-b[5])*jw+xx-b[8];double w=weights[d+5];
   for(int k=0;k<8;++k)s[k]+=w*moments[8*idx+k];
  }
  double sb=ssim(s[0],s[2],s[3],s[5],s[6]),sx=ssim(s[1],s[2],s[4],s[5],s[7]);ds=sx-sb;
  if(!isfinite(sb)||!isfinite(sx))atomicAdd((unsigned long long*)(stats+1),1ULL);
  if(debug){maps[2*at]=sb;maps[2*at+1]=sx;}
  if(x>=b[0]&&x<b[2]&&y>=b[1]&&y<b[3]){int64_t idx=(int64_t)ch*W*H+y*W+x;double B=(double)base[idx],X=get_trial(base,patches,b,ro[e],ch,x,y,W,H),G=__ddiv_rn((double)gt[idx],255.);double dx=X-G,db=B-G;mse=dx*dx-db*db;l1=fabs(dx)-fabs(db);}
 }
 sums[0][t]=mse;sums[1][t]=l1;sums[2][t]=ds;__syncthreads();
 for(int stride=128;stride>0;stride/=2){if(t<stride)for(int k=0;k<3;++k)sums[k][t]+=sums[k][t+stride];__syncthreads();}
 if(t==0)for(int k=0;k<3;++k)partial[3*(int64_t)blockIdx.x+k]=sums[k][0];
}
__global__ void reduce_kernel(const double* partial,const int64_t* ab,int K,int64_t denominator,double* scores){
 __shared__ double sums[3][256];int e=blockIdx.x,t=threadIdx.x;
 for(int k=0;k<3;++k){double v=0;for(int64_t j=ab[e]+t;j<ab[e+1];j+=256)v+=partial[3*j+k];sums[k][t]=v;}__syncthreads();
 for(int stride=128;stride>0;stride/=2){if(t<stride)for(int k=0;k<3;++k)sums[k][t]+=sums[k][t+stride];__syncthreads();}
 if(t==0)for(int k=0;k<3;++k)scores[3*e+k]=sums[k][0]/(double)denominator;
}
void tensor_check(torch::Tensor x,torch::ScalarType dtype,int device){TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==dtype&&x.get_device()==device);}
std::vector<torch::Tensor> evaluate(torch::Tensor base,torch::Tensor gt,torch::Tensor patches,torch::Tensor boxes,
 torch::Tensor ro,torch::Tensor vo,torch::Tensor vb,torch::Tensor ao,torch::Tensor ab,torch::Tensor weights,std::vector<int64_t> totals,bool debug){
 TORCH_CHECK(base.is_cuda()&&base.dim()==3&&base.size(0)==3&&base.size(1)>0&&base.size(2)>0);int device=base.get_device();c10::cuda::CUDAGuard guard(base.device());
 tensor_check(base,torch::kFloat32,device);tensor_check(gt,torch::kUInt8,device);tensor_check(patches,torch::kFloat32,device);tensor_check(boxes,torch::kInt32,device);
 for(auto p:{ro,vo,vb,ao,ab})tensor_check(p,torch::kInt64,device);tensor_check(weights,torch::kFloat64,device);
 int64_t H=base.size(1),W=base.size(2),K=boxes.size(0);TORCH_CHECK(W*H<=INT_MAX/3&&K>0&&boxes.dim()==2&&boxes.size(1)==10&&weights.numel()==11&&gt.sizes()==base.sizes());
 TORCH_CHECK(ro.numel()==K+1&&vo.numel()==K+1&&vb.numel()==K+1&&ao.numel()==K+1&&ab.numel()==K+1&&totals.size()==5);
 TORCH_CHECK(totals[0]==patches.numel()&&totals[1]>=0&&totals[2]>=0&&totals[3]>=0&&totals[4]>=0&&totals[2]<=INT_MAX&&totals[4]<=INT_MAX);
 auto opt=base.options().dtype(torch::kFloat64);auto moments=torch::empty({totals[1],8},opt),partial=torch::empty({totals[4],3},opt),scores=torch::empty({K,3},opt),maps=torch::empty({debug?totals[3]:0,2},opt),stats=torch::zeros({2},base.options().dtype(torch::kInt64));
 cudaStream_t stream=c10::cuda::getCurrentCUDAStream();
 if(totals[2])vertical_kernel<<<(unsigned)totals[2],256,0,stream>>>(base.data_ptr<float>(),gt.data_ptr<unsigned char>(),patches.data_ptr<float>(),boxes.data_ptr<int>(),ro.data_ptr<int64_t>(),vo.data_ptr<int64_t>(),vb.data_ptr<int64_t>(),K,W,H,weights.data_ptr<double>(),moments.data_ptr<double>(),stats.data_ptr<int64_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 if(totals[4])horizontal_kernel<<<(unsigned)totals[4],256,0,stream>>>(base.data_ptr<float>(),gt.data_ptr<unsigned char>(),patches.data_ptr<float>(),boxes.data_ptr<int>(),ro.data_ptr<int64_t>(),vo.data_ptr<int64_t>(),ao.data_ptr<int64_t>(),ab.data_ptr<int64_t>(),K,W,H,weights.data_ptr<double>(),moments.data_ptr<double>(),partial.data_ptr<double>(),maps.data_ptr<double>(),debug,stats.data_ptr<int64_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();reduce_kernel<<<(unsigned)K,256,0,stream>>>(partial.data_ptr<double>(),ab.data_ptr<int64_t>(),K,3*W*H,scores.data_ptr<double>());C10_CUDA_KERNEL_LAUNCH_CHECK();
 return {scores,stats,maps};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("evaluate",&evaluate);}
