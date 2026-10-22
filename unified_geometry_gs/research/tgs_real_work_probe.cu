// Separate observer of actual v5 sorted state. No optimizer or replacement renderer.
#include <torch/extension.h>
#include <cuda_runtime.h>
#define STABLE_MODE 3
#include "stable_window.cuh"
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " CUDA contiguous required")
__device__ bool coefficient_domain(double c){return isfinite(c)&&(c==0||(fabs(c)>=0x1p-60&&fabs(c)<=0x1p20));}
__device__ bool split_domain(float c){return isfinite(c)&&(c==0||(fabsf(c)>=0x1p-60f&&fabsf(c)<=0x1p20f));}
__device__ float cvup(double d){return __double2float_ru(fabs(d));}
__global__ void cache_kernel(const double* c,const float2* anchors,const uint2* rmin,const uint2* rmax,const int* radii,
                            float* a,float* cache,float2* B,int* domain,int N,int W,int H){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N||radii[i]<=0)return;
    c+=10*i;a+=18*i;cache+=16*i;float2 an=anchors[i];uint2 mn=rmin[i],mx=rmax[i];
    float bx=fmaxf(fabsf(__fsub_rn((float)(16*mn.x),an.x)),fabsf(__fsub_rn((float)(min(W,(int)(16*mx.x))-1),an.x)));
    float by=fmaxf(fabsf(__fsub_rn((float)(16*mn.y),an.y)),fabsf(__fsub_rn((float)(min(H,(int)(16*mx.y))-1),an.y)));
    B[i]=make_float2(bx,by);bool ok=isfinite(an.x)&&isfinite(an.y)&&fabsf(an.x)<0x1p23f&&fabsf(an.y)<0x1p23f&&an.x==truncf(an.x)&&an.y==truncf(an.y)&&bx<=8192&&by<=8192&&mx.x>mn.x&&mx.y>mn.y;
    float ef=0;
    for(int k=0;k<3;k++){
        float M=0,D=0,M64=0,Dhi=0;int ids[3]={2*k,2*k+1,6+k};
        for(int t=0;t<3;t++){
            double z=c[ids[t]];float h=__double2float_rn(z);double rem=__dadd_rn(z,-(double)h);float l=__double2float_rn(rem);
            double delta=__dadd_rn(rem,-(double)l);a[6*k+t]=h;a[6*k+3+t]=l;
            ok&=coefficient_domain(z)&&split_domain(h)&&split_domain(l);float b=t==0?bx:(t==1?by:1);
            M=upadd(M,upmul(upadd(fabsf(h),fabsf(l)),b));D=upadd(D,upmul(cvup(delta),b));M64=upadd(M64,upmul(cvup(z),b));Dhi=upadd(Dhi,upmul(cvup(rem),b));
        }
        cache[4*k]=M;cache[4*k+1]=D;cache[4*k+2]=upmul(0x1p-51f,M64);
        cache[4*k+3]=upadd(upadd(Dhi,upmul(0x1p-22f,M)),cache[4*k+2]);ef=fmaxf(ef,cache[4*k+3]);
    }
    cache[12]=ef;cache[13]=__double2float_rn(c[9]);cache[14]=cvup(__dadd_rn(c[9],-(double)cache[13]));cache[15]=__double2float_ru(c[9]);domain[i]=ok;
}
__device__ void two_sum(float a,float b,float& p,float& q){p=__fadd_rn(a,b);float z=__fsub_rn(p,a);q=__fadd_rn(__fsub_rn(a,__fsub_rn(p,z)),__fsub_rn(b,z));}
__device__ float dot2(const float* a,float x,float y){
    float p=0,s=0;
    for(int k=0;k<6;k++){float v=k%3==0?x:(k%3==1?y:1);float h=__fmul_rn(a[k],v),r=__fmaf_rn(a[k],v,-h);if(k==0){p=h;s=r;}else{float q;two_sum(p,h,p,q);s=__fadd_rn(s,__fadd_rn(q,r));}}
    return __fadd_rn(p,s);
}
__global__ void profile_kernel(const uint2* ranges,const uint32_t* ids,const double* coeff,const float2* anchor,const float4* conic,const float* sigma,const float* depth,const float* rgb,
    const float* a,const float* cache,const float2* B,const int* domain,const int* smap,
    int W,int H,int* counts,float* image,float* aux,uint32_t* path,float* maxblend,double* samples){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=W*H)return;int X=p%W,Y=p/W;uint2 range=ranges[(Y/16)*((W+15)/16)+X/16];
    int* ct=counts+24*p;float T=1,C[3]={0},N[3]={0},D=0,M1=0,M2=0,distortion=0,median=0;uint32_t last=0,mid=0xffffffff;
    int si=smap[p];uint32_t chosen=range.y>range.x?1u+((uint32_t)p*1664525u+1013904223u)%min(range.y-range.x,256u):0;
    for(uint32_t j=range.x;j<range.y;j++){
        int id=ids[j];uint32_t ordinal=j-range.x+1;const double* c=coeff+10*id;const float* aa=a+18*id;const float* ca=cache+16*id;
        float2 an=anchor[id];float x=__fsub_rn((float)X,an.x),y=__fsub_rn((float)Y,an.y);float4 co=conic[id];
        auto w=stable_window(nullptr,nullptr,nullptr,c,an,make_float2(X,Y),sigma[id],co.w,0);
        ++ct[0];int stage=6;float coarse=-CUDART_INF_F,ph=-CUDART_INF_F,E=0,vals[3]={0},errs[3]={0};int active=0;float cand=NAN,allow=NAN,ab=NAN;bool accept=false;
        bool coord_ok=fabsf(x)<=B[id].x&&fabsf(y)<=B[id].y;
        if(domain[id]&&!coord_ok)++ct[15];
        if(domain[id]&&coord_ok){
            for(int k=0;k<3;k++){float t=__fadd_rn(__fadd_rn(__fmul_rn(aa[6*k],x),__fmul_rn(aa[6*k+1],y)),aa[6*k+2]);coarse=fmaxf(coarse,t);}
            if(coarse>ca[12]){stage=1;if(!w.outside)++ct[11];}
            else{
                for(int k=0;k<3;k++){
                    float r=dot2(aa+6*k,x,y);vals[k]=r;float e=upadd(upmul(0x1.000002p0f,upadd(upadd(upmul(0x1p-24f,fabsf(r)),upmul(0x1.88p-43f,ca[4*k])),ca[4*k+1])),ca[4*k+2]);
                    errs[k]=e;E=fmaxf(E,e);if(r>ph){ph=r;active=k;}
                }
                if(ph>E){stage=2;if(!w.outside)++ct[12];}
                else if(ph < -E){
                    bool unique=true;float lower=__fsub_rd(vals[active],errs[active]);for(int k=0;k<3;k++)if(k!=active)unique&=lower>__fadd_ru(vals[k],errs[k]);
                    stage=unique?3:4;if(w.outside)++ct[13];if(unique&&active!=w.active)++ct[14];
                    if(unique){
                        float u=__fmul_rn(-ph,ca[13]);
                        float du=upadd(upadd(upmul(E,ca[15]),upmul(fabsf(ph),ca[14])),upadd(upmul(0x1p-24f,upmul(fabsf(ph),fabsf(ca[13]))),upadd(0x1p-125f,upmul(0x1p-52f,upmul(upadd(fabsf(ph),E),ca[15])))));
                        float lo=__fsub_rd(u,du);ab=lo>0?__fdiv_ru(upmul(upmul(.99f,sigma[id]),du),lo):CUDART_INF_F;
                        if(ab<=1e-6f){++ct[16];float raw=__fmul_rn(co.w,__powf(fmaxf(u,0),sigma[id]));cand=fminf(.99f,raw);allow=upadd(ab,2e-6f);
                            if(fabsf(__fsub_rn(cand,w.alpha))>allow)++ct[18];
                            if((cand>=1.0f/255.0f)!=(w.alpha>=1.0f/255.0f))++ct[19];
                            if((raw>=.99f)!=w.capped)++ct[20];
                            accept=fabsf(__fsub_rn(cand,1.0f/255.0f))>allow&&fabsf(__fsub_rn(raw,.99f))>allow;
                            if(accept)++ct[22];else ++ct[23];
                        }else ++ct[17];
                    }
                }else stage=5;
            }
        }
        ++ct[stage];
        if(si>=0 && ordinal==chosen){double* s=samples+24*si;s[0]=1;s[1]=id;s[2]=X;s[3]=Y;s[4]=ordinal;s[5]=stage;s[6]=coarse;s[7]=ca[12];s[8]=ph;s[9]=E;
            for(int k=0;k<3;k++){s[10+2*k]=vals[k];s[11+2*k]=errs[k];}s[16]=w.outside;s[17]=w.active;s[18]=w.alpha;s[19]=cand;s[20]=allow;s[21]=ab;s[22]=active;s[23]=accept;++ct[21];}
        if(w.outside){++ct[7];continue;}float alpha=w.alpha;if(alpha<1.0f/255.0f){++ct[8];continue;}
        float test=__fmul_rn(T,__fsub_rn(1,alpha));if(test<.0001f){++ct[10];break;}
        ++ct[9];float weight=__fmul_rn(alpha,T);atomicMax((int*)(maxblend+id),__float_as_int(weight));
        float A=__fsub_rn(1,T),m=phase_depth_m(depth[id],.2f,100.f);
        distortion=__fmaf_rn(__fsub_rn(__fmaf_rn(__fmul_rn(m,m),A,M2),__fmul_rn(__fmul_rn(2.f,m),M1)),weight,distortion);
        D=__fmaf_rn(depth[id],weight,D);M1=__fmaf_rn(m,weight,M1);M2=__fmaf_rn(__fmul_rn(m,m),weight,M2);
        if(T>.5f){median=depth[id];mid=ordinal;}float nn[3]={co.x,co.y,co.z};
        for(int k=0;k<3;k++){N[k]=__fmaf_rn(nn[k],weight,N[k]);C[k]=__fmaf_rn(__fmul_rn(rgb[3*id+k],alpha),T,C[k]);}
        T=test;last=ordinal;
    }
    int hw=W*H;for(int k=0;k<3;k++){image[k*hw+p]=__fmaf_rn(T,0.f,C[k]);aux[(2+k)*hw+p]=N[k];}
    aux[p]=D;aux[hw+p]=__fsub_rn(1,T);aux[5*hw+p]=median;aux[6*hw+p]=distortion;path[p]=last;path[hw+p]=mid;
}
std::vector<torch::Tensor> profile(torch::Tensor ranges,torch::Tensor ids,torch::Tensor coeff,torch::Tensor anchors,torch::Tensor conic,torch::Tensor sigma,torch::Tensor depth,torch::Tensor rgb,torch::Tensor rmin,torch::Tensor rmax,torch::Tensor radii,torch::Tensor smap,int W,int H,int S){
    for(auto t:{ranges,ids,coeff,anchors,conic,sigma,depth,rgb,rmin,rmax,radii,smap})CHECK_CUDA(t);
    int n=coeff.size(0);auto fo=anchors.options().dtype(torch::kFloat32),io=anchors.options().dtype(torch::kInt32);
    auto a=torch::zeros({n,18},fo),ca=torch::zeros({n,16},fo),B=torch::zeros({n,2},fo),domain=torch::zeros({n},io);
    auto counts=torch::zeros({W*H,24},io),image=torch::zeros({3,H,W},fo),aux=torch::zeros({7,H,W},fo),path=torch::zeros({2,H,W},io),maxblend=torch::zeros({n},fo),samples=torch::zeros({S,24},fo.dtype(torch::kFloat64));
    cache_kernel<<<(n+255)/256,256>>>(coeff.data_ptr<double>(),(float2*)anchors.data_ptr<float>(),(uint2*)rmin.data_ptr<int>(),(uint2*)rmax.data_ptr<int>(),radii.data_ptr<int>(),a.data_ptr<float>(),ca.data_ptr<float>(),(float2*)B.data_ptr<float>(),domain.data_ptr<int>(),n,W,H);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"cache launch");
    profile_kernel<<<(W*H+127)/128,128>>>((uint2*)ranges.data_ptr<int>(),(uint32_t*)ids.data_ptr<int>(),coeff.data_ptr<double>(),(float2*)anchors.data_ptr<float>(),(float4*)conic.data_ptr<float>(),sigma.data_ptr<float>(),depth.data_ptr<float>(),rgb.data_ptr<float>(),a.data_ptr<float>(),ca.data_ptr<float>(),(float2*)B.data_ptr<float>(),domain.data_ptr<int>(),smap.data_ptr<int>(),W,H,counts.data_ptr<int>(),image.data_ptr<float>(),aux.data_ptr<float>(),(uint32_t*)path.data_ptr<int>(),maxblend.data_ptr<float>(),samples.data_ptr<double>());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"profile launch");return {counts,image,aux,path,maxblend,samples,a,ca,B,domain};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("profile",&profile);}
