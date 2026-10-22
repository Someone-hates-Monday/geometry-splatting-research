// Standalone, edge-only diagnostic. No renderer/optimizer dependency.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#define CUDA_OK(c) do { auto e=(c); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
__device__ float ua(float a,float b){return __fadd_ru(a,b);}
__device__ float um(float a,float b){return __fmul_ru(a,b);}
__device__ void twosum(float a,float b,float &s,float &e){
    s=__fadd_rn(a,b);float z=__fsub_rn(s,a);
    e=__fadd_rn(__fsub_rn(a,__fsub_rn(s,z)),__fsub_rn(b,z));
}
__global__ void edge(const double* c,const float* a,const float* xy,const float* cache,
                     float* out,double* oracle,uint32_t* gate,int n){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    c+=3*j;a+=6*j;xy+=2*j;cache+=4*j;out+=34*j;
    float x=xy[0],y=xy[1];bool ok=isfinite(x)&&isfinite(y)&&fabsf(x)<=8192&&fabsf(y)<=8192&&x==truncf(x)&&y==truncf(y);
    for(int k=0;k<3;k++)ok&=isfinite(c[k])&&(c[k]==0||(fabs(c[k])>=0x1p-60&&fabs(c[k])<=0x1p20));
    for(int k=0;k<6;k++)ok&=isfinite(a[k])&&(a[k]==0||(fabsf(a[k])>=0x1p-60f&&fabsf(a[k])<=0x1p20f));
    gate[j]=ok;float p=0,s=0;
    for(int k=0;k<6;k++){
        float v=k%3==0?x:(k%3==1?y:1.0f);
        float h=__fmul_rn(a[k],v),r=__fmaf_rn(a[k],v,-h);
        out[6+2*k]=h;out[7+2*k]=r;
        if(k==0){p=h;s=r;}else{
            float q;twosum(p,h,p,q);out[18+2*(k-1)]=p;out[19+2*(k-1)]=q;
            s=__fadd_rn(s,__fadd_rn(q,r));
        }
        out[28+k]=s;
    }
    float r=__fadd_rn(p,s);out[1]=r;
    float e=um(0x1.000002p0f,ua(ua(um(0x1p-24f,fabsf(r)),um(0x1.88p-43f,cache[0])),cache[1]));
    float d64=um(0x1p-51f,cache[2]);out[2]=e;out[3]=ua(e,d64);
    out[0]=__fadd_rn(__fadd_rn(__fmul_rn(a[0],x),__fmul_rn(a[1],y)),a[2]);
    out[4]=ua(cache[3],um(0x1p-22f,cache[0]));out[5]=ua(out[4],d64);
    oracle[j]=__dadd_rn(__dadd_rn(__dmul_rn(c[0],(double)x),__dmul_rn(c[1],(double)y)),c[2]);
}
__global__ void power(const float* in,float* out,double* ref,int n){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    float u=in[2*j],s=in[2*j+1];out[2*j]=powf(u,s);out[2*j+1]=__powf(u,s);ref[j]=pow((double)u,(double)s);
}
template<class T> void rd(std::ifstream &s,std::vector<T>&v){s.read((char*)v.data(),v.size()*sizeof(T));if(!s)throw std::runtime_error("short input");}
template<class T> void wr(std::ofstream &s,const std::vector<T>&v){s.write((const char*)v.data(),v.size()*sizeof(T));if(!s)throw std::runtime_error("write failed");}
template<class T> T* upload(const std::vector<T>&v){T*p;CUDA_OK(cudaMalloc(&p,v.size()*sizeof(T)));CUDA_OK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T> T* alloc(const std::vector<T>&v){T*p;CUDA_OK(cudaMalloc(&p,v.size()*sizeof(T)));return p;}
template<class T> void fetch(std::vector<T>&v,T*p){CUDA_OK(cudaMemcpy(v.data(),p,v.size()*sizeof(T),cudaMemcpyDeviceToHost));CUDA_OK(cudaFree(p));}
int main(int argc,char**argv){try{
    if(argc!=4)throw std::runtime_error("input edge-output power-output");
    std::ifstream f(argv[1],std::ios::binary);int64_t n,m;f.read((char*)&n,8);f.read((char*)&m,8);
    if(!f||n<=0||n>1000000||m<=0||m>1000000)throw std::runtime_error("bad sizes");
    std::vector<double>c(n*3),o(n),po(m);std::vector<float>a(n*6),xy(n*2),cache(n*4),v(n*34),pin(m*2),pout(m*2);std::vector<uint32_t>g(n);
    rd(f,c);rd(f,a);rd(f,xy);rd(f,cache);rd(f,pin);char extra;if(f.read(&extra,1))throw std::runtime_error("trailing input");
    double*dc=upload(c),*dref=alloc(o),*dpref=alloc(po);float*da=upload(a),*dxy=upload(xy),*dca=upload(cache),*dv=alloc(v),*dpi=upload(pin),*dpo=alloc(pout);uint32_t*dg=alloc(g);
    edge<<<(n+127)/128,128>>>(dc,da,dxy,dca,dv,dref,dg,n);CUDA_OK(cudaGetLastError());
    power<<<(m+127)/128,128>>>(dpi,dpo,dpref,m);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
    fetch(v,dv);fetch(o,dref);fetch(g,dg);fetch(pout,dpo);fetch(po,dpref);
    for(void*p:{(void*)dc,(void*)da,(void*)dxy,(void*)dca,(void*)dpi})CUDA_OK(cudaFree(p));
    std::ofstream e(argv[2],std::ios::binary),p(argv[3],std::ios::binary);wr(e,v);wr(e,o);wr(e,g);wr(p,pout);wr(p,po);
    cudaDeviceProp prop;CUDA_OK(cudaGetDeviceProperties(&prop,0));std::cout<<"rows="<<n<<" power="<<m<<" GPU="<<prop.name<<" sm="<<prop.major<<prop.minor<<"\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
