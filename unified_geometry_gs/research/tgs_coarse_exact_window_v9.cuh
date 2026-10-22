// Single-pass conditional coarse rejection; all surviving windows use original native64 arithmetic.
#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
struct StableWindow {float phi,u,Cx,alpha,alpha_error;int active;bool outside,capped,high,uncertain;};
__device__ __forceinline__ float upadd(float a,float b){return __fadd_ru(a,b);}
__device__ __forceinline__ float upmul(float a,float b){return __fmul_ru(a,b);}
__device__ __forceinline__ bool coarse_domain64(double a){return isfinite(a)&&(a==0||(fabs(a)>=0x1p-60&&fabs(a)<=0x1p20));}
__device__ __forceinline__ bool coarse_domain32(float a){return isfinite(a)&&(a==0||(fabsf(a)>=0x1p-60f&&fabsf(a)<=0x1p20f));}
// hi planes9, common positive error bound1, Bx/By2, domain1, reserved1.
__device__ __forceinline__ void coarse_precompute(float* p,const double* c,float2 anchor,uint2 mn,uint2 mx,int W,int H){
    for(int k=0;k<14;k++)p[k]=0;
    float bx=fmaxf(fabsf(__fsub_rn((float)(16*mn.x),anchor.x)),fabsf(__fsub_rn((float)(min(W,(int)(16*mx.x))-1),anchor.x)));
    float by=fmaxf(fabsf(__fsub_rn((float)(16*mn.y),anchor.y)),fabsf(__fsub_rn((float)(min(H,(int)(16*mx.y))-1),anchor.y)));
    p[10]=bx;p[11]=by;bool ok=W<=8192&&H<=8192&&isfinite(anchor.x)&&isfinite(anchor.y)&&fabsf(anchor.x)<0x1p23f&&fabsf(anchor.y)<0x1p23f&&anchor.x==truncf(anchor.x)&&anchor.y==truncf(anchor.y)&&bx<=8192&&by<=8192&&mx.x>mn.x&&mx.y>mn.y;
    for(int k=0;k<3;k++){
        float M=0,D=0,M64=0;int indices[3]={2*k,2*k+1,6+k};
        for(int t=0;t<3;t++){
            double z=c[indices[t]];float a=__double2float_rn(z);p[3*k+t]=a;ok&=coarse_domain64(z)&&coarse_domain32(a);float b=t==0?bx:(t==1?by:1);
            M=upadd(M,upmul(fabsf(a),b));D=upadd(D,upmul(__double2float_ru(fabs(__dadd_rn(z,-(double)a))),b));M64=upadd(M64,upmul(__double2float_ru(fabs(z)),b));
        }
        p[9]=fmaxf(p[9],upadd(upadd(D,upmul(0x1p-22f,M)),upmul(0x1p-51f,M64)));
    }
    p[12]=ok?1.f:0.f;
}
__device__ __forceinline__ StableWindow stable_window(const float2* n,const float* b,const float* p,const double* c,float2 anchor,float2 pixel,float sigma,float opacity,float rinv,bool force_oracle=false){
    StableWindow z={};z.phi=-CUDART_INF_F;z.active=0;
    if(STABLE_MODE==2&&!force_oracle){
        float x=__fsub_rn(pixel.x,anchor.x),y=__fsub_rn(pixel.y,anchor.y);
        if(p[12]==1.f&&fabsf(x)<=p[10]&&fabsf(y)<=p[11]){
            float v=-CUDART_INF_F;for(int k=0;k<3;k++){float d=__fadd_rn(__fadd_rn(__fmul_rn(p[3*k],x),__fmul_rn(p[3*k+1],y)),p[3*k+2]);v=fmaxf(v,d);}
            if(v>p[9]){z.phi=v;z.outside=true;return z;}
        }else z.uncertain=true;
    }
    z.high=true;double best=-CUDART_INF;double x=__dadd_rn((double)pixel.x,-(double)anchor.x),y=__dadd_rn((double)pixel.y,-(double)anchor.y);
    for(int k=0;k<3;k++){double d=__dadd_rn(__dadd_rn(__dmul_rn(c[2*k],x),__dmul_rn(c[2*k+1],y)),c[6+k]);if(d>best){best=d;z.active=k;}}
    z.phi=__double2float_rn(best);z.outside=best>0;if(z.outside)return z;
    double u=fmax(0.,__dmul_rn(-best,c[9])),cx=pow(u,(double)sigma),raw=__dmul_rn((double)opacity,cx);
    z.u=__double2float_rn(u);z.Cx=__double2float_rn(cx);z.capped=raw>=.99;z.alpha=__double2float_rn(fmin(.99,raw));return z;
}
__device__ __forceinline__ float phase_depth_m(float depth,float near_,float far_){return __fmul_rn(__fdividef(far_,__fsub_rn(far_,near_)),__fsub_rn(1.f,__fdividef(near_,depth)));}
