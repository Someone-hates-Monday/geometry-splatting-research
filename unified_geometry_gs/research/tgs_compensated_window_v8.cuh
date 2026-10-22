// v8 isolated window: actual tile coarse filter + Dot2 + separated active interval.
#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
struct StableWindow {float phi,u,Cx,alpha,alpha_error;int active;bool outside,capped,high,uncertain;};
__device__ __forceinline__ float upadd(float a,float b){return __fadd_ru(a,b);}
__device__ __forceinline__ float upmul(float a,float b){return __fmul_ru(a,b);}
__device__ __forceinline__ bool comp_domain64(double c){return isfinite(c)&&(c==0||(fabs(c)>=0x1p-60&&fabs(c)<=0x1p20));}
__device__ __forceinline__ bool comp_domain32(float c){return isfinite(c)&&(c==0||(fabsf(c)>=0x1p-60f&&fabsf(c)<=0x1p20f));}
__device__ __forceinline__ float comp_abs_up(double v){return __double2float_ru(fabs(v));}
// p[0:18]=three hi/lo planes; [18:34]=four bounds/edge+coarse+Rhi/Rerr/Rup;
// [34:36]=local pixel extents, [36]=domain, [37]=reserved. Old10 cache floats retained separately.
__device__ __forceinline__ void comp_precompute(float* p,const double* coeff,float2 anchor,uint2 mn,uint2 mx,int W,int H){
    for(int k=0;k<38;k++)p[k]=0;
    float bx=fmaxf(fabsf(__fsub_rn((float)(16*mn.x),anchor.x)),fabsf(__fsub_rn((float)(min(W,(int)(16*mx.x))-1),anchor.x)));
    float by=fmaxf(fabsf(__fsub_rn((float)(16*mn.y),anchor.y)),fabsf(__fsub_rn((float)(min(H,(int)(16*mx.y))-1),anchor.y)));
    p[34]=bx;p[35]=by;float* cache=p+18;
    bool ok=W<=8192&&H<=8192&&isfinite(anchor.x)&&isfinite(anchor.y)&&fabsf(anchor.x)<0x1p23f&&fabsf(anchor.y)<0x1p23f&&anchor.x==truncf(anchor.x)&&anchor.y==truncf(anchor.y)&&bx<=8192&&by<=8192&&mx.x>mn.x&&mx.y>mn.y&&isfinite(coeff[9])&&coeff[9]>=0x1p-20&&coeff[9]<=0x1p20;
    for(int k=0;k<3;k++){
        int ids[3]={2*k,2*k+1,6+k};float M=0,D=0,M64=0,Dhi=0;
        for(int t=0;t<3;t++){
            double c=coeff[ids[t]];float hi=__double2float_rn(c);double rem=__dadd_rn(c,-(double)hi);float lo=__double2float_rn(rem);double delta=__dadd_rn(rem,-(double)lo);
            p[6*k+t]=hi;p[6*k+3+t]=lo;ok&=comp_domain64(c)&&comp_domain32(hi)&&comp_domain32(lo);float b=t==0?bx:(t==1?by:1);
            M=upadd(M,upmul(upadd(fabsf(hi),fabsf(lo)),b));D=upadd(D,upmul(comp_abs_up(delta),b));M64=upadd(M64,upmul(comp_abs_up(c),b));Dhi=upadd(Dhi,upmul(comp_abs_up(rem),b));
        }
        cache[4*k]=M;cache[4*k+1]=D;cache[4*k+2]=upmul(0x1p-51f,M64);
        cache[4*k+3]=upadd(upadd(Dhi,upmul(0x1p-22f,M)),cache[4*k+2]);cache[12]=fmaxf(cache[12],cache[4*k+3]);
    }
    cache[13]=__double2float_rn(coeff[9]);cache[14]=comp_abs_up(__dadd_rn(coeff[9],-(double)cache[13]));cache[15]=__double2float_ru(coeff[9]);p[36]=ok?1.f:0.f;
}
__device__ __forceinline__ void comp_two_sum(float a,float b,float& p,float& q){p=__fadd_rn(a,b);float z=__fsub_rn(p,a);q=__fadd_rn(__fsub_rn(a,__fsub_rn(p,z)),__fsub_rn(b,z));}
__device__ __forceinline__ float comp_dot2(const float* a,float x,float y){
    float p=0,s=0;
    for(int k=0;k<6;k++){float v=k%3==0?x:(k%3==1?y:1);float h=__fmul_rn(a[k],v),r=__fmaf_rn(a[k],v,-h);if(k==0){p=h;s=r;}else{float q;comp_two_sum(p,h,p,q);s=__fadd_rn(s,__fadd_rn(q,r));}}
    return __fadd_rn(p,s);
}
__device__ __forceinline__ float comp_distance_down(float a,float b){return a>=b?__fsub_rd(a,b):__fsub_rd(b,a);}
__device__ __forceinline__ StableWindow stable_window(const float2* n,const float* b,const float* errors,const double* coeff,float2 anchor,float2 pixel,float sigma,float opacity,float rinv,bool force_oracle=false){
    StableWindow z={};z.phi=-CUDART_INF_F;z.active=0;
    if(STABLE_MODE==2&&!force_oracle){
        const float* p=errors+10;const float* ca=p+18;float x=__fsub_rn(pixel.x,anchor.x),y=__fsub_rn(pixel.y,anchor.y);
        bool valid=p[36]==1.f&&fabsf(x)<=p[34]&&fabsf(y)<=p[35];
        if(valid){
            float coarse=-CUDART_INF_F;
            for(int k=0;k<3;k++){float d=__fadd_rn(__fadd_rn(__fmul_rn(p[6*k],x),__fmul_rn(p[6*k+1],y)),p[6*k+2]);coarse=fmaxf(coarse,d);}
            if(coarse>ca[12]){z.phi=coarse;z.outside=true;return z;}
            float r[3],e[3],lo[3],hi[3],lower=-CUDART_INF_F,upper=-CUDART_INF_F;
            for(int k=0;k<3;k++){
                r[k]=comp_dot2(p+6*k,x,y);e[k]=upadd(upmul(0x1.000002p0f,upadd(upadd(upmul(0x1p-24f,fabsf(r[k])),upmul(0x1.88p-43f,ca[4*k])),ca[4*k+1])),ca[4*k+2]);
                lo[k]=__fsub_rd(r[k],e[k]);hi[k]=__fadd_ru(r[k],e[k]);lower=fmaxf(lower,lo[k]);upper=fmaxf(upper,hi[k]);if(r[k]>z.phi){z.phi=r[k];z.active=k;}
            }
            if(lower>0){z.outside=true;return z;}
            bool unique=true;for(int k=0;k<3;k++)if(k!=z.active)unique&=lo[z.active]>hi[k];
            z.uncertain=!(upper<0)||!unique;
            if(!z.uncertain){
                const float E=e[z.active];z.u=__fmul_rn(-z.phi,ca[13]);
                const float du=upadd(upadd(upmul(E,ca[15]),upmul(fabsf(z.phi),ca[14])),upadd(upmul(0x1p-24f,upmul(fabsf(z.phi),fabsf(ca[13]))),upadd(0x1p-125f,upmul(0x1p-52f,upmul(upadd(fabsf(z.phi),E),ca[15])))));
                const float ulo=__fsub_rd(z.u,du);float ab=CUDART_INF_F;if(ulo>0)ab=__fdiv_ru(upmul(upmul(.99f,sigma),du),ulo);
                if(ab<=1e-6f){
                    z.Cx=fmaxf(0.f,__powf(fmaxf(z.u,0.f),sigma));float raw=__fmul_rn(opacity,z.Cx);z.alpha=fminf(.99f,raw);z.capped=raw>=.99f;
                    // Empirical power allowance retained; no all-input power enclosure claim.
                    float allow=upadd(ab,2e-6f);
                    // Capped/near-cap windows always use the oracle; do not infer raw-error bounds from a capped value.
                    if(isfinite(z.Cx)&&isfinite(raw)&&raw<__fsub_rd(.99f,allow)&&comp_distance_down(z.alpha,1.f/255.f)>allow){z.alpha_error=allow;return z;}
                }
            }
        }else z.uncertain=true;
    }
    z.high=true;double best=-CUDART_INF;const double dx=__dadd_rn((double)pixel.x,-(double)anchor.x),dy=__dadd_rn((double)pixel.y,-(double)anchor.y);
    for(int k=0;k<3;k++){
        const double d=__dadd_rn(__dadd_rn(__dmul_rn(coeff[2*k],dx),__dmul_rn(coeff[2*k+1],dy)),coeff[6+k]);if(d>best){best=d;z.active=k;}
    }
    z.phi=__double2float_rn(best);z.outside=best>0;if(z.outside)return z;
    const double u=fmax(0.0,__dmul_rn(-best,coeff[9])),Cx=pow(u,(double)sigma),raw=__dmul_rn((double)opacity,Cx);
    z.u=__double2float_rn(u);z.Cx=__double2float_rn(Cx);z.capped=raw>=.99;z.alpha=__double2float_rn(fmin(.99,raw));z.alpha_error=0;return z;
}
__device__ __forceinline__ float phase_depth_m(float depth,float near_,float far_){return __fmul_rn(__fdividef(far_,__fsub_rn(far_,near_)),__fsub_rn(1.f,__fdividef(near_,depth)));}
