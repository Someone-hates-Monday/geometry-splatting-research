// Diagnostic mixed precision window. STABLE_MODE: 1=arithmetic,2=adaptive,3=oracle.
#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
struct StableWindow {
    float phi,u,Cx,alpha;
    int active;
    bool outside,capped,high,uncertain;
};
__device__ __forceinline__ float upadd(float a,float b) {return __fadd_ru(a,b);}
__device__ __forceinline__ float upmul(float a,float b) {return __fmul_ru(a,b);}
__device__ __forceinline__ StableWindow stable_window(const float2* n,const float* b,
    const float* errors,const double* coeff,float2 anchor,float2 pixel,float sigma,float opacity,float rinv) {
    StableWindow z={};z.phi=-CUDART_INF_F;z.active=0;
    const float x=pixel.x-anchor.x,y=pixel.y-anchor.y;
    bool fallback=STABLE_MODE==3;
    if(STABLE_MODE!=3) {
        float bound=0.0f;
        for(int k=0;k<3;++k) {
            float d=n[k].x*x+n[k].y*y+b[k];
            if(d>z.phi) {z.phi=d;z.active=k;}
            if(STABLE_MODE==2) {
                // gamma3 <= 4*2^-24. Positive operations are rounded upward.
                float magnitude=upadd(upadd(upmul(fabsf(n[k].x),fabsf(x)),upmul(fabsf(n[k].y),fabsf(y))),fabsf(b[k]));
                float arithmetic=upadd(upmul(0x1p-22f,magnitude),0x1.8p-125f);
                float cache=upadd(upadd(upmul(errors[2*k],fabsf(x)),upmul(errors[2*k+1],fabsf(y))),errors[6+k]);
                bound=fmaxf(bound,upadd(cache,arithmetic));
            }
        }
        if(STABLE_MODE==1) fallback=true;
        if(STABLE_MODE==2) {
            bool exact_integer_subtraction=fabsf(anchor.x)<0x1p23f && fabsf(anchor.y)<0x1p23f && fabsf(x)<0x1p24f && fabsf(y)<0x1p24f;
            if(z.phi>bound && exact_integer_subtraction) {z.outside=true;return z;}
            z.uncertain=(fabsf(z.phi)<=bound)||!exact_integer_subtraction;
            fallback=z.uncertain;
            z.u=-z.phi*rinv;
            const float rbound=upadd(fabsf(rinv),errors[9]);
            const float du=upadd(upadd(upmul(bound,rbound),upmul(fabsf(z.phi),errors[9])),upadd(upmul(0x1p-24f,upmul(fabsf(z.phi),fabsf(rinv))),0x1p-125f));
            const float lower=__fsub_rd(z.u,du);
            float alpha_bound=CUDART_INF_F;
            if(lower>0) alpha_bound=__fdiv_ru(upmul(upmul(.99f,sigma),du),lower);
            z.Cx=fmaxf(0.0f,__powf(fmaxf(z.u,0.0f),sigma));
            const float raw=opacity*z.Cx;z.capped=raw>=.99f;z.alpha=fminf(.99f,raw);
            // Pow/cutoff allowance is explicitly empirical in the protocol.
            const float allowance=upadd(alpha_bound,2e-6f);
            fallback |= !(lower>0) || !(alpha_bound<=1e-6f)
                || fabsf(z.alpha-1.0f/255.0f)<=allowance || fabsf(raw-.99f)<=allowance;
            if(!fallback) return z;
        }
    }
    z.high=true;
    double best=-CUDART_INF;const double dx=(double)pixel.x-anchor.x,dy=(double)pixel.y-anchor.y;
    for(int k=0;k<3;++k) {
        const double nx=STABLE_MODE==1?(double)n[k].x:coeff[2*k];
        const double ny=STABLE_MODE==1?(double)n[k].y:coeff[2*k+1];
        const double off=STABLE_MODE==1?(double)b[k]:coeff[6+k];
        const double d=nx*dx+ny*dy+off;
        if(d>best) {best=d;z.active=k;}
    }
    z.phi=(float)best;z.outside=best>0.0;
    if(z.outside) return z;
    const double ir=STABLE_MODE==1?(double)rinv:coeff[9];
    const double u=fmax(0.0,-best*ir),Cx=pow(u,(double)sigma),raw=(double)opacity*Cx;
    z.u=(float)u;z.Cx=(float)Cx;z.capped=raw>=.99;z.alpha=(float)fmin(.99,raw);
    return z;
}
