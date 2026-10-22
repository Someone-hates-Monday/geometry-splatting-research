#pragma once
#include "guarded_interval.cuh"
namespace gp {
using gi::I;using gi::pt;using gi::add;using gi::sub;using gi::mul;using gi::div;using gi::sq;
constexpr int WIDTH=26;
__device__ __forceinline__ I root(I a){if(a.lo<0)return gi::entire();return {__dsqrt_rd(a.lo),__dsqrt_ru(a.hi)};}
__device__ __forceinline__ I dot(const I* a,const I* b,int n){I v=pt(0);for(int i=0;i<n;i++)v=add(v,mul(a[i],b[i]));return v;}
__device__ __forceinline__ I norm(const I* a,int n){I v=pt(0);for(int i=0;i<n;i++)v=add(v,sq(a[i]));return root(v);}
__device__ __forceinline__ double norm_upper(const I* a,int n){
 double v=0;for(int i=0;i<n;i++){double b=fmax(fabs(a[i].lo),fabs(a[i].hi));v=__dadd_ru(v,__dmul_ru(b,b));}return __dsqrt_ru(v);
}
__device__ __forceinline__ void mm(const I* a,const I* b,I* c){
 for(int i=0;i<3;i++)for(int j=0;j<3;j++){I v=pt(0);for(int k=0;k<3;k++)v=add(v,mul(a[3*i+k],b[3*k+j]));c[3*i+j]=v;}
}
__device__ __forceinline__ void mv(const I* a,const I* b,I* c){for(int i=0;i<3;i++)c[i]=dot(a+3*i,b,3);}
__device__ __forceinline__ bool inverse(const I* a,I* out){
 I m[9]={sub(mul(a[4],a[8]),mul(a[5],a[7])),sub(mul(a[2],a[7]),mul(a[1],a[8])),sub(mul(a[1],a[5]),mul(a[2],a[4])),
         sub(mul(a[5],a[6]),mul(a[3],a[8])),sub(mul(a[0],a[8]),mul(a[2],a[6])),sub(mul(a[2],a[3]),mul(a[0],a[5])),
         sub(mul(a[3],a[7]),mul(a[4],a[6])),sub(mul(a[1],a[6]),mul(a[0],a[7])),sub(mul(a[0],a[4]),mul(a[1],a[3]))};
 I det=add(add(mul(a[0],m[0]),mul(a[1],m[3])),mul(a[2],m[6]));
 if(!gi::valid(det)||det.lo<=0&&det.hi>=0)return false;for(int i=0;i<9;i++)out[i]=div(m[i],det);return true;
}
__device__ __forceinline__ double midpoint(I a){return .5*a.lo+.5*a.hi;}
__device__ __forceinline__ double ndot(const double* a,const double* b){return __dadd_rn(__dadd_rn(__dmul_rn(a[0],b[0]),__dmul_rn(a[1],b[1])),__dmul_rn(a[2],b[2]));}
__device__ __forceinline__ bool qr(const double* f,double* q,double* u){
 double c0[3]={f[0],f[3],f[6]},c1[3]={f[1],f[4],f[7]},c2[3]={f[2],f[5],f[8]};
 double q0[3],q1[3],q2[3],v[3];double n=__dsqrt_rn(ndot(c0,c0));if(!(n>0)||!isfinite(n))return false;
 for(int i=0;i<3;i++)q0[i]=c0[i]/n;for(int i=0;i<3;i++)v[i]=c1[i];
 for(int pass=0;pass<2;pass++){double r=ndot(q0,v);for(int i=0;i<3;i++)v[i]-=r*q0[i];}
 n=__dsqrt_rn(ndot(v,v));if(!(n>0)||!isfinite(n))return false;for(int i=0;i<3;i++)q1[i]=v[i]/n;
 q2[0]=q0[1]*q1[2]-q0[2]*q1[1];q2[1]=q0[2]*q1[0]-q0[0]*q1[2];q2[2]=q0[0]*q1[1]-q0[1]*q1[0];
 n=__dsqrt_rn(ndot(q2,q2));if(!(n>0)||!isfinite(n))return false;for(int i=0;i<3;i++)q2[i]/=n;
 double* columns[3]={q0,q1,q2};const double* source[3]={c0,c1,c2};
 for(int i=0;i<3;i++){if(ndot(columns[i],source[i])<0)for(int j=0;j<3;j++)columns[i][j]=-columns[i][j];}
 for(int i=0;i<3;i++)for(int j=0;j<3;j++){q[3*i+j]=columns[j][i];u[3*i+j]=i<=j?ndot(columns[i],source[j]):0.;}
 return true;
}

__device__ __forceinline__ void rotation(const double* src,I* r){
 if(src[10]==0){for(int i=0;i<9;i++)r[i]=pt(src[11+i]);return;}
 I w=pt(src[6]),x=pt(src[7]),y=pt(src[8]),z=pt(src[9]),two=pt(2),one=pt(1);
 r[0]=sub(one,mul(two,add(sq(y),sq(z))));r[1]=mul(two,sub(mul(x,y),mul(w,z)));r[2]=mul(two,add(mul(x,z),mul(w,y)));
 r[3]=mul(two,add(mul(x,y),mul(w,z)));r[4]=sub(one,mul(two,add(sq(x),sq(z))));r[5]=mul(two,sub(mul(y,z),mul(w,x)));
 r[6]=mul(two,sub(mul(x,z),mul(w,y)));r[7]=mul(two,add(mul(y,z),mul(w,x)));r[8]=sub(one,mul(two,add(sq(x),sq(y))));
}

__device__ __forceinline__ void decode(const double* src,const I* f,const I* mu,const double* qhat,const double* u64,int mode,double* out){
 out[0]=5;I uv[9],qv[9],qt[9];double u[9];int upper[6]={0,1,2,4,5,8};
 for(int i=0;i<9;i++){u[i]=mode?u64[i]:double(float(u64[i]));uv[i]=pt(u[i]);qv[i]=pt(qhat[i]);qt[i]=pt(qhat[3*(i%3)+i/3]);}
 I gram[9];mm(qt,qv,gram);for(int i=0;i<9;i++)gram[i]=sub(gram[i],pt(i%4==0?1.:0.));double delta=norm_upper(gram,9);out[23]=delta;
 if(!(delta<1)){out[0]=3;return;}
 I qu[9],res[9];mm(qv,uv,qu);for(int i=0;i<9;i++)res[i]=sub(f[i],qu[i]);
 I polar=div(pt(delta),add(pt(1),root(sub(pt(1),pt(delta)))));
 I eta_box=add(pt(norm_upper(res,9)),mul(polar,pt(norm_upper(uv,9))));double eta=double(__double2float_ru(eta_box.hi));
 double mean[3];I err[3];for(int i=0;i<3;i++){mean[i]=midpoint(mu[i]);err[i]=sub(mu[i],pt(mean[i]));}
 double nu=double(__double2float_ru(norm_upper(err,3)));out[11]=eta;out[12]=nu;
 I rawq[3]={pt(src[32]),pt(src[33]),pt(src[34])};I qn=norm(rawq,3),d[3],offset[3],true_offset[3];
 if(rawq[2].lo==0){out[0]=4;return;}
 for(int i=0;i<3;i++)d[i]=div(rawq[i],qn);
 for(int i=0;i<2;i++){I ratio=div(rawq[i],rawq[2]);offset[i]=sub(pt(mean[i]),mul(ratio,pt(mean[2])));true_offset[i]=sub(mu[i],mul(ratio,mu[2]));}
 offset[2]=pt(0);true_offset[2]=pt(0);
 I ray[3],w[3];mv(uv,d,ray);mv(uv,offset,w);I a=add(add(sq(ray[0]),sq(ray[1])),sq(ray[2])),rho=root(a);
 if(!(rho.lo>eta)||!gi::valid(rho)){out[0]=4;return;}
 I correction=div(dot(ray,w,3),a),p=add(mul(div(pt(mean[2]),rawq[2]),qn),correction),r[3],x[3];
 for(int i=0;i<3;i++){r[i]=sub(w[i],mul(correction,ray[i]));x[i]=sub(offset[i],mul(correction,d[i]));}
 I e=sq(r[0]);for(int i=1;i<3;i++)e=add(e,sq(r[i]));
 I rho_max=add(pt(rho.hi),pt(eta)),rho_min=sub(pt(rho.lo),pt(eta));
 I zeta=add(mul(pt(eta),norm(x,3)),mul(add(pt(norm_upper(uv,9)),pt(eta)),pt(nu)));
 I dp=div(add(mul(pt(eta),root(pt(e.hi))),mul(rho_max,zeta)),sq(rho_min));
 I dr=add(zeta,mul(pt(dp.hi),rho_max));I de=add(mul(mul(pt(2),root(pt(e.hi))),dr),sq(dr));
 p=add(p,{-dp.hi,dp.hi});I ebox=add(e,{-de.hi,de.hi});ebox.lo=fmax(0.,ebox.lo);rho=add(rho,{-eta,eta});
 bool aligned=true_offset[0].lo==0&&true_offset[0].hi==0&&true_offset[1].lo==0&&true_offset[1].hi==0;
 if(aligned){p=mul(div(mu[2],rawq[2]),qn);ebox=pt(0);}
 double qnorm=__dsqrt_rn(src[32]*src[32]+src[33]*src[33]+src[34]*src[34]);double dn[3]={src[32]/qnorm,src[33]/qnorm,src[34]/qnorm};
 double off[3]={mean[0]-src[32]/src[34]*mean[2],mean[1]-src[33]/src[34]*mean[2],0},un[3],wn[3];
 for(int i=0;i<3;i++){un[i]=ndot(u+3*i,dn);wn[i]=ndot(u+3*i,off);}
 double an=ndot(un,un);double cn=ndot(un,wn)/an;double cross[3]={wn[1]*un[2]-wn[2]*un[1],wn[2]*un[0]-wn[0]*un[2],wn[0]*un[1]-wn[1]*un[0]};
 out[8]=mean[2]/src[34]*qnorm+cn;out[9]=__dsqrt_rn(an);out[10]=aligned?0.:ndot(cross,cross)/an;out[13]=aligned?1.:0.;
 out[1]=p.lo;out[2]=p.hi;out[3]=rho.lo;out[4]=rho.hi;out[5]=ebox.lo;out[6]=ebox.hi;out[7]=src[35];
 for(int i=0;i<6;i++)out[14+i]=u[upper[i]];for(int i=0;i<3;i++)out[20+i]=mean[i];
 I alpha=mul(pt(src[35]),gi::exp(mul(pt(-.5),ebox)));double cap=double(float(.99));alpha.lo=fmax(0.,fmin(cap,alpha.lo));alpha.hi=fmin(fmin(src[35],cap),alpha.hi);out[24]=alpha.lo;out[25]=alpha.hi;
 bool okay=gi::valid(p)&&gi::valid(rho)&&gi::valid(ebox)&&gi::valid(alpha);
 for(int i=1;i<WIDTH;i++)okay=okay&&isfinite(out[i]);out[0]=okay?0:5;
}

__global__ void preprocess(int n,const double* source,double* output){
 int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=n)return;const double* src=source+36*index;double* low=output+2*WIDTH*index;double* high=low+WIDTH;
 I A[9],R[9],Ai[9],Ri[9],f[9],mean[3],m[3]={pt(src[0]),pt(src[1]),pt(src[2])};
 for(int i=0;i<9;i++)A[i]=pt(src[20+i]);rotation(src,R);
 if(!inverse(A,Ai)||!inverse(R,Ri)){low[0]=high[0]=1;return;}
 mm(Ri,Ai,f);for(int i=0;i<3;i++)for(int j=0;j<3;j++)f[3*i+j]=div(f[3*i+j],pt(src[3+i]));
 mv(A,m,mean);for(int i=0;i<3;i++)mean[i]=add(mean[i],pt(src[29+i]));
 double fnom[9],q[9],u[9];for(int i=0;i<9;i++)fnom[i]=midpoint(f[i]);
 if(!qr(fnom,q,u)){low[0]=high[0]=2;return;}decode(src,f,mean,q,u,0,low);decode(src,f,mean,q,u,1,high);
}

__global__ void mass(int count,const double* packets,const int* sections,double* output){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=count*2)return;int model=id/2,mode=id%2;int start=sections[2*model],n=sections[2*model+1];I beta=pt(1);bool okay=true;
 for(int j=0;j<n;j++){const double* p=packets+(2*(start+j)+mode)*WIDTH;if(p[0]!=0){okay=false;break;}beta=mul(beta,sub(pt(1),{p[24],p[25]}));}
 double* out=output+3*id;out[0]=!okay?0:beta.hi<.5?1:beta.lo>.5?3:beta.lo==.5&&beta.hi==.5?2:0;out[1]=beta.lo;out[2]=beta.hi;
}

__global__ void endpoints(int count,const double* packets,const int* sections,const double* t,double* output){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=count*2)return;int query=id/2,mode=id%2;int start=sections[2*query],n=sections[2*query+1];gi::Layer c[192];
 double* out=output+4*id;if(n>192){out[3]=0;return;}
 for(int j=0;j<n;j++){const double* p=packets+(2*(start+j)+mode)*WIDTH;if(p[0]!=0){out[3]=0;return;}c[j]={{p[1],p[2]},{p[3],p[4]},{p[5],p[6]},p[7]};}
 double scale;bool okay;I r=gi::residual(c,n,t[query],scale,okay);out[0]=r.lo;out[1]=r.hi;out[2]=scale;out[3]=okay?1.:0.;
}
} // namespace gp
