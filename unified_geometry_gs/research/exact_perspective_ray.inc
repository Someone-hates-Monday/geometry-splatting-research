// Inference diagnostic only: per-ray coefficients, unchanged affine selection.
struct ExactPerspectiveRay {
    float peak, reciprocal_sigma, alpha;
    bool valid;
};
__device__ __forceinline__ ExactPerspectiveRay exact_packet_ray(
    float2 xy,float4 plane,float4 co,float2 query,int W,int H,float fx,float fy) {
    double mx=(double(xy.x)-double(W-1)/2)/double(fx);
    double my=(double(xy.y)-double(H-1)/2)/double(fy);
    double u=fmin(1.3*double(W)/(2*fx),fmax(-1.3*double(W)/(2*fx),mx));
    double v=fmin(1.3*double(H)/(2*fy),fmax(-1.3*double(H)/(2*fy),my));
    double z=double(plane.z)/sqrt(mx*mx+my*my+1),r=u*u+v*v+1;
    double b=r*double(plane.w)*double(plane.w),scale=z*z*b;
    double cx=(double(plane.x)*fx*sqrt(r)/z+u)/r;
    double cy=(double(plane.y)*fy*sqrt(r)/z+v)/r;
    double kx=double(co.x)*fx*fx/scale,ky=double(co.z)*fy*fy/scale,kxy=double(co.y)*fx*fy/scale;
    double ax=double(query.x)-u,ay=double(query.y)-v,dx=mx-u,dy=my-v;
    double A=1+cx*ax+cy*ay,B=1+cx*dx+cy*dy;
    double aa=kx*ax*ax+2*kxy*ax*ay+ky*ay*ay;
    double am=ax*(kx*dx+kxy*dy)+ay*(kxy*dx+ky*dy);
    double den=A*A+aa,cross=A*B+am;
    double vx=A*dx-B*ax,vy=A*dy-B*ay,wedge=ax*dy-ay*dx;
    double det=kx*ky-kxy*kxy;
    double gram=det*wedge*wedge+kx*vx*vx+2*kxy*vx*vy+ky*vy*vy;
    double energy=scale*gram/den,qn=sqrt(double(query.x)*query.x+double(query.y)*query.y+1);
    double p=z*qn*cross/den,rs=sqrt(b*den)/qn,alpha=double(co.w)*exp(-energy/2);
    bool valid=det>0&&den>0&&energy>=0&&isfinite(p)&&isfinite(rs)&&rs>0&&isfinite(alpha)&&alpha>=0;
    return {float(p),float(rs),fminf(.99f,float(alpha)),valid};
}
__device__ __forceinline__ ExactPerspectiveRay exact_factor_ray(
    const float* f,float4 co,float2 query) {
    double qn=sqrt(double(query.x)*query.x+double(query.y)*query.y+1);
    double dx=double(query.x)/qn,dy=double(query.y)/qn,dz=1/qn;
    double ox=double(f[0])*f[9]+double(f[3])*f[10]+double(f[6])*f[11];
    double oy=double(f[1])*f[9]+double(f[4])*f[10]+double(f[7])*f[11];
    double oz=double(f[2])*f[9]+double(f[5])*f[10]+double(f[8])*f[11];
    double ux=double(f[0])*dx+double(f[3])*dy+double(f[6])*dz;
    double uy=double(f[1])*dx+double(f[4])*dy+double(f[7])*dz;
    double uz=double(f[2])*dx+double(f[5])*dy+double(f[8])*dz;
    double a=ux*ux+uy*uy+uz*uz,p=(ox*ux+oy*uy+oz*uz)/a;
    double vx=ox-p*ux,vy=oy-p*uy,vz=oz-p*uz,energy=vx*vx+vy*vy+vz*vz;
    double rs=sqrt(a),alpha=double(co.w)*exp(-energy/2);
    bool valid=a>0&&isfinite(p)&&isfinite(rs)&&rs>0&&isfinite(alpha)&&alpha>=0;
    return {float(p),float(rs),fminf(.99f,float(alpha)),valid};
}
