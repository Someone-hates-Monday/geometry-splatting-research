// Independent camera-factor construction from production activated parameters.
__global__ void exact_factor_preprocess(int P,const float3* means,const glm::vec3* scales,
    const float4* rotations,const float* view,float mod,float* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=P)return;
    float4 q=rotations[i];float w=q.x,x=q.y,y=q.z,z=q.w;
    glm::mat3 R=glm::mat3(
        1.f-2.f*(y*y+z*z),2.f*(x*y-w*z),2.f*(x*z+w*y),
        2.f*(x*y+w*z),1.f-2.f*(x*x+z*z),2.f*(y*z-w*x),
        2.f*(x*z-w*y),2.f*(y*z+w*x),1.f-2.f*(x*x+y*y));
    glm::mat3 W=glm::mat3(view[0],view[4],view[8],view[1],view[5],view[9],view[2],view[6],view[10]);
    glm::mat3 S=glm::mat3(1.f);
    S[0][0]=__frcp_rn(mod*scales[i].x);S[1][1]=__frcp_rn(mod*scales[i].y);S[2][2]=__frcp_rn(mod*scales[i].z);
    glm::mat3 F=S*R*W;float* f=out+12*i;
    for(int c=0;c<3;c++)for(int r=0;r<3;r++)f[3*c+r]=F[c][r];
    float3 t=transformPoint4x3(means[i],view);f[9]=t.x;f[10]=t.y;f[11]=t.z;
}
void FORWARD::build_exact_factors(int P,const float* means,const float* scales,
    const float* rotations,const float* view,float mod,float* out) {
    exact_factor_preprocess<<<(P+255)/256,256>>>(P,(const float3*)means,(const glm::vec3*)scales,
        (const float4*)rotations,view,mod,out);
}
