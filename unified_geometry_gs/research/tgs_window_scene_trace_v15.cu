// Local research observer of the pinned TGS alpha compositor.
// Original algorithm provenance/licenses: sources/tgs_native_diagnostic_v1/native/
// LICENSE.md and LICENSE_GS.md. No production renderer is modified.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void replay_kernel(const float* p,int N,int X,int Y,bool no_stop,double* trace,float* summary){
    float T=1.0f,C[3]={0,0,0};int last=0,stop=0;
    for(int j=0;j<N;++j){
        const float* q=p+17*j;float max_val=-INFINITY;bool outside=false;
        for(int k=0;k<3;++k){
            float dist=(q[2*k]*(float)X+q[2*k+1]*(float)Y+q[6+k]);
            if(dist>0){outside=true;break;}
            max_val=fmaxf(max_val,dist);
        }
        float u=max_val*q[9];float cx=outside?0.0f:fmaxf(0.0f,__powf(u,q[10]));
        float alpha=fminf(.99f,q[11]*cx);float incoming=T,test=T;bool included=false,terminated=false;
        if(!outside && alpha>=1.0f/255.0f){
            test=T*(1-alpha);
            if(!no_stop && test<.0001f){terminated=true;stop=j+1;}
            else{
                for(int ch=0;ch<3;++ch)C[ch]+=q[13+ch]*alpha*T;
                T=test;last=j+1;included=true;
            }
        }
        double values[16]={(double)q[16],(double)(j+1),(double)max_val,(double)u,(double)alpha,
            (double)outside,(double)incoming,(double)test,(double)included,(double)terminated,
            (double)T,(double)C[0],(double)C[1],(double)C[2],(double)q[12],(double)cx};
        for(int k=0;k<16;++k)trace[j*16+k]=values[k];
        if(terminated)break;
    }
    for(int k=0;k<3;++k)summary[k]=C[k];summary[3]=1-T;summary[4]=last;summary[5]=stop;
}

std::vector<torch::Tensor> replay(torch::Tensor p,int X,int Y,bool no_stop){
    TORCH_CHECK(p.is_cuda()&&p.is_contiguous()&&p.scalar_type()==torch::kFloat32);
    TORCH_CHECK(p.dim()==2&&p.size(1)==17);
    auto trace=torch::full({p.size(0),16},NAN,p.options().dtype(torch::kFloat64));
    auto summary=torch::zeros({6},p.options());
    replay_kernel<<<1,1>>>(p.data_ptr<float>(),p.size(0),X,Y,no_stop,trace.data_ptr<double>(),summary.data_ptr<float>());
    return {trace,summary};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("replay",&replay);}
