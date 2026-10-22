"""All14 frozen depth violations, exact endpoints and predeclared power/geometry swaps."""
import hashlib,json,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_compensated_common import ROOT,OUT,sha,align,offsets,modules,workload
TRACE=OUT/'depth_trace';MODES=['guard','oracle','float_u_double_power','native_u_fast_power','float_u_standard_power','native_u_standard_power','pair_phi_double_power']
def build():
    d=TRACE/'build';d.mkdir();a=(ROOT/'tgs_compensated_window_v8.cuh').read_text();a=a[a.index('struct StableWindow'):].replace('STABLE_MODE','2');b=(ROOT/'experiments/tgs_phase_cost_v5/generated/native/cuda_rasterizer/stable_window.cuh').read_text();b=b[b.index('struct StableWindow'):].replace('STABLE_MODE','3')
    source='#include <torch/extension.h>\n#include <cuda_runtime.h>\n#include <math_constants.h>\nnamespace A {\n'+a+'\n}\nnamespace B {\n'+b+'\n}\n'
    source+=r'''
__device__ double pair_dot(const float* a,float x,float y){float p=0,s=0;for(int k=0;k<6;k++){float v=k%3==0?x:(k%3==1?y:1);float h=__fmul_rn(a[k],v),r=__fmaf_rn(a[k],v,-h);if(k==0){p=h;s=r;}else{float q;A::comp_two_sum(p,h,p,q);s=__fadd_rn(s,__fadd_rn(q,r));}}return __dadd_rn((double)p,(double)s);}
__device__ float alpha64(double u,float s,float o){return __double2float_rn(fmin(.99,__dmul_rn((double)o,pow(fmax(0.,u),(double)s))));}
__global__ void trace_kernel(const float* in,const double* coeff,int N,int X,int Y,int mode,double* out,float* summary){
    float T=1,C[3]={0},D=0,median=0;int last=0,stop=0,mid=-1;
    for(int j=0;j<N;j++){
        const float* p=in+57*j;const double* c=coeff+10*j;float2 an=make_float2(p[0],p[1]),px=make_float2(X,Y);
        auto ga=A::stable_window(nullptr,nullptr,p+8,c,an,px,p[2],p[3],0,false);auto ob=B::stable_window(nullptr,nullptr,nullptr,c,an,px,p[2],p[3],0,false);
        double dx=__dadd_rn((double)X,-(double)an.x),dy=__dadd_rn((double)Y,-(double)an.y),best=-CUDART_INF;
        for(int k=0;k<3;k++){double d=__dadd_rn(__dadd_rn(__dmul_rn(c[2*k],dx),__dmul_rn(c[2*k+1],dy)),c[6+k]);if(d>best)best=d;}
        double un=fmax(0.,__dmul_rn(-best,c[9]));float unf=__double2float_rn(un);float stdcx=powf(fmaxf(ga.u,0.f),p[2]);
        float alternatives[7]={ga.alpha,ob.alpha,alpha64((double)ga.u,p[2],p[3]),fminf(.99f,__fmul_rn(p[3],__powf(unf,p[2]))),fminf(.99f,__fmul_rn(p[3],stdcx)),fminf(.99f,__fmul_rn(p[3],powf(unf,p[2]))),0};
        double bp=-CUDART_INF;for(int k=0;k<3;k++){double d=pair_dot(p+18+6*k,(float)dx,(float)dy);if(d>bp)bp=d;}alternatives[6]=alpha64(__dmul_rn(-bp,c[9]),p[2],p[3]);
        // cache starts p+8; its18hi/lo start p+18. No counterfactual changes native-fallback windows.
        float alpha=mode==1?ob.alpha:(ga.high?ga.alpha:alternatives[mode]);bool outside=mode==1?ob.outside:ga.outside;
        float incoming=T,test=T;bool include=false,terminate=false;float weight=__fmul_rn(alpha,T);
        if(!outside&&alpha>=1.f/255.f){test=__fmul_rn(T,__fsub_rn(1.f,alpha));if(test<.0001f){terminate=true;stop=j+1;}else{D=__fmaf_rn(p[4],weight,D);if(T>.5f){median=p[4];mid=j+1;}for(int k=0;k<3;k++)C[k]=__fmaf_rn(__fmul_rn(p[5+k],alpha),T,C[k]);T=test;last=j+1;include=true;}}
        double row[36]={(double)p[56],(double)(j+1),(double)(mode==1?ob.phi:ga.phi),(double)(mode==1?ob.u:ga.u),(double)(mode==1?ob.Cx:ga.Cx),(double)alpha,(double)outside,(double)(mode==1?ob.high:ga.high),(double)(mode==1?ob.active:ga.active),(double)incoming,(double)test,(double)include,(double)terminate,(double)T,(double)p[4],(double)C[0],(double)C[1],(double)C[2],(double)D,(double)median,(double)weight,un,best,(double)ga.alpha,(double)ob.alpha,(double)alternatives[2],(double)alternatives[3],(double)alternatives[4],(double)alternatives[5],(double)alternatives[6],(double)ga.u,(double)unf,(double)ga.Cx,(double)stdcx,(double)alpha,bp};
        for(int k=0;k<36;k++)out[j*36+k]=row[k];if(terminate)break;
    }
    for(int k=0;k<3;k++)summary[k]=__fmaf_rn(T,0.f,C[k]);summary[3]=D;summary[4]=__fsub_rn(1.f,T);summary[5]=median;summary[6]=last;summary[7]=stop;summary[8]=mid;
}
std::vector<torch::Tensor> trace(torch::Tensor a,torch::Tensor c,int X,int Y,int mode){
    TORCH_CHECK(a.is_cuda()&&c.is_cuda()&&a.is_contiguous()&&c.is_contiguous()&&a.scalar_type()==torch::kFloat32&&c.scalar_type()==torch::kFloat64&&a.dim()==2&&a.size(1)==57&&c.size(0)==a.size(0)&&c.size(1)==10&&mode>=0&&mode<7);
    auto rows=torch::full({a.size(0),36},NAN,a.options().dtype(torch::kFloat64));auto summary=torch::zeros({9},a.options());trace_kernel<<<1,1>>>(a.data_ptr<float>(),c.data_ptr<double>(),a.size(0),X,Y,mode,rows.data_ptr<double>(),summary.data_ptr<float>());TORCH_CHECK(cudaGetLastError()==cudaSuccess);return{rows,summary};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("trace",&trace);}
'''
    src=d/'trace.cu';src.write_text(source);os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2');os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    from torch.utils.cpp_extension import load
    flags=['-O3','--ftz=false','--prec-div=true','--prec-sqrt=true'];m=load(name='tgs_diag_compensated_depth_trace_v8',sources=[str(src)],extra_cflags=['-O3'],extra_cuda_cflags=flags,build_directory=str(d),verbose=True)
    (TRACE/'build_receipt.json').write_text(json.dumps(dict(module_file=m.__file__,module_sha256=sha(Path(m.__file__)),source_sha256=sha(src),script_sha256=sha(Path(__file__)),flags=flags),indent=2)+'\n');return m
def gather(r,work,ci,x,y):
    n=work.n;cam=work.cameras[ci];w,h=cam['w'],cam['h'];hw=w*h;tile=(y//16)*((w+15)//16)+x//16;roff=align(align(12*hw)+8*hw);start,end=r[6].narrow(0,roff+8*tile,8).view(torch.int32).cpu().numpy().tolist();ids=r[5].narrow(0,4*start,4*(end-start)).view(torch.int32).long();raw=r[4]
    pos=0;field={}
    for name,size in [('depth',4),('clamped',3),('radii',4),('anchor',8),('conic',16),('phi_center',8),('rgb',12)]:pos=align(pos);field[name]=pos;pos+=size*n
    mx,mn,count,err,coef,pre=offsets(raw.numel(),n)
    def get(off,dtype,width):return raw.narrow(0,off,n*width*torch.empty((),dtype=dtype).element_size()).view(dtype).reshape(n,width).index_select(0,ids)
    packed=torch.cat([get(field['anchor'],torch.float32,2),work.sigma.reshape(n,1).index_select(0,ids),get(field['conic'],torch.float32,4)[:,3:],get(field['depth'],torch.float32,1),get(field['rgb'],torch.float32,3),get(err,torch.float32,48),ids.float()[:,None]],1).contiguous();cc=get(coef,torch.float64,10).contiguous();pathoff=align(12*hw);p=y*w+x
    paths=[int(r[6].narrow(0,pathoff+4*(p+k*hw),4).view(torch.int32)[0]) for k in [0,1]];expected=np.r_[r[1][:,y,x].cpu().numpy(),r[2][[0,1,5],y,x].cpu().numpy(),paths].astype(np.float32)
    flag=int(r[6].narrow(0,r[6].numel()-128-4*hw+4*p,4).view(torch.int32)[0]);return packed,cc,expected,flag
def main():
    assert not TRACE.exists();TRACE.mkdir();problem=json.loads((OUT/'depth_failure_plan.json').read_text());bad=[(r['camera'],p) for r in problem['rows'] for p in r['bad']];assert len(bad)==14
    (TRACE/'plan.json').write_text(json.dumps(dict(protocol_sha256=sha(ROOT/'TGS_COMPENSATED_DEPTH_TRACE_PROTOCOL.md'),failure_plan_sha256=sha(OUT/'depth_failure_plan.json'),helper_sha256=sha(ROOT/'tgs_compensated_window_v8.cuh'),script_sha256=sha(Path(__file__)),pixels=[dict(camera=ci,**p) for ci,p in bad],modes=MODES,training=False,timing=False),indent=2)+'\n')
    mod=build();mods=modules();work=workload();ci=bad[0][0];assert all(c==ci for c,p in bad);renders={v:work.forward(mods[v],ci) for v in ['guard','oracle']};torch.cuda.synchronize();rows=[]
    for pi,(ci,pixel) in enumerate(bad):
        x,y=pixel['pixel'];a,c,ga,flag=gather(renders['guard'],work,ci,x,y);b,oc,ob,of=gather(renders['oracle'],work,ci,x,y);assert flag==0 and of==0 and torch.equal(a,b) and torch.equal(c,oc)
        assert float(ga[3])==pixel['guard'] and float(ob[3])==pixel['oracle'];np.savez_compressed(TRACE/f'packet_{pi}.npz',packed=a.cpu().numpy(),coeff=c.cpu().numpy(),guard_endpoint=ga,oracle_endpoint=ob,pixel=np.array([x,y]))
        outcomes=[]
        for mode,name in enumerate(MODES):
            trace,summary=mod.trace(a,c,x,y,mode);torch.cuda.synchronize();tr=trace.cpu().numpy();s=summary.cpu().numpy();np.savez_compressed(TRACE/f'pixel_{pi}_{name}.npz',trace=tr,summary=s)
            if mode in [0,1]:assert np.array_equal(s[[0,1,2,3,4,5,6,8]],ga if mode==0 else ob),('endpoint',pi,name,s,ga,ob)
            outcomes.append(dict(mode=name,weighted_depth=float(s[3]),difference_oracle=float(s[3]-ob[3]),last=int(s[6]),stop=int(s[7]),median=int(s[8]),RGB_abs=float(np.abs(s[:3]-ob[:3]).max())))
        row=dict(index=pi,camera=ci,pixel=[x,y],packet_rows=len(a),endpoints_bitwise=True,outcomes=outcomes);rows.append(row);(TRACE/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
    (TRACE/'results.json').write_text(json.dumps(dict(rows=rows,all_endpoints_bitwise=True,plan_sha256=sha(TRACE/'plan.json'),training=False,timing=False),indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except BaseException:
        (TRACE/'failure.txt').write_text(traceback.format_exc());raise
