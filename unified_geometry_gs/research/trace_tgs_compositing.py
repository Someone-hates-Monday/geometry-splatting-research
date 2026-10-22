"""Trace two frozen real pixels with the exact v3 window helper on CUDA."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
from benchmark_tgs_backward import ROOT,Workload
OUT=ROOT/'experiments/tgs_compositing_trace_v1'
OLD=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def align(n):return (n+127)//128*128

def build():
    base=OUT/'build';base.mkdir(parents=True,exist_ok=True)
    header=(ROOT/'tgs_adaptive_window.cuh').read_text();body=header[header.index('struct StableWindow'):]
    source='#include <torch/extension.h>\n#include <cuda_runtime.h>\n#include <math_constants.h>\n'
    source+='namespace A {\n'+body.replace('STABLE_MODE','2')+'\n}\nnamespace B {\n'+body.replace('STABLE_MODE','3')+'\n}\n'
    source+=r'''
template<int MODE> __global__ void trace_kernel(const float* a,const double* coeff,const int N,const int X,const int Y,double* out,float* summary) {
    float T=1.0f,C[3]={0,0,0},D=0.0f,median=0.0f;int last=0,stop=0,median_id=0;
    for(int j=0;j<N;++j) {
        const float* p=a+29*j;
        float2 n[3]={make_float2(p[0],p[1]),make_float2(p[2],p[3]),make_float2(p[4],p[5])};
        float offsets[3]={p[6],p[7],p[8]};
        float phi,u,cx,alpha;bool outside,high,uncertain,capped;
        if(MODE==2) {
            auto w=A::stable_window(n,offsets,p+18,coeff+10*j,make_float2(p[9],p[10]),make_float2(X,Y),p[11],p[12],p[13]);
            phi=w.phi;u=w.u;cx=w.Cx;alpha=w.alpha;outside=w.outside;high=w.high;uncertain=w.uncertain;capped=w.capped;
        } else {
            auto w=B::stable_window(n,offsets,p+18,coeff+10*j,make_float2(p[9],p[10]),make_float2(X,Y),p[11],p[12],p[13]);
            phi=w.phi;u=w.u;cx=w.Cx;alpha=w.alpha;outside=w.outside;high=w.high;uncertain=w.uncertain;capped=w.capped;
        }
        const float incoming=T;float test=T;bool included=false,terminated=false;
        if(!outside && alpha>=1.0f/255.0f) {
            test=T*(1-alpha);
            if(test<.0001f) {terminated=true;stop=j+1;}
            else {
                const float blending_weight=alpha*T;
                D+=p[14]*blending_weight;
                if(T>.5f) {median=p[14];median_id=j+1;}
                for(int ch=0;ch<3;++ch) C[ch]+=p[15+ch]*alpha*T;
                T=test;included=true;last=j+1;
            }
        }
        double row[24]={(double)p[28],(double)(j+1),(double)phi,(double)u,(double)cx,(double)alpha,
            (double)outside,(double)high,(double)uncertain,(double)capped,(double)incoming,(double)test,
            (double)included,(double)terminated,(double)T,(double)p[14],(double)C[0],(double)C[1],(double)C[2],(double)D,(double)median,
            (double)(alpha*incoming),(double)p[12],(double)p[11]};
        for(int k=0;k<24;++k)out[24*j+k]=row[k];
        if(terminated)break;
    }
    for(int k=0;k<3;++k)summary[k]=C[k];summary[3]=D;summary[4]=1-T;summary[5]=median;
    summary[6]=last;summary[7]=stop;summary[8]=median_id;
}
std::vector<torch::Tensor> trace(torch::Tensor input,torch::Tensor coeff,int X,int Y,int mode) {
    TORCH_CHECK(input.is_cuda() && coeff.is_cuda() && input.is_contiguous() && coeff.is_contiguous());
    TORCH_CHECK(input.scalar_type()==torch::kFloat32 && coeff.scalar_type()==torch::kFloat64);
    TORCH_CHECK(input.dim()==2 && input.size(1)==29 && coeff.size(0)==input.size(0) && coeff.size(1)==10);
    auto out=torch::full({input.size(0),24},NAN,input.options().dtype(torch::kFloat64));
    auto summary=torch::zeros({9},input.options());
    if(mode==2)trace_kernel<2><<<1,1>>>(input.data_ptr<float>(),coeff.data_ptr<double>(),input.size(0),X,Y,out.data_ptr<double>(),summary.data_ptr<float>());
    else trace_kernel<3><<<1,1>>>(input.data_ptr<float>(),coeff.data_ptr<double>(),input.size(0),X,Y,out.data_ptr<double>(),summary.data_ptr<float>());
    return {out,summary};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("trace",&trace);}
'''
    p=base/'trace.cu';assert not p.exists();p.write_text(source)
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    module=load(name='tgs_diag_compositing_trace_v1',sources=[str(p)],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(base),verbose=True)
    (OUT/'build_receipt.json').write_text(json.dumps(dict(module_file=module.__file__,module_sha256=sha(Path(module.__file__)),source_sha256=sha(p),helper_sha256=sha(ROOT/'tgs_adaptive_window.cuh'),script_sha256=sha(Path(__file__)),training=False),indent=2)+'\n')
    return module

def gather(res,work,ci,x,y):
    n=work.n;c=work.cameras[ci];hw=c['h']*c['w'];tile=(y//16)*((c['w']+15)//16)+(x//16)
    rangesoff=align(align(12*hw)+8*hw)
    start,end=res[6].narrow(0,rangesoff+8*tile,8).view(torch.int32).cpu().numpy().tolist()
    ids=res[5].narrow(0,4*start,4*(end-start)).view(torch.int32).long()
    raw=res[4]
    def get(offset,dtype,width):return raw.narrow(0,offset,torch.empty((),dtype=dtype).element_size()*n*width).view(dtype).reshape(n,width).index_select(0,ids)
    pos=0;fields={}
    for name,size in [('depth',4),('clamped',3),('radii',4),('anchor',8),('conic',16),('phi_center',8),('rgb',12)]:
        pos=align(pos);fields[name]=pos;pos+=size*n
    maxoff=raw.numel()-128-8*n;minoff=maxoff-align(8*n);counts=minoff-align(12*n);errors=counts-align(40*n);coeff=errors-align(80*n);pw=coeff-align(12*n);normals=pw-align(24*n);offsets=normals-align(12*n)
    packed=torch.cat([get(normals,torch.float32,6),get(offsets,torch.float32,3),get(fields['anchor'],torch.float32,2),work.sigma.reshape(n,1).index_select(0,ids),get(fields['conic'],torch.float32,4)[:,3:],-get(fields['phi_center'],torch.float32,2)[:,:1],get(fields['depth'],torch.float32,1),get(fields['rgb'],torch.float32,3),get(errors,torch.float32,10),ids.float()[:,None]],1).contiguous()
    double=get(coeff,torch.float64,10).contiguous()
    offset=align(12*hw);last=int(res[6].narrow(0,offset+4*(y*c['w']+x),4).view(torch.int32)[0]);median=int(res[6].narrow(0,offset+4*(hw+y*c['w']+x),4).view(torch.int32)[0])
    native=np.r_[res[1][:,y,x].cpu().numpy(),float(res[2][0,y,x]),float(res[2][1,y,x]),float(res[2][5,y,x]),last,median]
    return packed,double,native,dict(tile=tile,start=start,end=end,ids=ids.cpu().numpy())

def main():
    assert not OUT.exists();OUT.mkdir()
    old=json.loads((OLD/'real_independent_audit.json').read_text());pixels=[(0,*old['rows'][0]['aux_channels'][0]['worst_pixel']),(1,*old['rows'][1]['aux_channels'][0]['worst_pixel'])]
    (OUT/'plan.json').write_text(json.dumps(dict(pixels=pixels,selection='v3 frozen weighted-depth worst pixels, before new tracing',old_real_audit_sha256=sha(OLD/'real_independent_audit.json'),script_sha256=sha(Path(__file__)),training=False,timing=False,semantics='existing skip-current unchanged'),indent=2)+'\n')
    module=build();mods={}
    for v in ['adaptive','oracle64']:
        r=json.loads((OLD/v/'build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256']
        s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);mods[v]=m
    work=Workload();rows=[]
    for ci,x,y in pixels:
        endpoints={}
        for variant,mode in [('adaptive',2),('oracle64',3)]:
            res=work.forward(mods[variant],ci);torch.cuda.synchronize();packed,coeff,native,metadata=gather(res,work,ci,x,y)
            trace,summary=module.trace(packed,coeff,x,y,mode);torch.cuda.synchronize();trace=trace.cpu().numpy();summary=summary.cpu().numpy()
            expected=summary[[0,1,2,3,4,5,6,8]];error=float(np.abs(expected-native).max())
            valid=trace[np.isfinite(trace[:,0])];endpoint=dict(camera=ci,pixel=[x,y],variant=variant,candidates=len(packed),evaluated=len(valid),native=native.tolist(),replayed=expected.tolist(),endpoint_max_abs=error,endpoint_bitwise_equal=bool(np.array_equal(expected.astype(np.float32),native.astype(np.float32))),stop_ordinal=int(summary[7]),last_ordinal=int(summary[6]),median_ordinal=int(summary[8]))
            rows.append(endpoint);endpoints[variant]=endpoint
            np.savez_compressed(OUT/f'camera_{ci}_{variant}.npz',packed=packed.cpu().numpy(),double_coeff=coeff.cpu().numpy(),trace=trace,summary=summary,native=native,ids=metadata['ids'])
            (OUT/f'camera_{ci}_{variant}_trace.json').write_text(json.dumps(dict(columns=['triangle_id','ordinal','phi','u','Cx','alpha','outside','high','uncertain','capped','incoming_T','test_T','included','terminated','after_T','depth','R','G','B','weighted_depth','median_depth','potential_weight','opacity','sigma'],rows=valid.tolist()),indent=2)+'\n')
            print(json.dumps(endpoint),flush=True);del res,packed,coeff
    (OUT/'results.json').write_text(json.dumps(dict(plan_sha256=sha(OUT/'plan.json'),rows=rows,all_endpoints_bitwise_equal=all(r['endpoint_bitwise_equal'] for r in rows),training=False),indent=2)+'\n')
if __name__=='__main__':main()
