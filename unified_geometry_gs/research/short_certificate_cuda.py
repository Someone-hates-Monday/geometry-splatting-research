"""Generate one rigorously bounded short-series certificate with strict fallback."""
from pathlib import Path
import hashlib, re

def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)

def short_header(base):
    body = base[base.index('GI_INLINE I log_one(double x){'):]
    body = once(body, '} // namespace gi', '} // namespace gf')
    body = once(body, 'struct Layer {I p,rho,energy;double amplitude;};', 'using Layer=gi::Layer;')
    starts = [body.index('GI_INLINE I '+name) for name in ('log_one(', 'log1p(', 'exp_one(', 'expm1(')]
    ends = [body.index('GI_INLINE I log(I'), body.index('GI_INLINE I exp_one('),
            body.index('GI_INLINE I exp(I'), body.index('using Layer=')]
    pieces = [(starts[i], ends[i], i) for i in range(4)]
    for start, end, which in sorted(pieces, reverse=True):
        part = body[start:end]
        if which == 0:
            part = once(part, 'j<25', 'j<9')
            part = once(part, 'pt(51)', 'pt(19)')
        elif which == 1:
            part = once(part, 'k<=24', 'k<=8')
            part = once(part, 'power(pt(r),25)', 'power(pt(r),9)')
            part = once(part, 'pt(25)', 'pt(9)')
        elif which == 2:
            part = once(part, 'j<=24', 'j<=12')
            part = once(part, 'j<=25', 'j<=13')
        else:
            part = once(part, 'j<=24', 'j<=8')
            part = once(part, 'j<=25', 'j<=9')
        body = body[:start] + part + body[end:]
    # I/Layer are gi types: qualify local functions to prevent argument-dependent
    # lookup from selecting the long-series functions in namespace gi.
    names = ['log_one','log','log1p','exp_one','exp','expm1','log_tail','prefix','scale_value','residual']
    for name in names:
        body = re.sub(r'\b'+name+r'\(', 'gf::'+name+'(', body)
        body = body.replace('GI_INLINE I gf::'+name+'(', 'GI_INLINE I '+name+'(')
    imported = ['I','pt','add','neg','sub','mul','div','div_integer','sq','ln2','valid','entire','power']
    return '#pragma once\n#include "guarded_interval.cuh"\nnamespace gf {\n' + ''.join('using gi::'+n+';\n' for n in imported) + body

def variant_root(base):
    text = once(base, '#include "guarded_preprocess.cuh"', '#include "guarded_preprocess.cuh"\n#include "short_interval.cuh"')
    start = text.index('__device__ __forceinline__ bool certify(')
    end = text.index('__global__ void solve(', start)
    fast = text[start:end].replace('bool certify(', 'bool certify_fast(').replace('gi::residual(', 'gf::residual(')
    text = text[:end] + fast + text[end:]
    text = once(text, 'double* out=output+12*model;', 'double* out=output+14*model;')
    text = once(text, 'out[0]=0;if(n<=0', 'out[0]=0;out[12]=0;out[13]=0;if(n<=0')
    text = once(text, 'if(certify(bounds,n,z,out)){out[0]=1;return;}',
        'if(certify_fast(bounds,n,z,out)){out[0]=1;out[12]=1;return;}\n'
        '  out[13]+=1;\n'
        '  if(certify(bounds,n,z,out)){out[0]=1;out[12]=2;return;}')
    return text

def load(out, old, variant):
    import torch
    from torch.utils.cpp_extension import load as compile_extension
    from research.guarded_interval_cuda import FLAGS
    build = out/('variant_generated' if variant else 'control_generated')
    build.mkdir()
    source = {p.name:p.read_text(encoding='utf-8') for p in (old/'variant_generated').iterdir() if p.suffix in ('.cu','.cpp','.cuh')}
    assert set(source) == {'root.cu','binding.cpp','guarded_interval.cuh','guarded_preprocess.cuh','guarded_root.cuh'}
    if variant:
        source['short_interval.cuh'] = short_header(source['guarded_interval.cuh'])
        source['guarded_root.cuh'] = variant_root(source['guarded_root.cuh'])
        source['binding.cpp'] = once(source['binding.cpp'], 'torch::full({b.size(0),12}', 'torch::full({b.size(0),14}')
        source['root.cu'] += r'''
__global__ void scalar_kernel(const double* input,double* output,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int op=int(input[2*i]);gi::I x=gi::pt(input[2*i+1]);gi::I value;
 if(op==0)value=gf::log(x);else if(op==1)value=gf::exp(x);else if(op==2)value=gf::log1p(x);else if(op==3)value=gf::expm1(x);else value=gf::add(gi::pt(1),x);
 output[2*i]=value.lo;output[2*i+1]=value.hi;
}
void scalars(const double* a,double* b,int n,cudaStream_t s){scalar_kernel<<<(n+63)/64,64,0,s>>>(a,b,n);}
'''
        marker = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)'
        declarations = r'''
void scalars(const double*,double*,int,cudaStream_t);
torch::Tensor scalar_run(torch::Tensor a){
 TORCH_CHECK(a.is_cuda()&&a.is_contiguous()&&a.scalar_type()==torch::kFloat64&&a.dim()==2&&a.size(1)==2,"Scalar CUDA double op/value rows required");
 c10::cuda::CUDAGuard guard(a.device());auto out=torch::full({a.size(0),2},NAN,a.options());scalars(a.data_ptr<double>(),out.data_ptr<double>(),a.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
'''
        source['binding.cpp'] = once(source['binding.cpp'], marker, declarations+marker)
        source['binding.cpp'] = once(source['binding.cpp'], 'm.def("solve",&run);', 'm.def("solve",&run);m.def("scalars",&scalar_run);')
    h = hashlib.sha256((torch.__version__+str(FLAGS)).encode())
    for name,text in sorted(source.items()):
        h.update(name.encode());h.update(text.encode());(build/name).write_text(text,encoding='utf-8')
    return compile_extension(name=('short_certificate_variant_' if variant else 'short_certificate_control_')+h.hexdigest()[:12],
        sources=[str(build/'binding.cpp'),str(build/'root.cu')],extra_cflags=['-O3'],extra_cuda_cflags=FLAGS,build_directory=str(build),verbose=True)
