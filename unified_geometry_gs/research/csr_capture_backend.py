"""Private full-image CSR capture at the existing actual acceptance site."""
from pathlib import Path
import hashlib

def once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)
def section(text,start,end,transform):
    a=text.index(start);b=text.index(end,a)+len(end)
    return text[:a]+transform(text[a:b])+text[b:]

def payloads(base):
    data=dict(base)
    tensors=',\n    const torch::Tensor& csr_offsets,\n    const torch::Tensor& csr_records,\n    const torch::Tensor& csr_status'
    def native(text):
        text=once(text,'const bool debug) {','const bool debug'+tensors+') {')
        guard='''
    TORCH_CHECK(require_depth,"CSR capture requires depth");
    for(auto t:{csr_offsets,csr_records,csr_status})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==means3D.device(),"Same-device contiguous CSR tensors required");
    TORCH_CHECK(csr_offsets.scalar_type()==torch::kInt64&&csr_offsets.dim()==1&&csr_offsets.numel()==int64_t(image_height)*image_width+1,"CSR offsets shape/dtype invalid");
    TORCH_CHECK(csr_records.scalar_type()==torch::kInt32&&csr_records.dim()==2&&csr_records.size(1)==2,"CSR records must be int32 pairs");
    TORCH_CHECK(csr_status.scalar_type()==torch::kInt32&&csr_status.dim()==2&&csr_status.size(0)==image_height&&csr_status.size(1)==image_width,"CSR status shape/dtype invalid");
    c10::cuda::CUDAGuard csr_guard(means3D.device());
'''
        text=once(text,'    const int P = means3D.size(0);',guard+'    const int P = means3D.size(0);')
        return once(text,'            debug);','            debug,\n            csr_offsets.data_ptr<int64_t>(),csr_records.data_ptr<int>(),csr_status.data_ptr<int>(),csr_records.size(0));')
    data['rasterize_points.cu']=once(data['rasterize_points.cu'],'#include <torch/extension.h>','#include <torch/extension.h>\n#include <c10/cuda/CUDAGuard.h>')
    data['rasterize_points.cu']=section(data['rasterize_points.cu'],'RasterizeGaussiansCUDA(',
        'return std::make_tuple(rendered, out_color, out_alpha, out_normal, out_mdepth, radii, geomBuffer, binningBuffer, imgBuffer, tileBuffer);\n}',native)
    data['rasterize_points.h']=section(data['rasterize_points.h'],'RasterizeGaussiansCUDA(','const bool debug);',
        lambda t:once(t,'const bool debug);','const bool debug'+tensors+');'))
    validate_decl='torch::Tensor CSRValidateCUDA(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,int,int);\n'
    data['rasterize_points.h']+='\n'+validate_decl
    data['ext.cpp']=once(data['ext.cpp'],'  m.def("mark_visible", &markVisible);','  m.def("mark_visible", &markVisible);\n  m.def("validate_csr", &CSRValidateCUDA);')
    pointer=',\n    const int64_t* topology_offsets, int* topology_records, int* topology_status, int64_t topology_capacity'
    data['cuda_rasterizer/rasterizer.h']=section(data['cuda_rasterizer/rasterizer.h'],'    static int forward(',
        'bool debug         = false);',lambda t:once(t,'bool debug         = false);',
        'bool debug         = false,\n        const int64_t* topology_offsets=nullptr,int* topology_records=nullptr,int* topology_status=nullptr,int64_t topology_capacity=0);'))
    def impl(t):
        t=once(t,'bool debug) {','bool debug'+pointer+') {')
        return once(t,'                   require_depth),','                   require_depth,topology_offsets,topology_records,topology_status,topology_capacity),')
    data['cuda_rasterizer/rasterizer_impl.cu']=section(data['cuda_rasterizer/rasterizer_impl.cu'],
        'int CudaRasterizer::Rasterizer::forward(', '    return num_rendered;\n}',impl)
    f=data['cuda_rasterizer/render_forward.cu']
    f=once(f,'float* __restrict__ normal_length) {','float* __restrict__ normal_length'+pointer+') {')
    f=once(f,'bool require_depth) {','bool require_depth'+pointer+') {')
    f=once(f,'out_normal, out_mdepth, normal_length)','out_normal, out_mdepth, normal_length, topology_offsets, topology_records, topology_status, topology_capacity)')
    old='if constexpr(FIRST){topology_depth_accepted++;if(!ball)topology_nonball++;}'
    new='''if constexpr(FIRST){
                        if(inside&&topology_offsets){
                            int64_t slot=topology_offsets[pix_id]+int64_t(topology_depth_accepted);
                            if(slot>=0&&slot<topology_capacity&&slot<topology_offsets[pix_id+1]){
                                topology_records[2*slot]=int(point_list[range.x+i*BLOCK_SIZE+j]);
                                topology_records[2*slot+1]=int(contributor)-1;
                            }else topology_status[pix_id]|=1;
                        }
                        topology_depth_accepted++;if(!ball)topology_nonball++;
                    }'''
    f=once(f,old,new)
    f=once(f,'out_mdepth[5 * H * W + pix_id] = T<=MIN_TRANSMITTANCE?1.f:0.f;',
        'out_mdepth[5 * H * W + pix_id] = T<=MIN_TRANSMITTANCE?1.f:0.f;\n'
        '            if(topology_offsets&&int64_t(topology_depth_accepted)!=topology_offsets[pix_id+1]-topology_offsets[pix_id])topology_status[pix_id]|=2;')
    data['cuda_rasterizer/render_forward.cu']=f
    data['cuda_rasterizer/render_forward.h']=once(data['cuda_rasterizer/render_forward.h'],'    bool require_depth);',
        '    bool require_depth,\n    const int64_t* topology_offsets=nullptr,int* topology_records=nullptr,int* topology_status=nullptr,int64_t topology_capacity=0);')
    data['csr_check.cu']=r'''
#include <cuda_runtime.h>
#include <cstdint>
__global__ void csr_check_kernel(const int* records,const int64_t* offsets,const int* ranges,const int* points,const int* contrib,int* result,int W,int H,int P,int64_t L,int Npoints){
 int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=W*H)return;
 int64_t start=offsets[pixel],end=offsets[pixel+1];int flags=0;
 if(start<0||end<start||end>L){result[pixel]=1;return;}
 int x=pixel%W,y=pixel/W,tile=(y/16)*((W+15)/16)+x/16;
 int base=ranges[2*tile],limit=ranges[2*tile+1],previous=-1;
 if(base<0||limit<base||limit>Npoints){result[pixel]=1;return;}
 for(int64_t i=start;i<end;i++){
   int id=records[2*i],position=records[2*i+1];
   bool valid=position>=0&&position<contrib[pixel]&&position<limit-base;
   if(!valid)flags|=2;
   else if(id!=points[base+position])flags|=4;
   if(position<=previous)flags|=8;
   if(id<0||id>=P)flags|=16;
   previous=position;
 }
 result[pixel]=flags;
}
void csr_check_launch(const int* a,const int64_t* b,const int* c,const int* d,const int* e,int* f,int W,int H,int P,int64_t L,int Npoints,cudaStream_t stream){
 csr_check_kernel<<<(W*H+127)/128,128,0,stream>>>(a,b,c,d,e,f,W,H,P,L,Npoints);
}
'''
    data['csr_check.cpp']=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void csr_check_launch(const int*,const int64_t*,const int*,const int*,const int*,int*,int,int,int,int64_t,int,cudaStream_t);
torch::Tensor CSRValidateCUDA(torch::Tensor records,torch::Tensor offsets,torch::Tensor ranges,torch::Tensor points,torch::Tensor contrib,int W,int H,int P){
 for(auto t:{records,offsets,ranges,points,contrib})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==records.device(),"Same-device contiguous required");
 TORCH_CHECK(records.scalar_type()==torch::kInt32&&records.dim()==2&&records.size(1)==2&&offsets.scalar_type()==torch::kInt64&&offsets.dim()==1&&offsets.numel()==int64_t(W)*H+1,"CSR shapes/dtypes invalid");
 TORCH_CHECK(ranges.scalar_type()==torch::kInt32&&ranges.dim()==2&&ranges.size(1)==2&&ranges.size(0)==((W+15)/16)*((H+15)/16),"Tile ranges invalid");
 TORCH_CHECK(points.scalar_type()==torch::kInt32&&points.dim()==1&&contrib.scalar_type()==torch::kInt32&&contrib.numel()==int64_t(W)*H&&W>0&&H>0&&P>0,"Points/image invalid");
 c10::cuda::CUDAGuard guard(records.device());auto out=torch::zeros({H,W},records.options());
 csr_check_launch(records.data_ptr<int>(),offsets.data_ptr<int64_t>(),ranges.data_ptr<int>(),points.data_ptr<int>(),contrib.data_ptr<int>(),out.data_ptr<int>(),W,H,P,records.size(0),points.numel(),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
'''
    return data

def load(out,old):
    import torch
    from research.exact_perspective_backend import ROOT,RASTER,FLAGS
    from torch.utils.cpp_extension import load as compile_extension
    original=old/'instrumented_generated/src'
    base={p.relative_to(original).as_posix():p.read_text(encoding='utf-8') for p in original.rglob('*') if p.is_file() and p.suffix in ('.cu','.cpp','.h')}
    data=payloads(base);h=hashlib.sha256((torch.__version__+str(FLAGS)).encode())
    for n,t in sorted(data.items()):h.update(n.encode());h.update(t.encode())
    vendor={p.relative_to(RASTER).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((RASTER/'third_party/glm/glm').rglob('*')) if p.is_file()}
    for n,d in sorted(vendor.items()):h.update(n.encode());h.update(d.encode())
    build=out/'csr_generated';tree=build/'src';tree.mkdir(parents=True)
    for n,t in data.items():
        p=tree/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,encoding='utf-8')
    files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/render_forward.cu','cuda_rasterizer/render_backward.cu',
           'cuda_rasterizer/sample_forward.cu','cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp','csr_check.cpp','csr_check.cu']
    module=compile_extension(name='csr_capture_'+h.hexdigest()[:12],sources=[str(tree/n) for n in files],
        extra_include_paths=[str(tree),str(tree/'cuda_rasterizer'),str(RASTER),str(RASTER/'cuda_rasterizer'),str(RASTER/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=FLAGS,build_directory=str(build),verbose=True)
    p=Path(module.__file__)
    return module,dict(binary_relative_path=p.relative_to(ROOT).as_posix(),binary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        generated_sha256={n:hashlib.sha256((tree/n).read_bytes()).hexdigest() for n in data},vendor_headers_sha256=vendor,flags=FLAGS,
        scope='Dynamic actual first-iteration ID/position capture, no truncation; not certified depth/backward.')
