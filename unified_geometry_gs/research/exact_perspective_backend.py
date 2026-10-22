"""Opt-in complete-image depth functional; inference-only, legacy affine selection."""
from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
import hashlib,json,threading
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
RASTER=ROOT/'submodules/diff-gaussian-rasterization'
VARIANTS=('affine','exact_packet','exact_factor')
FLAGS=['-O3','--use_fast_math','--extended-lambda','--expt-relaxed-constexpr','--allow-unsupported-compiler']
_LOCK=threading.RLock()

def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError(f'Expected one audited source anchor: {old!r}, got {text.count(old)}')
    return text.replace(old,new)

def payloads(variant):
    if variant not in VARIANTS:raise ValueError('Unknown depth-functional variant.')
    data={}
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        for p in RASTER.glob(pattern):data[p.relative_to(RASTER).as_posix()]=p.read_text(encoding='utf-8')
    # Do not export unmodified backward/sample entry points from modified binaries.
    ext=data['ext.cpp'];lines=ext.splitlines(keepends=True)
    data['ext.cpp']=''.join(line for line in lines if 'm.def(' not in line or '"rasterize_gaussians"' in line or '"mark_visible"' in line)
    if variant=='affine':return data
    forward=data['cuda_rasterizer/render_forward.cu']
    helper=(ROOT/'research/exact_perspective_ray.inc').read_text(encoding='utf-8')
    forward=replace_once(forward,'namespace cg = cooperative_groups;','namespace cg = cooperative_groups;\n'+helper)
    forward=replace_once(forward,'[[maybe_unused]] float tslope      = 0;','[[maybe_unused]] float tslope      = 0;\n    bool exact_invalid = false;')
    start=forward.index('        auto iter = [&] __device__(auto first_const) {')
    end=forward.index('        iter(std::true_type{});',start)
    section=forward[start:end]
    old='''                    float4 ray_plane   = collected_ray_planes[j];
                    const float t_peak = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                    const float rsigma = ray_plane.w;
                    const bool ball    = rsigma > 0;'''
    call=('exact_packet_ray(xy,ray_plane,con_o,pixnf,W,H,focal_x,focal_y)' if variant=='exact_packet' else
          'exact_factor_ray(exact_factors+12*point_list[range.x+i*BLOCK_SIZE+j],con_o,pixnf)')
    new='''                    float4 ray_plane = collected_ray_planes[j];
                    ExactPerspectiveRay recovered = '''+call+''';
                    exact_invalid = exact_invalid || !recovered.valid;
                    if (!recovered.valid) continue;
                    const float t_peak = recovered.peak;
                    const float rsigma = recovered.reciprocal_sigma;
                    alpha = recovered.alpha;
                    const bool ball = true;'''
    section=replace_once(section,old,new)
    section=replace_once(section,'in_range = (T_p[0] >= 0.5f) && (T_p[SPLIT] <= 0.5f) && in_range;',
        'in_range = !exact_invalid && (T_p[0] >= 0.5f) && (T_p[SPLIT] <= 0.5f) && in_range;')
    forward=forward[:start]+section+forward[end:]
    forward=replace_once(forward,'out_mdepth[H * W + pix_id]     = tslope;','out_mdepth[H * W + pix_id]     = exact_invalid ? nanf("") : tslope;')
    if variant=='exact_factor':
        # Supplementary pointer affects only this variant's private buffer layout.
        forward=replace_once(forward,'const float3* __restrict__ normals,','const float3* __restrict__ normals,\n        const float* __restrict__ exact_factors,')
        forward=replace_once(forward,'    const float3* normals,\n    const float focal_x,','    const float3* normals,\n    const float* exact_factors,\n    const float focal_x,')
        forward=replace_once(forward,'ray_planes, normals, focal_x, focal_y,','ray_planes, normals, exact_factors, focal_x, focal_y,')
        forward+='\n'+(ROOT/'research/exact_perspective_factor.inc').read_text(encoding='utf-8')
        header=data['cuda_rasterizer/render_forward.h']
        header=replace_once(header,'    const float3* normals,\n    const float focal_x,','    const float3* normals,\n    const float* exact_factors,\n    const float focal_x,')
        header=replace_once(header,'} // namespace FORWARD','void build_exact_factors(int,const float*,const float*,const float*,const float*,float,float*);\n} // namespace FORWARD')
        data['cuda_rasterizer/render_forward.h']=header
        h=data['cuda_rasterizer/rasterizer_impl.h']
        h=replace_once(h,'struct GeometryState {','struct GeometryState {\n    float* exact_factors;')
        data['cuda_rasterizer/rasterizer_impl.h']=h
        impl=data['cuda_rasterizer/rasterizer_impl.cu']
        impl=replace_once(impl,'    obtain(chunk, geom.depths, P, 128);\n    obtain(chunk, geom.ray_planes, P, 128);',
            '    obtain(chunk, geom.depths, P, 128);\n    obtain(chunk, geom.ray_planes, P, 128);\n    obtain(chunk, geom.exact_factors, 12 * P, 128);')
        marker='    // Compute prefix sum over full list of touched tile counts by Gaussians'
        fs=impl.index('int CudaRasterizer::Rasterizer::forward(');fe=impl.index('    return num_rendered;',fs)
        fsection=replace_once(impl[fs:fe],marker,'    CHECK_CUDA(FORWARD::build_exact_factors(P, means3D, scales, rotations, viewmatrix, scale_modifier, geomState.exact_factors), debug);\n'+marker)
        impl=impl[:fs]+fsection+impl[fe:]
        s=impl.index('    CHECK_CUDA(FORWARD::render(');e=impl.index('    return num_rendered;',s)
        callsection=impl[s:e]
        callsection=replace_once(callsection,'                   geomState.normals,','                   geomState.normals,\n                   geomState.exact_factors,')
        impl=impl[:s]+callsection+impl[e:];data['cuda_rasterizer/rasterizer_impl.cu']=impl
    data['cuda_rasterizer/render_forward.cu']=forward
    return data

class InferenceBackend:
    def __init__(self,module,variant,record):
        self._module=module;self.variant=variant;self.record=record;self.__file__=module.__file__
    def rasterize_gaussians(self,*args):
        if torch.is_grad_enabled():raise RuntimeError('Exact perspective depth backend is inference-only: use torch.no_grad().')
        if len(args)!=25:raise ValueError('Audited rasterizer forward ABI changed.')
        if float(args[18])!=0.:raise ValueError('Exact perspective integration currently requires kernel_size=0.')
        if args[6].numel()!=0:raise ValueError('Precomputed covariance path is outside this integration audit.')
        if args[4].numel()==0 or args[5].numel()==0:raise ValueError('Filtered scales and activated rotations are required.')
        return self._module.rasterize_gaussians(*args)
    def mark_visible(self,*args):return self._module.mark_visible(*args)
    def __getattr__(self,name):
        if name.startswith('_'):raise AttributeError(name)
        raise RuntimeError(f'{name} is unavailable: backward, sample, integrate and SDF have no exact depth integration.')

@lru_cache(maxsize=3)
def load_backend(variant='exact_packet'):
    data=payloads(variant);h=hashlib.sha256((torch.__version__+json.dumps(FLAGS)).encode())
    for name,value in sorted(data.items()):h.update(name.encode());h.update(value.encode())
    glm=[p for p in (RASTER/'third_party/glm/glm').rglob('*') if p.is_file()]
    for p in sorted(glm):h.update(p.relative_to(RASTER).as_posix().encode());h.update(p.read_bytes())
    signature=h.hexdigest()[:12];build=ROOT/'research/build'/f'exact_perspective_{variant}_{signature}';tree=build/'src'
    for name,value in data.items():
        p=tree/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_text(value,encoding='utf-8')
        elif p.read_text(encoding='utf-8')!=value:raise ValueError('Existing generated source differs from content identity.')
    from torch.utils.cpp_extension import load
    files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/render_forward.cu','cuda_rasterizer/render_backward.cu',
        'cuda_rasterizer/sample_forward.cu','cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp']
    module=load(name=f'uggs_exact_perspective_{variant}_{signature}',sources=[str(tree/n) for n in files],
        extra_include_paths=[str(tree),str(tree/'cuda_rasterizer'),str(RASTER),str(RASTER/'cuda_rasterizer'),str(RASTER/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=FLAGS,build_directory=str(build),verbose=False)
    record={'variant':variant,'signature':signature,'binary_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        'generated_sha256':{n:hashlib.sha256((tree/n).read_bytes()).hexdigest() for n in data},
        'scope':'image median-depth functional under legacy affine selection; inference only',
        'coefficient_evaluation':'FP64 scalar arithmetic then FP32 original solver' if variant!='affine' else 'unchanged production FP32',
        'additional_geometry_bytes_per_primitive':48 if variant=='exact_factor' else 0,'nvcc_flags':FLAGS,
        'original_source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in RASTER.rglob('*') if p.is_file() and p.suffix in ('.cu','.cpp','.h','.hpp','.inl')}}
    return InferenceBackend(module,variant,record)

@contextmanager
def activate(backend):
    if not isinstance(backend,InferenceBackend):raise TypeError('Use this module load_backend to enforce inference scope.')
    if torch.is_grad_enabled():raise RuntimeError('Enter torch.no_grad() before activating this inference-only backend.')
    import diff_gaussian_rasterization as rasterizer
    with _LOCK:
        original=rasterizer._C;rasterizer._C=backend
        try:yield
        finally:rasterizer._C=original

def metadata(backend):return backend.record

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=VARIANTS,default='exact_packet')
    b=load_backend(p.parse_args().variant);print(json.dumps(b.record,indent=2),flush=True)
