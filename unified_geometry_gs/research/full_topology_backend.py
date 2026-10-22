"""Original full-image affine renderer with actual instruction-context counts."""
from pathlib import Path
import hashlib
import torch
from research.exact_perspective_backend import payloads as original, RASTER, FLAGS

def once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)

def payloads(instrumented):
    data=original('affine')
    if not instrumented:return data
    # Expanded output is private to native diagnostic callers.
    data['rasterize_points.cu']=once(data['rasterize_points.cu'],
        'torch::full({2, H, W}, 0.0, float_opts)', 'torch::full({6, H, W}, 0.0, float_opts)')
    f=data['cuda_rasterizer/render_forward.cu']
    f=once(f,'uint32_t last_contributor          = 0;',
        'uint32_t last_contributor          = 0;\n'
        '    uint32_t topology_rgb_accepted=0, topology_depth_accepted=0, topology_nonball=0;')
    f=once(f,'const float aT = alpha * T;',
        'topology_rgb_accepted++;\n            const float aT = alpha * T;')
    f=once(f,'const bool ball    = rsigma > 0;',
        'const bool ball    = rsigma > 0;\n'
        '                    if constexpr(FIRST){topology_depth_accepted++;if(!ball)topology_nonball++;}')
    f=once(f,'out_mdepth[H * W + pix_id]     = tslope;',
        'out_mdepth[H * W + pix_id]     = tslope;\n'
        '            out_mdepth[2 * H * W + pix_id] = float(topology_rgb_accepted);\n'
        '            out_mdepth[3 * H * W + pix_id] = float(topology_depth_accepted);\n'
        '            out_mdepth[4 * H * W + pix_id] = float(topology_nonball);\n'
        '            out_mdepth[5 * H * W + pix_id] = T<=MIN_TRANSMITTANCE?1.f:0.f;')
    data['cuda_rasterizer/render_forward.cu']=f
    return data

def load(out,instrumented):
    data=payloads(instrumented);name='instrumented' if instrumented else 'control'
    h=hashlib.sha256((torch.__version__+str(FLAGS)).encode())
    for n,t in sorted(data.items()):h.update(n.encode());h.update(t.encode())
    # Include vendor header identities in generated build identity.
    vendor={p.relative_to(RASTER).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((RASTER/'third_party/glm/glm').rglob('*')) if p.is_file()}
    for n,d in sorted(vendor.items()):h.update(n.encode());h.update(d.encode())
    build=out/(name+'_generated');tree=build/'src';tree.mkdir(parents=True)
    for n,t in data.items():
        p=tree/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,encoding='utf-8')
    files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/render_forward.cu','cuda_rasterizer/render_backward.cu',
           'cuda_rasterizer/sample_forward.cu','cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp']
    from torch.utils.cpp_extension import load as compile_extension
    module=compile_extension(name='full_topology_'+name+'_'+h.hexdigest()[:12],sources=[str(tree/n) for n in files],
        extra_include_paths=[str(tree),str(tree/'cuda_rasterizer'),str(RASTER),str(RASTER/'cuda_rasterizer'),str(RASTER/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=FLAGS,build_directory=str(build),verbose=True)
    p=Path(module.__file__)
    return module,dict(binary_relative_path=p.relative_to(Path(__file__).resolve().parents[1]).as_posix(),binary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        generated_sha256={n:hashlib.sha256((tree/n).read_bytes()).hexdigest() for n in data},vendor_headers_sha256=vendor,flags=FLAGS,
        scope='Native full affine forward, counts only in actual acceptance sites; no depth-functional/backward change.')
