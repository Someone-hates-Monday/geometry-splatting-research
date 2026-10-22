"""Isolated native tree with honest additional precision caches and counters."""
import hashlib,json,os,re,shutil,sys,time
from pathlib import Path
from tgs_native_diagnostic_backend import ROOT,SOURCE
from tgs_backward_fixed_backend import once
OUT=ROOT/'experiments/tgs_adaptive_window_v3'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def append_signature(text,final,extra):
    pattern=rf'({final})(\s*\))'
    value,count=re.subn(pattern,lambda m:m[1]+', '+extra+m[2],text)
    assert count>0,(final,count)
    return value

EXTRA='const double* stable_coeff, const float* stable_error, unsigned* stable_count'
PRE_EXTRA='double* stable_coeff, float* stable_error, unsigned* stable_count'

def generated_sources():
    original=SOURCE/'native';old=ROOT/'experiments/tgs_stable_window_v2/direct/generated'
    base=OUT/'generated';native=base/'native';cu=native/'cuda_rasterizer'
    assert not base.exists(),'Never overwrite a generated version'
    shutil.copytree(original/'cuda_rasterizer',cu)
    for name in ['rasterize_points.cu','rasterize_points.h','ext.cpp']:shutil.copy2(original/name,native/name)
    shutil.copy2(ROOT/'tgs_adaptive_window.cuh',cu/'stable_window.cuh')
    h=(cu/'rasterizer_impl.h').read_text();h=once(h,'uint2* rect_min;','double* stable_coeff;\n        float* stable_error;\n        unsigned* stable_count;\n        uint2* rect_min;');(cu/'rasterizer_impl.h').write_text(h)
    impl=(cu/'rasterizer_impl.cu').read_text()
    impl=once(impl,'obtain(chunk, geom.rect_min, P, 128);','obtain(chunk, geom.stable_coeff, P*10, 128);\n     obtain(chunk, geom.stable_error, P*10, 128);\n     obtain(chunk, geom.stable_count, P*3, 128);\n     obtain(chunk, geom.rect_min, P, 128);')
    args='geomState.stable_coeff, geomState.stable_error, geomState.stable_count'
    impl=once(impl,'prefiltered\n\t ), debug)','prefiltered, '+args+'\n\t ), debug)')
    impl=once(impl,'max_blending), debug)','max_blending, '+args+'), debug)')
    assert impl.count('dL_dsigma_factor), debug)')==2
    impl=impl.replace('dL_dsigma_factor), debug)','dL_dsigma_factor, '+args+'), debug)')
    (cu/'rasterizer_impl.cu').write_text(impl)
    h=(cu/'forward.h').read_text();h=append_signature(h,'bool prefiltered',PRE_EXTRA);h=append_signature(h,'float\\* max_blending',EXTRA);(cu/'forward.h').write_text(h)
    h=(cu/'backward.h').read_text();h=append_signature(h,'float\\* dL_dsigma_factor',EXTRA);(cu/'backward.h').write_text(h)
    text=(old/'forward.cu').read_text();text='#include "stable_window.cuh"\n'+text
    text=append_signature(text,'bool prefiltered',PRE_EXTRA)
    text=append_signature(text,r'float\* (?:__restrict__ )?max_blending',EXTRA)
    text=once(text,'\t\t max_blending\n','\t\t max_blending, stable_coeff, stable_error, stable_count\n')
    text=once(text,'\t\t prefiltered\n','\t\t prefiltered, stable_coeff, stable_error, stable_count\n')
    text=once(text,'density_factor[idx] = 0.0f;','density_factor[idx] = 0.0f;\n     stable_count[3*idx]=stable_count[3*idx+1]=stable_count[3*idx+2]=0;')
    marker='    // Exact real-arithmetic level set:'
    block=r'''
    // Original-projection double coefficients, independent of float vertex cache.
    double2 v64[3];double perimeter64=0.0;
    for(int i=0;i<3;++i) v64[i]=make_double2(q64[i].x-(double)center_triangle_2D.x,q64[i].y-(double)center_triangle_2D.y);
    const double area64=(v64[1].x-v64[0].x)*(v64[2].y-v64[0].y)-(v64[1].y-v64[0].y)*(v64[2].x-v64[0].x);
    const double sign64=area64>0?1.0:-1.0;
    for(int i=0;i<3;++i) {
        const double dx=v64[(i+1)%3].x-v64[i].x,dy=v64[(i+1)%3].y-v64[i].y,len=hypot(dx,dy);
        perimeter64+=len;
        const double nx=sign64*dy/len,ny=-sign64*dx/len,off=-(nx*v64[i].x+ny*v64[i].y);
        stable_coeff[10*idx+2*i]=nx;stable_coeff[10*idx+2*i+1]=ny;stable_coeff[10*idx+6+i]=off;
        stable_error[10*idx+2*i]=nextafterf((float)fabs(nx-(double)normals[cumsum_for_triangle+i].x),CUDART_INF_F);
        stable_error[10*idx+2*i+1]=nextafterf((float)fabs(ny-(double)normals[cumsum_for_triangle+i].y),CUDART_INF_F);
        stable_error[10*idx+6+i]=nextafterf((float)fabs(off-(double)offsets[cumsum_for_triangle+i]),CUDART_INF_F);
    }
    const double invradius64=perimeter64/fabs(area64);
    stable_coeff[10*idx+9]=invradius64;
    stable_error[10*idx+9]=nextafterf((float)fabs(invradius64-(double)(1.0f/radius)),CUDART_INF_F);
'''
    text=once(text,marker,block+'\n'+marker)
    begin=text.index('\t\t\t for (int k = 0; k < 3; k++) {',text.index('// Iterate over current batch'))
    end=text.index('\t\t\t if (alpha <',begin)
    text=text[:begin]+r'''
             const StableWindow window=stable_window(collected_normals+base,collected_offsets+base,
                 stable_error+10*j_id,stable_coeff+10*j_id,collected_xy[j],pixf,sigma_pre,con_o.w,-phi_center_min.x);
             atomicAdd(stable_count+3*j_id,1u);
             if(window.high) atomicAdd(stable_count+3*j_id+1,1u);
             if(window.uncertain) atomicAdd(stable_count+3*j_id+2,1u);
             if(window.outside) continue;
             float alpha=window.alpha;
''' +text[end:]
    (cu/'forward.cu').write_text(text)
    text=(old/'backward.cu').read_text();text='#include "stable_window.cuh"\n'+text
    text=append_signature(text,r'float\* (?:__restrict__ )?dL_dsigma_factor',EXTRA)
    # Kernel call sites, not declarations.
    text,count=re.subn(r'dL_dsigma_factor(\s*\)\s*;)',r'dL_dsigma_factor, stable_coeff, stable_error, stable_count\1',text)
    assert count==2,count
    begin=text.index('\t\t\t for (int k = 0; k < 3; k++) {',text.index('// Iterate over triangles'))
    end=text.index('\t\t\t if (alpha <',begin)
    text=text[:begin]+r'''
             const int window_id=collected_id[j];
             const StableWindow window=stable_window(collected_normals+base,collected_offsets+base,
                 stable_error+10*window_id,stable_coeff+10*window_id,collected_xy[j],pixf,sigma_pre,con_o.w,-phi_center_min.x);
             if(window.outside) continue;
             max_val=window.phi;
             const float phi_x=window.phi,phi_final=window.u,Cx=window.Cx,alpha=window.alpha;
''' +text[end:]
    text=once(text,'if (con_o.w * Cx >= 0.99f) dL_dalpha = 0.0f;','if (window.capped) dL_dalpha = 0.0f;')
    text=once(text,'if (!selected_edge && distances[k] == max_val)','if (!selected_edge && k == window.active)')
    (cu/'backward.cu').write_text(text)
    return native

def main():
    plan=json.loads((OUT/'input_plan.json').read_text());assert plan['protocol_sha256']==sha(ROOT/'TGS_ADAPTIVE_WINDOW_PROTOCOL_V3.md')
    for name in ['experiments/tgs_stable_window_v2/evidence_manifest.json','sources/tgs_native_diagnostic_v1/manifest.json']:
        manifest=json.loads((ROOT/name).read_text())
        for e in manifest['files']:
            prefix=ROOT if name.startswith('experiments') else SOURCE
            assert sha(prefix/e['path'])==e['sha256']
    native=generated_sources()
    sources=[native/'cuda_rasterizer/rasterizer_impl.cu',native/'cuda_rasterizer/forward.cu',native/'cuda_rasterizer/backward.cu',native/'cuda_rasterizer/utils.cu',native/'rasterize_points.cu',native/'ext.cpp']
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    for mode,variant in [(1,'arithmetic64'),(2,'adaptive'),(3,'oracle64')]:
        build=OUT/variant/'build';build.mkdir(parents=True,exist_ok=True);started=time.time()
        mod=load(name='tgs_diag_adaptive_v3_'+variant,sources=[str(p) for p in sources],extra_include_paths=[str(SOURCE/'native/third_party/glm'),str(native/'cuda_rasterizer')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math',f'-DSTABLE_MODE={mode}'],build_directory=str(build),verbose=True)
        r=dict(module_file=mod.__file__,module_sha256=sha(Path(mod.__file__)),generator_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'tgs_adaptive_window.cuh'),protocol_sha256=plan['protocol_sha256'],mode=mode,variant=variant,sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in native.rglob('*') if p.is_file()],extra_cuda_cflags=['-O3','--use_fast_math',f'-DSTABLE_MODE={mode}'],extra_cache_bytes_per_triangle=132,seconds=time.time()-started,torch=torch.__version__,cuda=torch.version.cuda,training=False,instrumented=True)
        path=OUT/variant/'build_receipt.json';assert not path.exists();path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
