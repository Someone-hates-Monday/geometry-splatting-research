"""Explicit phase arithmetic; compile instrumented and non-counter twins."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
from tgs_native_diagnostic_backend import ROOT
from tgs_backward_fixed_backend import once
OUT=ROOT/'experiments/tgs_phase_cost_v5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def generate():
    dst=OUT/'generated/native';assert not dst.exists();shutil.copytree(ROOT/'experiments/tgs_compositing_guard_v4/generated/native',dst)
    cu=dst/'cuda_rasterizer';text=(cu/'stable_window.cuh').read_text()
    replacements=[
        ('const float x=pixel.x-anchor.x,y=pixel.y-anchor.y;','const float x=__fsub_rn(pixel.x,anchor.x),y=__fsub_rn(pixel.y,anchor.y);'),
        ('float d=n[k].x*x+n[k].y*y+b[k];','float d=__fadd_rn(__fadd_rn(__fmul_rn(n[k].x,x),__fmul_rn(n[k].y,y)),b[k]);'),
        ('z.u=-z.phi*rinv;','z.u=__fmul_rn(-z.phi,rinv);'),
        ('const float raw=opacity*z.Cx;','const float raw=__fmul_rn(opacity,z.Cx);'),
        ('fabsf(z.alpha-1.0f/255.0f)','fabsf(__fsub_rn(z.alpha,1.0f/255.0f))'),
        ('fabsf(raw-.99f)','fabsf(__fsub_rn(raw,.99f))'),
        ('const double dx=(double)pixel.x-anchor.x,dy=(double)pixel.y-anchor.y;','const double dx=__dadd_rn((double)pixel.x,-(double)anchor.x),dy=__dadd_rn((double)pixel.y,-(double)anchor.y);'),
        ('const double u=fmax(0.0,-best*ir),Cx=pow(u,(double)sigma),raw=(double)opacity*Cx;','const double u=fmax(0.0,__dmul_rn(-best,ir)),Cx=pow(u,(double)sigma),raw=__dmul_rn((double)opacity,Cx);'),
        ('z.phi=(float)best;','z.phi=__double2float_rn(best);'),
        ('z.u=(float)u;z.Cx=(float)Cx;z.capped=raw>=.99;z.alpha=(float)fmin(.99,raw);','z.u=__double2float_rn(u);z.Cx=__double2float_rn(Cx);z.capped=raw>=.99;z.alpha=__double2float_rn(fmin(.99,raw));')]
    for a,b in replacements:text=once(text,a,b)
    text+='\n__device__ __forceinline__ float phase_depth_m(float depth,float near_,float far_) {return __fmul_rn(__fdividef(far_,__fsub_rn(far_,near_)),__fsub_rn(1.0f,__fdividef(near_,depth)));}\n'
    (cu/'stable_window.cuh').write_text(text)
    text=(cu/'forward.cu').read_text()
    start=text.index('             unsigned* counter=PREFLIGHT?');end=text.index('             if(window.outside)',start)
    text=text[:start]+'#if WINDOW_COUNTERS\n'+text[start:end]+'#endif\n'+text[end:]
    replacements=[
        ('const float test=T*(1-alpha);','const float test=__fmul_rn(T,__fsub_rn(1.0f,alpha));'),
        ('float test_T = T * (1 - alpha);','float test_T = __fmul_rn(T,__fsub_rn(1.0f,alpha));'),
        ('float blending_weight = alpha * T;','float blending_weight = __fmul_rn(alpha,T);'),
        ('float A = 1-T;','float A = __fsub_rn(1.0f,T);'),
        ('float m = far_n / (far_n - near_n) * (1 - near_n / collected_depths[j]);','float m = phase_depth_m(collected_depths[j],near_n,far_n);'),
        ('distortion += (m * m * A + M2 - 2 * m * M1) * blending_weight;','distortion = __fmaf_rn(__fsub_rn(__fmaf_rn(__fmul_rn(m,m),A,M2),__fmul_rn(__fmul_rn(2.0f,m),M1)),blending_weight,distortion);'),
        ('D  += collected_depths[j] * blending_weight;','D = __fmaf_rn(collected_depths[j],blending_weight,D);'),
        ('M1 += m * blending_weight;','M1 = __fmaf_rn(m,blending_weight,M1);'),
        ('M2 += m * m * blending_weight;','M2 = __fmaf_rn(__fmul_rn(m,m),blending_weight,M2);'),
        ('N[ch] += normal[ch] * blending_weight;','N[ch] = __fmaf_rn(normal[ch],blending_weight,N[ch]);'),
        ('C[ch] += features[j_id * CHANNELS + ch] * alpha * T;','C[ch] = __fmaf_rn(__fmul_rn(features[j_id * CHANNELS + ch],alpha),T,C[ch]);'),
        ('C[ch] + T * bg_color[ch]','__fmaf_rn(T,bg_color[ch],C[ch])')]
    for a,b in replacements:text=once(text,a,b)
    (cu/'forward.cu').write_text(text)
    text=(cu/'backward.cu').read_text();text=once(text,'const float m_d = far_n / (far_n - near_n) * (1 - near_n / collected_depths[j]);','const float m_d = phase_depth_m(collected_depths[j],near_n,far_n);');(cu/'backward.cu').write_text(text)
    return dst

def main():
    OUT.mkdir(exist_ok=True)
    for e in json.loads((ROOT/'experiments/tgs_compositing_guard_v4/evidence_manifest.json').read_text())['files']:assert sha(ROOT/e['path'])==e['sha256']
    native=generate();sources=[native/'cuda_rasterizer/rasterizer_impl.cu',native/'cuda_rasterizer/forward.cu',native/'cuda_rasterizer/backward.cu',native/'cuda_rasterizer/utils.cu',native/'rasterize_points.cu',native/'ext.cpp']
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2');os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    for variant,mode,counters in [('guard_diag',2,1),('oracle_diag',3,1),('guard',2,0),('oracle',3,0)]:
        build=OUT/variant/'build';build.mkdir(parents=True,exist_ok=True);started=time.time();flags=['-O3','--use_fast_math',f'-DSTABLE_MODE={mode}',f'-DWINDOW_COUNTERS={counters}']
        mod=load(name='tgs_diag_phase_v5_'+variant,sources=[str(p) for p in sources],extra_include_paths=[str(ROOT/'sources/tgs_native_diagnostic_v1/native/third_party/glm'),str(native/'cuda_rasterizer')],extra_cflags=['-O3'],extra_cuda_cflags=flags,build_directory=str(build),verbose=True)
        receipt=dict(module_file=mod.__file__,module_sha256=sha(Path(mod.__file__)),generator_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'TGS_PHASE_COST_PROTOCOL_V5.md'),variant=variant,mode=mode,instrumented=bool(counters),counter_buffers_reserved_and_zeroed=True,extra_cuda_cflags=flags,build_seconds=time.time()-started,torch=torch.__version__,cuda=torch.version.cuda,sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in native.rglob('*') if p.is_file()],training=False)
        path=OUT/variant/'build_receipt.json';assert not path.exists();path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
