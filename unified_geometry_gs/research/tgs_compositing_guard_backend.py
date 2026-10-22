"""Two-pass compositing classification prototype; both-pass costs explicit."""
import hashlib,json,os,re,shutil,sys,time
from pathlib import Path
from tgs_native_diagnostic_backend import ROOT
from tgs_backward_fixed_backend import once
OUT=ROOT/'experiments/tgs_compositing_guard_v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def final_arg(text,name,extra):
    result,n=re.subn(r'('+name+r')(\s*\))',lambda m:m[1]+', '+extra+m[2],text);assert n>0;return result

def generate():
    src=ROOT/'experiments/tgs_adaptive_window_v3/generated/native';dst=OUT/'generated/native';assert not dst.exists();shutil.copytree(src,dst)
    cu=dst/'cuda_rasterizer'
    text=(cu/'stable_window.cuh').read_text()
    text=once(text,'float phi,u,Cx,alpha;','float phi,u,Cx,alpha,alpha_error;')
    text=once(text,'float sigma,float opacity,float rinv)','float sigma,float opacity,float rinv,bool force_oracle=false)')
    text=once(text,'bool fallback=STABLE_MODE==3;','bool fallback=STABLE_MODE==3 || force_oracle;')
    text=once(text,'if(STABLE_MODE!=3) {','if(STABLE_MODE!=3 && !force_oracle) {')
    text=once(text,'if(!fallback) return z;','if(!fallback) {z.alpha_error=allowance;return z;}')
    text=once(text,'const double d=nx*dx+ny*dy+off;','const double d=__dadd_rn(__dadd_rn(__dmul_rn(nx,dx),__dmul_rn(ny,dy)),off);')
    (cu/'stable_window.cuh').write_text(text)
    text=(cu/'rasterizer_impl.h').read_text();text=once(text,'double* stable_coeff;','unsigned* preflight_count;\n        double* stable_coeff;');text=once(text,'struct ImageState\n\t {','struct ImageState\n\t {\n        uint32_t* precision_replay;');(cu/'rasterizer_impl.h').write_text(text)
    text=(cu/'rasterizer_impl.cu').read_text()
    text=once(text,'obtain(chunk, geom.stable_coeff, P*10, 128);','obtain(chunk, geom.preflight_count, P*3, 128);\n     obtain(chunk, geom.stable_coeff, P*10, 128);')
    text=once(text,'obtain(chunk, img.ranges, N, 128);','obtain(chunk, img.ranges, N, 128);\n     obtain(chunk, img.precision_replay, N, 128);')
    text=once(text,'prefiltered, geomState.stable_coeff, geomState.stable_error, geomState.stable_count','prefiltered, geomState.stable_coeff, geomState.stable_error, geomState.stable_count, geomState.preflight_count, imgState.precision_replay')
    text=once(text,'max_blending, geomState.stable_coeff, geomState.stable_error, geomState.stable_count','max_blending, geomState.stable_coeff, geomState.stable_error, geomState.stable_count, geomState.preflight_count, imgState.precision_replay')
    assert text.count('dL_dsigma_factor, geomState.stable_coeff, geomState.stable_error, geomState.stable_count')==2
    text=text.replace('dL_dsigma_factor, geomState.stable_coeff, geomState.stable_error, geomState.stable_count','dL_dsigma_factor, geomState.stable_coeff, geomState.stable_error, geomState.stable_count, imgState.precision_replay')
    (cu/'rasterizer_impl.cu').write_text(text)
    for name in ['forward.h','forward.cu']:
        text=(cu/name).read_text();text=final_arg(text,r'unsigned\* stable_count','unsigned* preflight_count, uint32_t* replay')
        if name.endswith('.cu'):
            text=once(text,'template <uint32_t CHANNELS>','template <uint32_t CHANNELS, bool PREFLIGHT>')
            text=once(text,'stable_count[3*idx]=stable_count[3*idx+1]=stable_count[3*idx+2]=0;','stable_count[3*idx]=stable_count[3*idx+1]=stable_count[3*idx+2]=0;\n     preflight_count[3*idx]=preflight_count[3*idx+1]=preflight_count[3*idx+2]=0;')
            text=once(text,'float T = 1.0f;','float T = 1.0f;\n     float T_lower=1.0f,T_upper=1.0f;uint32_t precision_flags=0;')
            text=once(text,'collected_xy[j],pixf,sigma_pre,con_o.w,-phi_center_min.x);','collected_xy[j],pixf,sigma_pre,con_o.w,-phi_center_min.x,(STABLE_MODE==2 && !PREFLIGHT && replay[pix_id]!=0));')
            text=once(text,'atomicAdd(stable_count+3*j_id,1u);','unsigned* counter=PREFLIGHT?preflight_count:stable_count;\n             atomicAdd(counter+3*j_id,1u);')
            text=text.replace('atomicAdd(stable_count+3*j_id+','atomicAdd(counter+3*j_id+')
            marker='\t\t\t float test_T = T * (1 - alpha);'
            classify=r'''
             if(PREFLIGHT) {
                 const float a_lower=fmaxf(0.0f,__fsub_rd(alpha,window.alpha_error));
                 const float a_upper=fminf(.99f,__fadd_ru(alpha,window.alpha_error));
                 const float next_lower=__fmul_rd(T_lower,__fsub_rd(1.0f,a_upper));
                 const float next_upper=__fmul_ru(T_upper,__fsub_ru(1.0f,a_lower));
                 if(T_lower<=.5f && T_upper>.5f) precision_flags|=1u;
                 if(next_lower<.0001f && next_upper>=.0001f) precision_flags|=2u;
                 if(precision_flags) {done=true;continue;}
                 const float test=T*(1-alpha);
                 if(test<.0001f) {done=true;continue;}
                 T=test;T_lower=next_lower;T_upper=next_upper;
                 continue;
             }
'''
            text=once(text,marker,classify+'\n'+marker)
            text=once(text,'\t // All threads that treat valid pixel write out their final','\t if(PREFLIGHT) {if(inside)replay[pix_id]=precision_flags;return;}\n\t // All threads that treat valid pixel write out their final')
            # Pass extra arguments at both wrapper launch sites, then duplicate only render.
            text=once(text,'max_blending, stable_coeff, stable_error, stable_count\n','max_blending, stable_coeff, stable_error, stable_count, preflight_count, replay\n')
            text=once(text,'prefiltered, stable_coeff, stable_error, stable_count\n','prefiltered, stable_coeff, stable_error, stable_count, preflight_count, replay\n')
            start=text.index('\t renderCUDA<NUM_CHANNELS>');end=text.index('\n\t\t );',start)+len('\n\t\t );')
            call=text[start:end];pre=call.replace('renderCUDA<NUM_CHANNELS>','renderCUDA<NUM_CHANNELS,true>');fin=call.replace('renderCUDA<NUM_CHANNELS>','renderCUDA<NUM_CHANNELS,false>')
            text=text[:start]+'if(STABLE_MODE==2) {\n'+pre+'\n} else {cudaMemset(replay,0,sizeof(uint32_t)*W*H);}\n'+fin+text[end:]
        (cu/name).write_text(text)
    for name in ['backward.h','backward.cu']:
        text=(cu/name).read_text();text=final_arg(text,r'unsigned\* stable_count','const uint32_t* replay')
        if name.endswith('.cu'):
            assert text.count('dL_dsigma_factor, stable_coeff, stable_error, stable_count')==2
            text=text.replace('dL_dsigma_factor, stable_coeff, stable_error, stable_count','dL_dsigma_factor, stable_coeff, stable_error, stable_count, replay')
            text=once(text,'collected_xy[j],pixf,sigma_pre,con_o.w,-phi_center_min.x);','collected_xy[j],pixf,sigma_pre,con_o.w,-phi_center_min.x,(STABLE_MODE==2 && replay[pix_id]!=0));')
        (cu/name).write_text(text)
    return dst

def main():
    for e in json.loads((ROOT/'experiments/tgs_adaptive_window_v3/evidence_manifest.json').read_text())['files']:assert sha(ROOT/e['path'])==e['sha256']
    native=generate();sources=[native/'cuda_rasterizer/rasterizer_impl.cu',native/'cuda_rasterizer/forward.cu',native/'cuda_rasterizer/backward.cu',native/'cuda_rasterizer/utils.cu',native/'rasterize_points.cu',native/'ext.cpp']
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2');os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    for mode,v in [(2,'adaptive'),(3,'oracle64')]:
        build=OUT/v/'build';build.mkdir(parents=True,exist_ok=True);started=time.time()
        flags=['-O3','--use_fast_math',f'-DSTABLE_MODE={mode}']
        mod=load(name='tgs_diag_guard_v4_'+v,sources=[str(p) for p in sources],extra_include_paths=[str(ROOT/'sources/tgs_native_diagnostic_v1/native/third_party/glm'),str(native/'cuda_rasterizer')],extra_cflags=['-O3'],extra_cuda_cflags=flags,build_directory=str(build),verbose=True)
        receipt=dict(module_file=mod.__file__,module_sha256=sha(Path(mod.__file__)),generator_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'TGS_COMPOSITING_GUARD_PROTOCOL_V4.md'),variant=v,mode=mode,extra_cuda_cflags=flags,cache_payload_bytes_per_triangle=144,extra_image_bytes_per_pixel=4,seconds=time.time()-started,torch=torch.__version__,cuda=torch.version.cuda,sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in native.rglob('*') if p.is_file()],training=False,instrumented=True)
        (OUT/v/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
