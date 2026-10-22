"""Compile four frozen v8 variants; do not mutate original/sealed sources."""
import json,os,sys,time,traceback
from pathlib import Path
from tgs_compensated_common import ROOT,OUT,sha
def main():
    plan=json.loads((OUT/'input_plan.json').read_text());assert sha(ROOT/'TGS_COMPENSATED_RENDERER_PROTOCOL_V8.md')==plan['protocol_sha256'];assert sha(ROOT/'tgs_compensated_window_v8.cuh')==plan['helper_sha256']
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h
    for p,h in plan['test_scripts'].items():assert sha(ROOT/p)==h
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2');os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    native=OUT/'generated/native';sources=[native/'cuda_rasterizer/rasterizer_impl.cu',native/'cuda_rasterizer/forward.cu',native/'cuda_rasterizer/backward.cu',native/'cuda_rasterizer/utils.cu',native/'rasterize_points.cu',native/'ext.cpp']
    for v,mode,count in [('guard_diag',2,1),('oracle_diag',3,1),('guard',2,0),('oracle',3,0)]:
        d=OUT/v/'build';assert not d.exists();d.mkdir(parents=True);started=time.time();flags=['-O3','--use_fast_math',f'-DSTABLE_MODE={mode}',f'-DWINDOW_COUNTERS={count}']
        m=load(name='tgs_diag_compensated_v8_'+v,sources=list(map(str,sources)),extra_include_paths=[str(ROOT/'sources/tgs_native_diagnostic_v1/native/third_party/glm'),str(native/'cuda_rasterizer')],extra_cflags=['-O3'],extra_cuda_cflags=flags,build_directory=str(d),verbose=True)
        receipt=dict(module_file=m.__file__,module_sha256=sha(Path(m.__file__)),generator_sha256=sha(Path(__file__)),protocol_sha256=plan['protocol_sha256'],variant=v,mode=mode,instrumented=bool(count),counter_buffers_reserved_and_zeroed=True,stable_error_stride=48,extra_cuda_cflags=flags,build_seconds=time.time()-started,torch=torch.__version__,cuda=torch.version.cuda,sources=[dict(path=p,sha256=h) for p,h in plan['sources'].items()],training=False)
        (OUT/v/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(variant=v,module_sha256=receipt['module_sha256'],build_seconds=receipt['build_seconds'])),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'build_failure.txt').write_text(traceback.format_exc());raise
