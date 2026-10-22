"""Build four new modules; old oracle references are imported read-only elsewhere."""
import json,os,sys,time,traceback
from pathlib import Path
from tgs_window_candidates_common import ROOT,OUT,sha
def main():
    plan=json.loads((OUT/'input_plan.json').read_text());assert sha(ROOT/'TGS_WINDOW_CANDIDATES_PROTOCOL_V9.md')==plan['protocol_sha256']
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h
    for p,h in plan['test_scripts'].items():assert sha(ROOT/p)==h
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2');os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    for v,family,count in [('coarse_diag','coarse',1),('standard_diag','standard',1),('coarse','coarse',0),('standard','standard',0)]:
        native=OUT/family/'generated/native';src=[native/'cuda_rasterizer/rasterizer_impl.cu',native/'cuda_rasterizer/forward.cu',native/'cuda_rasterizer/backward.cu',native/'cuda_rasterizer/utils.cu',native/'rasterize_points.cu',native/'ext.cpp'];build=OUT/v/'build';assert not build.exists();build.mkdir(parents=True);started=time.time();flags=['-O3','--use_fast_math','-DSTABLE_MODE=2',f'-DWINDOW_COUNTERS={count}']
        m=load(name='tgs_diag_window_v9_'+v,sources=list(map(str,src)),extra_include_paths=[str(ROOT/'sources/tgs_native_diagnostic_v1/native/third_party/glm'),str(native/'cuda_rasterizer')],extra_cflags=['-O3'],extra_cuda_cflags=flags,build_directory=str(build),verbose=True);receipt=dict(module_file=m.__file__,module_sha256=sha(Path(m.__file__)),generator_sha256=sha(Path(__file__)),protocol_sha256=plan['protocol_sha256'],variant=v,family=family,instrumented=bool(count),stable_error_stride=14 if family=='coarse' else 48,extra_cuda_cflags=flags,build_seconds=time.time()-started,torch=torch.__version__,cuda=torch.version.cuda,sources={p.relative_to(ROOT).as_posix():sha(p) for p in native.rglob('*') if p.is_file()},training=False);(OUT/v/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(variant=v,module_sha256=receipt['module_sha256'],build_seconds=receipt['build_seconds'])),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'build_failure.txt').write_text(traceback.format_exc());raise
