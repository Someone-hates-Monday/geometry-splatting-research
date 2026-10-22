"""Build exact pinned TGS native sources under a unique local module name."""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'sources/tgs_native_diagnostic_v1'
OUT = ROOT/'experiments/tgs_subdivision_v1'


def load_backend(shfix=False):
    assert json.loads((OUT/'math_results.json').read_text())['passed']
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((SOURCE/entry['path']).read_bytes()).hexdigest() == entry['sha256']
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
    os.environ['CC'] = '/usr/bin/gcc-11'
    os.environ['CXX'] = '/usr/bin/g++-11'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
    os.environ['MAX_JOBS'] = '2'
    os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    native = SOURCE/'native'
    build = OUT/('sh_direction_v1/fixed_build' if shfix else 'native_build')
    build.mkdir(parents=True,exist_ok=True)
    paths = ['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/forward.cu',
             'cuda_rasterizer/backward.cu','cuda_rasterizer/utils.cu',
             'rasterize_points.cu','ext.cpp']
    sources=[native/p for p in paths]
    changed=[]
    if shfix:
        generated=OUT/'sh_direction_v1/fixed_src'
        generated.mkdir(parents=True,exist_ok=True)
        old='(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)'
        new='glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)'
        for j in [1,2]:
            original=sources[j].read_text()
            assert original.count(old)==1
            fixed=original.replace(old,new)
            target=generated/sources[j].name
            if target.exists():
                assert target.read_text()==fixed
            else:
                target.write_text(fixed)
            changed.append({'original':str(sources[j]),'generated':str(target),
                            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                            'old':old,'new':new})
            sources[j]=target
    started = time.time()
    mod = load(name='tgs_diag_6d61f4c_shfix_v1' if shfix else 'tgs_diag_6d61f4c_v1',sources=[str(p) for p in sources],
               extra_include_paths=[str(native/'third_party/glm'),str(native/'cuda_rasterizer')],
               extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],
               build_directory=str(build),verbose=True)
    info = {'native_commit':manifest['native_commit'],'glm_commit':manifest['glm_commit'],
            'source_manifest_sha256':hashlib.sha256((SOURCE/'manifest.json').read_bytes()).hexdigest(),
            'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
            'python':platform.python_version(),'module_file':mod.__file__,
            'module_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
            'build_or_load_seconds':time.time()-started,'source_edits':changed if shfix else False,
            'environment_installations':False,'forward_and_backward_compiled':True,
            'backward_exercised':False,'flags':['-O3','--use_fast_math','sm_120'],
            'extra_include_paths':[str(native/'third_party/glm'),str(native/'cuda_rasterizer')],
            'build_ninja_sha256':hashlib.sha256((build/'build.ninja').read_bytes()).hexdigest(),
            'main_split_source_commit':'685ff270a8273afc3a2b094ae3148dd0b98e31e9'}
    receipt=OUT/('sh_direction_v1/fixed_build_receipt.json' if shfix else 'build_receipt.json')
    if not receipt.exists():
        receipt.write_text(json.dumps(info,indent=2)+'\n')
    versioned=build/('receipt_'+info['module_sha256'][:12]+'.json')
    if not versioned.exists():
        versioned.write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)
    return mod


if __name__ == '__main__':
    load_backend()
