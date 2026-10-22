"""Generate an auxiliary-arithmetic-only forward control and compile under a unique name."""
import argparse,difflib,json,os,sys,time,traceback
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
from tgs_aux_ablation_common_v21 import OUT,V19,V20,cases
SOURCE=ROOT/'sources/tgs_native_diagnostic_v1';NATIVE=SOURCE/'native'

def prepare():
    OUT.mkdir(exist_ok=True);src=OUT/'source';src.mkdir();original=(NATIVE/'cuda_rasterizer/forward.cu').read_text()
    marker='float blending_weight = alpha * T;';assert original.count(marker)==1;start=original.index(marker);end=original.index('if (T > 0.5)',start)
    generated=original[:start]+'// v21: auxiliary arithmetic ablated; RGB/alpha/ordinal recurrence unchanged.\n\t\t\t '+original[end:]
    for old,new in [('median_depth = collected_depths[j];','// v21: no auxiliary median-depth value'),
        ('for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * blending_weight;','// v21: no auxiliary normal accumulation')]:
        assert generated.count(old)==1;generated=generated.replace(old,new)
    assert 'blending_weight' not in generated
    target=src/'forward.cu';target.write_text(generated)
    (src/'forward.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),generated.splitlines(True),fromfile='pinned/forward.cu',tofile='aux_control/forward.cu')))
    manifest=json.loads((SOURCE/'manifest.json').read_text());files=[SOURCE/e['path'] for e in manifest['files']]
    for e in manifest['files']:assert sha(SOURCE/e['path'])==e['sha256']
    files.extend([Path(__file__),ROOT/'tgs_aux_ablation_common_v21.py',ROOT/'TGS_AUX_ABLATION_PROTOCOL_V21.md',target,src/'forward.diff',SOURCE/'manifest.json',
        V19/'plan.json',V19/'results.json',V19/'evidence_manifest.json',V20/'results.json',V20/'evidence_manifest.json',
        ROOT/'run_tgs_local_split_v19.py',ROOT/'tgs_local_split_v19.cu',ROOT/'tgs_split_color_common_v14.py',ROOT/'trace_tgs_window_scene_v15.py',
        ROOT/'run_tgs_real_events.py',ROOT/'tgs_cross_scene_common_v10.py',ROOT/'tgs_sh_isolation_common_v12.py',
        ROOT/'experiments/tgs_sh_isolation_v12/input_plan.json',ROOT/'experiments/tgs_sh_isolation_v12/identity_audit.json',
        ROOT/'experiments/tgs_coupled_v18/fresh/results.json'])
    prior=json.loads((V19/'plan.json').read_text());files.extend(ROOT/n for n in prior['inputs'] if n.endswith('point_cloud_state_dict.pt'))
    files.extend(ROOT/f'sources/tgs_cross_scene_v10/{s}/cameras.json' for s in ['flowers','room'])
    for scene in ['flowers','room']:
        files.extend(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_{i}/colors.npy' for i in [0,1])
        files.extend([V20/scene/'selection.json',V20/scene/'eligible.npz'])
    p=prior['native'];assert sha(Path(p['path']))==p['sha256']
    write_new(OUT/'plan.json',dict(inputs={str(f.relative_to(ROOT)):sha(f) for f in files},native=p,cases=cases(),full_forward_cap=204,training=False,
        unused_aux_outputs_zero=True,backward_supported=False,allocation_and_API_layout_preserved=True,build_flags=['-O3','--use_fast_math','sm_120']))
    print('AUX_CONTROL_PLAN_FROZEN',flush=True)

def build():
    p=json.loads((OUT/'plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    folder=OUT/'build';folder.mkdir()
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    sources=[NATIVE/'cuda_rasterizer/rasterizer_impl.cu',OUT/'source/forward.cu',NATIVE/'cuda_rasterizer/backward.cu',NATIVE/'cuda_rasterizer/utils.cu',NATIVE/'rasterize_points.cu',NATIVE/'ext.cpp']
    t0=time.perf_counter();print('BUILD_STARTED',flush=True)
    m=load(name='tgs_aux_arithmetic_control_v21',sources=[str(x) for x in sources],extra_include_paths=[str(NATIVE/'third_party/glm'),str(NATIVE/'cuda_rasterizer')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(folder),verbose=False)
    write_new(OUT/'build_receipt.json',dict(path=m.__file__,sha256=sha(Path(m.__file__)),torch=torch.__version__,cuda=torch.version.cuda,
        build_seconds=time.perf_counter()-t0,generated_forward_sha256=sha(OUT/'source/forward.cu'),backward_supported=False,flags=p['build_flags'],training=False))
    print('BUILD_COMPLETE',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','build']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
