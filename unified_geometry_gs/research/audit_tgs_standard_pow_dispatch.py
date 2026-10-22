"""Independent finite-grid dispatch audit, explicitly retain FTZ discrepancies."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_standard_pow_dispatch_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'independent_audit.json';assert not target.exists();plan=json.loads((OUT/'plan.json').read_text());result=json.loads((OUT/'results.json').read_text());assert result['plan_sha256']==sha(OUT/'plan.json') and plan['script_sha256']==sha(ROOT/'probe_tgs_standard_pow_dispatch.py')
    for p,h in plan['source_hashes'].items():assert sha(ROOT/p)==h
    assert sha(OUT/'input.npz')==plan['input_sha256'] and sha(OUT/'build/probe.cu')==plan['source_sha256'];assert sha(Path(result['module_file']))==result['module_sha256'] and sha(OUT/'sass.txt')==result['SASS_sha256']
    with np.load(OUT/'input.npz') as z:x=z['power'];ref=z['standard_reference']
    with np.load(OUT/'outputs.npz') as z:y=z['values']
    assert len(x)==4007 and y.shape==(4007,3);assert np.array_equal(y[:,0],y[:,2]);normal=(x[:,0]>=2.**-126)&(ref>=2.**-126);actual=np.arange(len(x))>=plan['v6_rows'];assert np.array_equal(y[normal,1],ref[normal]) and np.array_equal(y[actual,1],ref[actual]);assert np.count_nonzero(y[:,0]!=y[:,1])==765
    # Actual build.ninja used the default CUDA symlink because load was imported before CUDA_HOME.
    # Read-only provenance verification proves that default resolves to the intended12.8 toolchain.
    canonical=subprocess.run(['/bin/readlink','-f','/usr/local/cuda'],capture_output=True,text=True,timeout=10);version=subprocess.run(['/usr/local/cuda/bin/nvcc','--version'],capture_output=True,text=True,timeout=10);assert canonical.returncode==version.returncode==0 and canonical.stdout.strip()=='/usr/local/cuda-12.8' and 'V12.8.93' in version.stdout
    (OUT/'compiler_provenance.json').write_text(json.dumps(dict(default_path='/usr/local/cuda',canonical=canonical.stdout.strip(),nvcc_version=version.stdout,nvcc_sha256=sha(Path('/usr/local/cuda/bin/nvcc')),import_order_note='cpp_extension imported before CUDA_HOME assignment; actual default path resolved12.8. Set environment before import in future builds.'),indent=2)+'\n')
    metadata=dict(rows=len(x),normal_input_output_rows=int(normal.sum()),all_normal_standard_bitwise=True,trace_rows=int(actual.sum()),all_trace_standard_bitwise=True,fast_and_named_bitwise=True,explicit_fast_differences=765,total_standard_reference_mismatches=int(np.count_nonzero(y[:,1]!=ref)),FTZ_mismatches_retained=True,standard_power_interval_proved=False,renderer_fix_proved=False,training=False,timing=False,script_sha256=sha(Path(__file__)),results_sha256=sha(OUT/'results.json'),compiler_provenance_sha256=sha(OUT/'compiler_provenance.json'))
    target.write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)
if __name__=='__main__':main()
