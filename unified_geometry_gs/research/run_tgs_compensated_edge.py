"""Bounded standalone native build/execution, no training and no timing experiment."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compensated_edge_v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(cmd,log,timeout):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
    log.write_text(p.stdout);assert p.returncode==0,(cmd,p.returncode,p.stdout[-2000:]);return p.stdout
def main():
    assert not (OUT/'execution_plan.json').exists();plan=json.loads((OUT/'input_plan.json').read_text())
    for p,h in plan['files'].items():assert sha(OUT/p)==h
    assert sha(ROOT/'TGS_COMPENSATED_EDGE_PROTOCOL_V6.md')==plan['protocol_sha256']
    assert sha(ROOT/'tgs_compensated_edge_probe.cu')==plan['source_sha256']
    versions={'strict':['--ftz=false','--prec-div=true','--prec-sqrt=true'],'fast_ftz':['--use_fast_math','--ftz=true']}
    frozen=dict(input_plan_sha256=sha(OUT/'input_plan.json'),runner_sha256=sha(Path(__file__)),variants=versions,training=False,timing=False)
    (OUT/'execution_plan.json').write_text(json.dumps(frozen,indent=2)+'\n')
    command(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],OUT/'gpu.txt',20)
    command(['/usr/local/cuda-12.8/bin/nvcc','--version'],OUT/'nvcc.txt',20)
    receipts=[]
    for v,flags in versions.items():
        d=OUT/v;d.mkdir();exe=d/'probe';cmd=['/usr/local/cuda-12.8/bin/nvcc','-O3','-std=c++17','-arch=sm_120','-ccbin=/usr/bin/g++-11','--ptxas-options=-v',*flags,str(ROOT/'tgs_compensated_edge_probe.cu'),'-o',str(exe)]
        command(cmd,d/'compile.log',120);command([str(exe),str(OUT/'input.bin'),str(d/'edge.bin'),str(d/'power.bin')],d/'run.log',60)
        command(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(exe)],d/'sass.txt',60)
        receipt=dict(variant=v,command=cmd,binary_sha256=sha(exe),edge_sha256=sha(d/'edge.bin'),power_sha256=sha(d/'power.bin'),sass_sha256=sha(d/'sass.txt'))
        (d/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(receipt);print(json.dumps(receipt),flush=True)
    (OUT/'native_results.json').write_text(json.dumps(dict(variants=receipts,execution_plan_sha256=sha(OUT/'execution_plan.json'),training=False,timing=False),indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'native_failure.txt').write_text(traceback.format_exc());raise
