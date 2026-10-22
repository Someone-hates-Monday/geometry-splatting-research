from pathlib import Path
R=Path(__file__).resolve().parent
def replace(s,a,b):
    assert s.count(a)==1,a;return s.replace(a,b)
main=(R/'benchmark_history_gap_gpu.py').read_text(encoding='utf-8')
main=replace(main,'from research.history_gap_gpu import Engine,build,metadata_array','from research.history_gap_gpu import build,metadata_array\nfrom research.history_gap_gpu_v2 import Engine')
main=replace(main,"R/'history_gap_gpu.py',R/'history_gap_gpu.cu'","R/'history_gap_gpu.py',R/'history_gap_gpu_v2.py',R/'HISTORY_GAP_GPU_V2_AMENDMENT.md',R/'history_gap_gpu.cu'")
target=R/'benchmark_history_gap_gpu_v2.py';assert not target.exists();target.write_text(main,encoding='utf-8')
confirm=(R/'confirm_history_gap_gpu.py').read_text(encoding='utf-8')
confirm=replace(confirm,'from research.history_gap_gpu import Engine,metadata_array','from research.history_gap_gpu import metadata_array\nfrom research.history_gap_gpu_v2 import Engine')
confirm=replace(confirm,"GPU=R/'experiments/history_gap_gpu_v1';OUT=R/'experiments/history_gap_gpu_confirm_v2'","GPU=R/'experiments/history_gap_gpu_v2';OUT=R/'experiments/history_gap_gpu_confirm_v4'")
confirm=replace(confirm,"R/'history_gap_gpu.py',R/'history_gap_gpu.cu'","R/'history_gap_gpu.py',R/'history_gap_gpu_v2.py',R/'HISTORY_GAP_GPU_V2_AMENDMENT.md',R/'history_gap_gpu.cu'")
confirm=replace(confirm,'for case in range(32):','for case in range(64):\n            if case==32:rng=np.random.default_rng(13019)')
confirm=replace(confirm,'[case//4]','[(case%32)//4]');confirm=replace(confirm,'[case%3]','[(case%32)%3]')
confirm=replace(confirm,'dict(case=case,nodes=n,','dict(case=case,new_confirmation=case>=32,nodes=n,')
confirm=replace(confirm,'model {case+1}/32','model {case+1}/64');confirm=replace(confirm,"new_cases=32,","new_cases=64,")
confirm=replace(confirm,"print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)","print(json.dumps({k:v for k,v in read(OUT/'summary.json').items() if k!='rows'}),flush=True)")
target=R/'confirm_history_gap_gpu_v2.py';assert not target.exists();target.write_text(confirm,encoding='utf-8')
print('Generated V2 real benchmark and original32 plus fresh32 confirmation; all gates unchanged.')
