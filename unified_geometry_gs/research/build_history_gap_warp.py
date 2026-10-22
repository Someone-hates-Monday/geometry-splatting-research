from pathlib import Path
R=Path(__file__).resolve().parent
original=(R/'history_gap_gpu.cu').read_text(encoding='utf-8');start=original.index('__global__ void gap_kernel(');end=original.index('std::vector<torch::Tensor> solve(',start)
result=original[:start]+(R/'history_gap_warp_body.inc').read_text(encoding='utf-8')+'\n'+original[end:]
result=result.replace('c10::cuda::CUDAGuard guard(p.device());int M=sections.numel()-1;','TORCH_CHECK(p.numel()<=2147483647 && sections.numel()>=2,"index or empty batch"); c10::cuda::CUDAGuard guard(p.device());int M=sections.numel()-1;')
target=R/'history_gap_warp.cu';assert not target.exists();target.write_text(result,encoding='utf-8');print('Generated separate warp source; serial source unchanged.')
