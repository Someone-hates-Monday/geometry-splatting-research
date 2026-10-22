from pathlib import Path
import hashlib
from torch.utils.cpp_extension import load
def build(directory):
    source=Path(__file__).with_suffix('.cu');digest=hashlib.sha256(source.read_bytes()).hexdigest()[:12];directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    return load(name='history_gap_warp_'+digest,sources=[str(source)],build_directory=str(directory),extra_cuda_cflags=['-O3','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true','--allow-unsupported-compiler'],extra_cflags=['-O3'],verbose=False)
