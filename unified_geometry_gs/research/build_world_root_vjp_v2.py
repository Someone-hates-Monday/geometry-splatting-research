"""Preserve v1 evidence; make baseline compensation independent of grad flags."""
from pathlib import Path
import hashlib
R=Path(__file__).resolve().parent;p=R/'world_root_vjp.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='06db519f25ebdc053f6e19ba0befcdea127260e9255f3a28ae222d67e80e4f46';s=p.read_text(encoding='utf-8')
def once(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
once('def torch_fields(theta,raw,source,meta,arm):','def torch_fields(theta,raw,source,meta,arm,_baseline=False):')
once('    if theta.requires_grad:\n        with torch.no_grad():gamma0=torch_fields(base,raw,source,meta,arm)[4]\n    else:gamma0=gamma.detach()', '    if not _baseline:\n        with torch.no_grad():gamma0=torch_fields(base,raw,source,meta,arm,_baseline=True)[4]\n    else:gamma0=gamma.detach()')
once('residual_scaled=float(value))','residual_scaled=float(value.detach()))')
target=R/'world_root_vjp_v2.py';assert not target.exists();target.write_text(s,encoding='utf-8');print(hashlib.sha256(target.read_bytes()).hexdigest())
