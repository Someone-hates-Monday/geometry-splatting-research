"""Exact directed reciprocal table for the existing strict-series scalar divisions."""
from pathlib import Path
from fractions import Fraction
import hashlib,math
import torch
from research.guarded_interval_cuda import FLAGS

def proof_and_header(base):
    rows=[];literal=['{0.,0.}']
    for k in range(1,52):
        exact=Fraction(1,k);near=float(exact);f=Fraction(near)
        lo=math.nextafter(near,-math.inf) if f>exact else near
        hi=math.nextafter(near,math.inf) if f<exact else near
        assert Fraction(lo)<=exact<=Fraction(hi)
        assert lo==hi or math.nextafter(lo,math.inf)==hi
        rows.append(dict(k=k,lo_hex=lo.hex(),hi_hex=hi.hex(),numerator=1,denominator=k,adjacent_or_exact=True))
        literal.append('{'+lo.hex()+','+hi.hex()+'}')
    declaration='''#ifdef __CUDACC__
__device__ __constant__ I integer_reciprocal[52] = {TABLE};
#else
static constexpr I integer_reciprocal[52] = {TABLE};
#endif
GI_INLINE I div_integer(I a,int k){return mul(a,integer_reciprocal[k]);}
'''.replace('TABLE',','.join(literal))
    marker='GI_INLINE I div(I a,I b){';assert base.count(marker)==1;text=base.replace(marker,declaration+'\n'+marker)
    substitutions=[('div(term,pt(2*j+1))','div_integer(term,2*j+1)',1),('div(term,pt(k))','div_integer(term,k)',1),
                   ('div(mul(term,r),pt(j))','div_integer(mul(term,r),j)',1),('div(mul(term,x),pt(j))','div_integer(mul(term,x),j)',1),('div(mul(rem,pt(radius)),pt(j))','div_integer(mul(rem,pt(radius)),j)',1),
                   ('div(mul(rem,pt(r)),pt(j))','div_integer(mul(rem,pt(r)),j)',1)]
    for a,b,n in substitutions:assert text.count(a)==n;text=text.replace(a,b)
    return rows,text

def load(out,old,variant):
    build=out/('variant_generated' if variant else 'control_generated');build.mkdir()
    names=['root.cu','binding.cpp','guarded_interval.cuh','guarded_preprocess.cuh','guarded_root.cuh']
    sources={n:(old/'generated'/n).read_text(encoding='utf-8') for n in names}
    proof,optimized=proof_and_header(sources['guarded_interval.cuh'])
    if variant:sources['guarded_interval.cuh']=optimized
    h=hashlib.sha256(torch.__version__.encode())
    for n,t in sorted(sources.items()):h.update(n.encode());h.update(t.encode());(build/n).write_text(t,encoding='utf-8')
    from torch.utils.cpp_extension import load
    module=load(name=('reciprocal_variant_' if variant else 'reciprocal_control_')+h.hexdigest()[:12],sources=[str(build/'binding.cpp'),str(build/'root.cu')],extra_cflags=['-O3'],extra_cuda_cflags=FLAGS,build_directory=str(build),verbose=True)
    return module,proof
