"""Differentiate the inverse covariance factor directly in an isolated backend.

For the positive-scale branch only, replace inverse-to-covariance-to-factor
backpropagation by direct differentiation of (diag(1/s) R)^T diag(1/s) R.
All forward code, five subdivisions, masks, losses and reduction precision stay
unchanged. This is a development hypothesis, not a quality improvement claim.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import torch
from research.median_cuda_backend import activate,load_backend as original_backend
ROOT=Path(__file__).resolve().parents[1]


def replace_one(text,old,new):
    if text.count(old)!=1:raise ValueError(f'Stable inverse anchor changed: {old[:70]}')
    return text.replace(old,new)


def generate(backward):
    start=backward.index('__device__ __forceinline__ void computeCov3D(')
    end=backward.index('// Backward version of INVERSE 2D covariance',start)
    function=backward[start:end]
    inverse=function.replace('void computeCov3D(', 'void computeInvCov3D(')
    for field in ('x','y','z'):
        axis={'x':0,'y':1,'z':2}[field]
        inverse=replace_one(inverse,f'S[{axis}][{axis}]     = scale.{field};',f'S[{axis}][{axis}]     = 1.f / scale.{field};')
        inverse=replace_one(inverse,f'dL_dscale->{field}         = glm::dot(Rt[{axis}], dL_dMt[{axis}]);',
            f'dL_dscale->{field}         += -glm::dot(Rt[{axis}], dL_dMt[{axis}]) / (scale.{field} * scale.{field});')
        inverse=replace_one(inverse,f'dL_dMt[{axis}] *= scale.{field};',f'dL_dMt[{axis}] /= scale.{field};')
    inverse=replace_one(inverse,'    *dL_drot        = float4{dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w};',
        '    dL_drot->x += dL_dq.x; dL_drot->y += dL_dq.y; dL_drot->z += dL_dq.z; dL_drot->w += dL_dq.w;')
    backward=backward[:end]+inverse+backward[end:]
    backward=replace_one(backward,'    glm::mat3 dL_dVrk;',
        '    glm::mat3 dL_dVrk;\n    glm::mat3 dL_dInvVrk(0.f);')
    old='''            dL_dVrk = -glm::outerProduct(Vrk_inv * W_uvh,
                                         Vrk_inv * (W * glm::transpose(nJ_inv) * dL_dplane_append + W_uvh * dL_dvb_xvb));
            dL_dVrk = dL_dVrk / vb;'''
    new='''            if (scales) {
                dL_dInvVrk = glm::outerProduct(W_uvh,
                    W * glm::transpose(nJ_inv) * dL_dplane_append + W_uvh * dL_dvb_xvb) / vb;
                dL_dVrk = glm::mat3(0.f);
            } else {
                dL_dVrk = -glm::outerProduct(Vrk_inv * W_uvh,
                    Vrk_inv * (W * glm::transpose(nJ_inv) * dL_dplane_append + W_uvh * dL_dvb_xvb)) / vb;
            }'''
    backward=replace_one(backward,old,new)
    old='''    if (scales)
        computeCov3D(idx, scale_local, mod, R, rot, dL_dcov_local, dL_dr, min_id, dL_dscales, dL_drots);
    else {'''
    new='''    if (scales) {
        computeCov3D(idx, scale_local, mod, R, rot, dL_dcov_local, dL_dr, min_id, dL_dscales, dL_drots);
        if (well_conditioned) {
            const float inverse_grad[6] = {dL_dInvVrk[0][0],
                dL_dInvVrk[0][1] + dL_dInvVrk[1][0], dL_dInvVrk[0][2] + dL_dInvVrk[2][0],
                dL_dInvVrk[1][1], dL_dInvVrk[1][2] + dL_dInvVrk[2][1], dL_dInvVrk[2][2]};
            computeInvCov3D(idx, scale_local, mod, R, rot, inverse_grad, glm::vec3(0.f), min_id, dL_dscales, dL_drots);
        }
    } else {'''
    return replace_one(backward,old,new)


@lru_cache(maxsize=3)
def load_backend(variant='stable2'):
    if variant=='control':return original_backend('control')
    if variant not in ('stable2','stable1'):raise ValueError('Unknown stable inverse variant.')
    source=ROOT/'submodules/diff-gaussian-rasterization'
    payloads={}
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        for p in source.glob(pattern):payloads[p.relative_to(source).as_posix()]=p.read_text()
    payloads['cuda_rasterizer/render_backward.cu']=generate(payloads['cuda_rasterizer/render_backward.cu'])
    if variant=='stable1':
        payloads['cuda_rasterizer/config.h']=replace_one(payloads['cuda_rasterizer/config.h'],
            'constexpr int SAMPLE_BATCH_SIZE   = 2;','constexpr int SAMPLE_BATCH_SIZE   = 1;')
    h=hashlib.sha256(torch.__version__.encode())
    for name,value in sorted(payloads.items()):h.update(name.encode());h.update(value.encode())
    signature=h.hexdigest()[:12];build=ROOT/'research/build'/f'inverse_{variant}_{signature}'
    tree=build/'src'
    for name,value in payloads.items():
        p=tree/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or p.read_text()!=value:p.write_text(value)
    from torch.utils.cpp_extension import load
    files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/render_forward.cu',
        'cuda_rasterizer/render_backward.cu','cuda_rasterizer/sample_forward.cu',
        'cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp']
    return load(name=f'uggs_inverse_{variant}_{signature}',sources=[str(tree/p) for p in files],
        extra_include_paths=[str(tree),str(tree/'cuda_rasterizer'),str(source),str(source/'cuda_rasterizer'),str(source/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda',
            '--expt-relaxed-constexpr','--allow-unsupported-compiler'],build_directory=str(build),verbose=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=('stable2','stable1'),default='stable2')
    print(load_backend(p.parse_args().variant).__file__,flush=True)
