"""Float64 reference for bounded, fixed-order TGS RGB gradient diagnosis.

SH polynomial convention follows the archived author's utils/sh_utils.py
(PlenOctree, BSD-3-Clause, copyright 2021 PlenOctree Authors). Only mathematical
basis formulas/constants are expressed here; no training/model code is imported.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/tgs_mixed_backward_v1'
MODULES = {
    'control': ('experiments/tgs_subdivision_v1/sh_direction_v1/fixed_build/tgs_diag_6d61f4c_shfix_v1.so',
                '4fa625a41b94545010a579e9c98eb68fb557023c2429be890eaf20d750b00ef1'),
    'fixed': ('experiments/tgs_backward_v1/geometry_fixed/build/tgs_diag_6d61f4c_geometry_fixed_v1.so',
              '955dd08a6082063253e188cd7504ab78cbbad8b60411c0b5c1e26d56cbf662c2'),
}


def load_module(variant):
    relative, digest = MODULES[variant]
    path = ROOT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    # Read-only import: rebuilding via load_fixed() would rewrite the sealed
    # prior build receipt. Keep all previous evidence byte-identical.
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sh_basis(direction, degree):
    x, y, z = direction.unbind(-1)
    b = [x * 0 + .28209479177387814]
    if degree >= 1:
        b += [-.4886025119029199 * y, .4886025119029199 * z, -.4886025119029199 * x]
    if degree >= 2:
        b += [1.0925484305920792*x*y, -1.0925484305920792*y*z,
              .31539156525252005*(2*z*z-x*x-y*y), -1.0925484305920792*x*z,
              .5462742152960396*(x*x-y*y)]
    if degree >= 3:
        b += [-.5900435899266435*y*(3*x*x-y*y), 2.890611442640554*x*y*z,
              -.4570457994644658*y*(4*z*z-x*x-y*y),
              .3731763325901154*z*(2*z*z-3*x*x-3*y*y),
              -.4570457994644658*x*(4*z*z-x*x-y*y),
              1.445305721320277*z*(x*x-y*y), -.5900435899266435*x*(x*x-3*y*y)]
    return torch.stack(b, dim=-1)


def unpack(theta, f):
    n = int(f['n'])
    p = theta[:9*n].reshape(n, 3, 3)
    sigma, opacity = theta[9*n:10*n], theta[10*n:11*n]
    sh = theta[11*n:].reshape(n, (int(f['degree'])+1)**2, 3) if int(f['degree']) >= 0 else None
    return p, sigma, opacity, sh


def reference(theta, f):
    p, sigma, opacity, sh = unpack(theta, f)
    n, samples = len(p), len(f['pixels'])
    const = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)
    projection, view = const(f['projection']), const(f['view'])
    hom = torch.cat([p, torch.ones((n, 3, 1), dtype=torch.float64)], -1) @ projection.T
    xy = (hom[..., :2] / (hom[..., 3:] + 1e-7) + 1) * const([f['w']/2, f['h']/2]) - .5
    edge = xy.roll(-1, 1) - xy
    length = torch.linalg.vector_norm(edge, dim=-1)
    center = (length.roll(-1, 1)[..., None] * xy).sum(1) / length.sum(1)[:, None]
    normal = torch.stack([edge[..., 1], -edge[..., 0]], -1) / length[..., None]
    offset = -(normal * xy).sum(-1)
    flip = torch.where((normal * center[:, None]).sum(-1) + offset > 0, -1., 1.)
    normal, offset = normal * flip[..., None], offset * flip
    radius = -((normal[:, -1] * center).sum(-1) + offset[:, -1])
    all_phi = torch.einsum('sd,nkd->nsk', const(f['pixels']), normal) + offset[:, None]
    phi, active = all_phi.max(-1)
    inside = phi <= 0
    u = torch.where(inside, -phi / radius[:, None], torch.ones_like(phi))
    raw_alpha = opacity[:, None] * u.pow(sigma[:, None])
    alpha = torch.minimum(raw_alpha, const(.99))
    valid = inside & (alpha >= 1/255) & const(f.get('preprocess_visible', np.ones(n))).bool()[:, None]
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    if sh is None:
        raw_color = const(f['colors'])
    else:
        d = p.mean(1) - const(f['campos'])
        d = d / torch.linalg.vector_norm(d, dim=-1, keepdim=True)
        raw_color = torch.einsum('nb,nbc->nc', sh_basis(d, int(f['degree'])), sh) + .5
    color = raw_color.clamp_min(0)
    depths = torch.cat([p.mean(1), torch.ones((n, 1), dtype=torch.float64)], -1) @ view.T
    order = depths[:, 2].detach().argsort().tolist()
    trans = torch.ones(samples, dtype=torch.float64)
    rgb = torch.zeros((samples, 3), dtype=torch.float64)
    done = torch.zeros(samples, dtype=torch.bool)
    included = []
    terminated = []
    for i in order:
        test_t = trans * (1-alpha[i])
        stop = valid[i] & (test_t < 1e-4) & ~done
        take = valid[i] & ~done & ~stop
        a = torch.where(take, alpha[i], torch.zeros_like(alpha[i]))
        rgb = rgb + (trans*a)[:, None] * color[i]
        trans = trans * (1-a)
        done = done | stop
        included.append(take.tolist())
        terminated.append(stop.tolist())
    rgb = rgb + trans[:, None] * const(f['background'])
    loss = (rgb * const(f['pixel_grad'])).sum()
    sorted_phi = all_phi.sort(dim=-1, descending=True).values
    sorted_z = depths[:, 2].detach().sort().values
    fingerprint = dict(order=order, active=active.tolist(), inside=inside.tolist(), valid=valid.tolist(),
                       capped=(raw_alpha >= .99).tolist(), color_clamped=(raw_color < 0).tolist(),
                       included=included, terminated=terminated)
    meta = dict(fingerprint=fingerprint, raw_alpha=raw_alpha.detach().tolist(),
                alpha=alpha.detach().tolist(), inradius=radius.detach().tolist(),
                projected=xy.detach().tolist(), depth=depths[:, 2].detach().tolist(),
                raw_colors=raw_color.detach().tolist(), transmittance=trans.detach().tolist(),
                edge_gap_min=float((sorted_phi[..., 0] - sorted_phi[..., 1]).min().detach()),
                depth_gap_min=float((sorted_z[1:] - sorted_z[:-1]).min()) if n > 1 else None)
    return loss, rgb, meta


def fixtures():
    w, h, n = 320, 192, 3
    screen = np.array([[40.25, 40.25], [200.25, 40.25], [40.25, 160.25]])
    view = np.eye(4)
    c, s = np.cos(.23), np.sin(.23)
    view[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    view[:3, 3] = [-.31, .17, .26]
    projection = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0]]) @ view
    points = []
    for i, (scale, shift, depths) in enumerate([
        (.92, [0,0], [1.7,1.9,1.8]),
        (1., [3.5,-2.25], [2.4,2.7,2.5]),
        (1.04, [-2.75,3.5], [3.3,3.5,3.4]),
    ]):
        xy = ((screen-screen.mean(0))*scale + screen.mean(0) + shift) * [w/256,h/256]
        z = np.asarray(depths)
        camera = np.c_[((2*xy+1)/[w,h]-1)*z[:,None], z]
        world = (np.c_[camera,np.ones(3)] @ np.linalg.inv(view).T)[:,:3]
        points.append(world[[0,2,1]] if i == 1 else world)
    points = np.asarray(points, dtype=np.float32)
    pixels = np.rint(np.array([[72,65],[135,55],[55,105],[104,91]]) * [w/256,h/256]).astype(int)
    pixel_grad = (np.array([1,.7,-.4,1.1])[:,None]*np.array([1,-.3,.5])).astype(np.float32)
    rng = np.random.default_rng(20261020)
    sh3 = rng.normal(0,.09,(n,16,3)).astype(np.float32)
    sh3[:,0] = [[.3,.1,.2],[.1,.4,.2],[.2,.1,.4]]
    for name, degree in [('three_layer_color',-1), ('three_layer_sh1',1), ('three_layer_sh3',3)]:
        sh = sh3[:,:((degree+1)**2)] if degree >= 0 else np.empty(0,dtype=np.float32)
        theta = np.r_[points.ravel(), [.9,1.3,1.8], [.55,.7,.65], sh.ravel()].astype(np.float32)
        yield dict(name=name, n=n, w=w, h=h, degree=degree, theta=theta, pixels=pixels,
                   pixel_grad=pixel_grad, view=view.astype(np.float32), projection=projection.astype(np.float32),
                   campos=np.linalg.inv(view)[:3,3].astype(np.float32),
                   background=np.array([.08,.13,.21],dtype=np.float32),
                   colors=np.array([[.7,.4,.2],[.2,.7,.3],[.4,.2,.8]],dtype=np.float32),
                   tx=1.,ty=1.)


class Native:
    def __init__(self, mod, f):
        self.mod, self.f = mod, f
        self.t = lambda x, dtype=torch.float32: torch.tensor(x, dtype=dtype, device='cuda').contiguous()
        self.view, self.proj = self.t(f['view'].T.copy()), self.t(f['projection'].T.copy())
        self.bg, self.campos = self.t(f['background']), self.t(f['campos'])
        self.counts = self.t([3]*f['n'],torch.int32)
        self.offsets = self.t(np.arange(f['n'])*3,torch.int32)
        self.zeros, self.empty = self.t(np.zeros(f['n'])), self.t([])
        self.colors = self.t(f['colors']) if f['degree'] < 0 else self.empty
        grad = np.zeros((3,f['h'],f['w']),dtype=np.float32)
        for (x,y), g in zip(f['pixels'], f['pixel_grad']):
            grad[:,y,x] = g
        self.grad = self.t(grad)
        self.grad_others = self.t(np.zeros((7,f['h'],f['w'])))

    def forward(self, theta):
        f, t = self.f, self.t
        p, sigma, opacity, sh = unpack(theta, f)
        p, sigma, opacity = t(p),t(sigma),t(opacity)
        sh = t(sh) if sh is not None else self.empty
        res = self.mod.rasterize_triangles(self.bg,p,sigma,self.counts,self.offsets,
            self.colors,opacity,self.zeros,self.zeros,self.view,self.proj,f['n'],
            float(f['tx']),float(f['ty']),f['h'],f['w'],sh,max(f['degree'],0),self.campos,False,True)
        torch.cuda.synchronize()
        samples = np.array([res[1][:,y,x].cpu().numpy() for x,y in f['pixels']])
        loss = float((samples.astype(float)*f['pixel_grad'].astype(float)).sum())
        return loss,res,p,sigma,sh,samples

    def backward(self, data):
        loss,res,p,sigma,sh,samples = data
        f = self.f
        grads = self.mod.rasterize_triangles_backward(self.bg,p,sigma,self.counts,self.offsets,
            res[3],self.colors,self.view,self.proj,f['n'],float(f['tx']),float(f['ty']),
            self.grad,self.grad_others,sh,max(f['degree'],0),self.campos,
            res[4],res[0],res[5],res[6],True)
        torch.cuda.synchronize()
        arrays = [g.cpu().numpy() for g in grads]
        packed = np.r_[arrays[0].ravel(),arrays[1].ravel(),arrays[3].ravel(),
                       arrays[4].ravel() if f['degree']>=0 else []]
        return packed, arrays
