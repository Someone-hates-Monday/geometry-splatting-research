"""Bounded mathematical checks; no training or renderer invocation."""
import hashlib
import json
import platform
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import sympy as sp
import scipy
from scipy.optimize import linprog
from verify_tgs_perspective_certificate import systems, verify, split

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'experiments/tgs_subdivision_v1'


def analytic(v, sigma, opacity):
    sides = np.array([np.linalg.norm(v[(j+1)%3]-v[(j+2)%3]) for j in range(3)])
    area = abs(np.linalg.det(np.stack([v[1]-v[0], v[2]-v[0]])))/2
    s = sides@v / sides.sum()
    return 2*opacity*area/((sigma+1)*(sigma+2)), (v.sum(0)+sigma*s)/(sigma+3), s


def integrate(v, sigma, opacity, n):
    _, _, s = analytic(v, sigma, opacity)
    nodes, weights = np.polynomial.legendre.leggauss(n)
    t = ((nodes+1)/2)[:, None, None]
    q = ((nodes+1)/2)[None, :, None]
    w = np.outer(weights/2, weights/2)
    edges = np.roll(v, -1, axis=0)-v
    normals = np.stack([-edges[:,1], edges[:,0]], 1)/np.linalg.norm(edges,axis=1)[:,None]
    # Read the edge-distance window directly, not a radial replacement formula.
    radius = np.min(np.sum((s-v)*normals,axis=1))
    assert radius > 0
    mass, first, radial = 0., np.zeros(2), 0.
    for j in range(3):
        boundary = (1-q)*v[j]+q*v[(j+1)%3]
        pts = s+t*(boundary-s)
        u = np.min(np.sum((pts[:,:,None,:]-v)*normals,axis=-1),axis=-1)/radius
        jac = float(abs(np.linalg.det(np.stack([v[j]-s,v[(j+1)%3]-s]))))*t[:,:,0]
        vals = opacity*np.maximum(u,0)**sigma*jac*w
        mass += vals.sum()
        first += np.sum(vals[:,:,None]*pts,axis=(0,1))
        radial += np.sum(vals*np.linalg.norm(pts-s,axis=-1))
    return mass, first/mass, radial/mass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sigma = sp.Symbol('sigma', positive=True)
    # Layer-cake integrals: sigma*integral t^(sigma-1)*(1-t)^k dt.
    def beta_poly(k):
        return sp.factor(sum((-1)**j*sp.binomial(k,j)*sigma/(sigma+j) for j in range(k+1)))
    z = beta_poly(2)
    assert sp.simplify(z-2/((sigma+1)*(sigma+2))) == 0
    assert sp.simplify(beta_poly(3)/z-3/(sigma+3)) == 0
    r = sp.Rational
    lam = [r(17,48),r(15,48),r(16,48)]
    t = sp.Symbol('t', nonnegative=True)
    w = [(1+2*t)*x-t for x in lam]+[t]
    tstar = sp.solve(w[0]-t,t)[0]
    weights = [x.subs(t,tstar) for x in w]
    assert tstar == r(17,62) and weights == [r(17,62),r(13,62),r(15,62),r(17,62)]
    threshold = 1/(4*tstar)
    assert threshold == r(31,34)
    assert sp.solve((1+2*t)*(r(15,48)-r(16,48)),t) == []  # nonnegative t
    signed_t = sp.solve((1+2*sp.Symbol('q'))*(-r(1,48)),sp.Symbol('q'))[0]
    assert signed_t == -r(1,2)
    v = np.array([[0.,0.],[4.,0.],[0.,3.]])
    numeric = []
    for sig in [.5,1.,1.16,4.,20.]:
        expected_m, expected_mu, _ = analytic(v,sig,.5)
        results = []
        for n in [64,128,256,512]:
            m,mu,rad = integrate(v,sig,.5,n)
            results.append({'nodes_per_axis':n,'mass':m,'centroid':mu.tolist(),
                            'mass_abs_error':abs(m-expected_m),
                            'centroid_max_error':float(np.max(np.abs(mu-expected_mu))),
                            'normalized_mean_radius':rad})
        (OUT/'integration_progress.json').write_text(json.dumps({'sigma':sig,'results':results},indent=2)+'\n')
        print(json.dumps({'sigma':sig,'mass_errors':[a['mass_abs_error'] for a in results]}),flush=True)
        assert results[-1]['mass_abs_error'] < 3e-8
        assert results[-1]['centroid_max_error'] < 3e-8
        numeric.append({'sigma':sig,'expected_mass':expected_m,
                        'expected_centroid':expected_mu.tolist(),'cubature':results})
    child = np.array(split(v))
    cm = [integrate(c,1.,.5,64) for c in child]
    mass = sum(c[0] for c in cm)
    centroid = sum(c[0]*c[1] for c in cm)/mass
    assert abs(mass-1) < 1e-12
    assert np.max(np.abs(centroid-np.array([21/16,1]))) < 1e-12
    candidates = []
    # Bounded candidate search. The theorem predicts sufficiently large D,
    # not every finite D. Preserve non-obstructed candidates too.
    for k in [10,100,1000,10000]:
        mats,rhs,_,_ = systems(k)
        mat = np.array([[x.mid() for x in row] for row in mats[0]+mats[1]])
        b = np.array([x.mid() for x in rhs[0]+rhs[1]])
        # LP only proposes y; the stdlib verifier proves the inequalities.
        opt = linprog(b,A_ub=-mat.T,b_ub=-np.full(4,1e-4),bounds=[(-100,100)]*6,method='highs')
        candidates.append({'k':k,'success':bool(opt.success),'objective':opt.fun})
        if opt.success and opt.fun < -1e-3:
            break
    (OUT/'distance_candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
    assert opt.success and opt.fun < -1e-3
    cert = {'k':k,'distance':f'{k}*sqrt(7)','intrinsics':'I (same for both views)',
            'sigma':1,'parent_opacity':'1/2',
            'farkas_y':[str(F(float(x)).limit_denominator(10**8)) for x in opt.x]}
    evidence = verify(cert)
    (OUT/'perspective_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
    (OUT/'perspective_verified.json').write_text(json.dumps(evidence,indent=2)+'\n')
    report = {'passed':True,'scope':'continuous ideal-window checks; not CUDA or quality evidence',
              'runtime':{'python':platform.python_version(),'numpy':np.__version__,
                         'scipy':scipy.__version__,'sympy':sp.__version__},
              'symbolic':{'mass_factor':str(z),'normalized_radius_factor':str(beta_poly(3)/z),
                          'optimal_weights':[str(x) for x in weights],
                          'opacity_feasibility_threshold':str(threshold),'two_view_signed_t':str(signed_t)},
              'integration':numeric,'split_sigma1':{'mass':mass,'centroid':centroid.tolist(),
                                                    'predicted_displacement':[1/16,0]},
              'distance_candidates':candidates,
              'certificate_sha256':hashlib.sha256((OUT/'perspective_certificate.json').read_bytes()).hexdigest()}
    (OUT/'math_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'threshold':str(threshold),
                      'certificate_At_y':[x['display_midpoint'] for x in evidence['At_y']],
                      'certificate_bt_y':evidence['bt_y']['display_midpoint'],
                      'per_view_opacities':[[x['display_midpoint'] for x in a['opacity_enclosures']]
                                           for a in evidence['separate_view_feasibility']]}),flush=True)


if __name__ == '__main__':
    main()
