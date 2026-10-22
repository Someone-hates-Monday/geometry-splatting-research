"""Standalone rational-interval verifier: Python stdlib only, no LP solver.

sqrt bounds follow from integer squares; all subsequent endpoints are Fractions.
Camera distance D=k*sqrt(7), K=I, rotations as in TGS_SUBDIVISION_MOMENTS.md.
Screen coordinates are multiplied by D in both views (feasibility invariant).
"""
from fractions import Fraction as F
from math import isqrt
from itertools import permutations
import json
from pathlib import Path


class I:
    def __init__(self, lo=0, hi=None):
        self.lo = F(lo)
        self.hi = self.lo if hi is None else F(hi)
        assert self.lo <= self.hi

    @staticmethod
    def cast(x):
        return x if isinstance(x, I) else I(x)

    def __add__(self, other):
        b = I.cast(other)
        return I(self.lo+b.lo, self.hi+b.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -I.cast(other)

    def __rsub__(self, other):
        return I.cast(other) + -self

    def __mul__(self, other):
        b = I.cast(other)
        v = [self.lo*b.lo, self.lo*b.hi, self.hi*b.lo, self.hi*b.hi]
        return I(min(v), max(v))

    __rmul__ = __mul__

    def __truediv__(self, other):
        b = I.cast(other)
        assert not b.lo <= 0 <= b.hi, 'interval contains zero divisor'
        return self * I(1/b.hi, 1/b.lo)

    def mid(self):
        return float((self.lo+self.hi)/2)

    def record(self):
        return {'lo': str(self.lo), 'hi': str(self.hi),
                'display_midpoint': self.mid()}


def sqrt_q(q, digits=30):
    q = F(q)
    assert q >= 0
    scale = 10**digits
    n = isqrt(q.numerator*scale*scale // q.denominator)
    lo, hi = F(n, scale), F(n+1, scale)
    assert lo*lo <= q < hi*hi
    return I(lo, hi)


def triangle(v):
    # Coordinates are exact rationals; side-length radicals are enclosed.
    a = abs((v[1][0]-v[0][0])*(v[2][1]-v[0][1]) -
            (v[1][1]-v[0][1])*(v[2][0]-v[0][0]))/2
    assert a > 0
    sides = [sqrt_q(sum((v[(j+1)%3][d]-v[(j+2)%3][d])**2
                        for d in range(2))) for j in range(3)]
    s = [sum(sides[j]*v[j][d] for j in range(3))/sum(sides)
         for d in range(2)]
    # sigma=1 centroid (3g+s)/4.
    mu = [(sum(v[j][d] for j in range(3))+s[d])/4 for d in range(2)]
    return a, mu


def split(v):
    m01 = tuple((v[0][d]+v[1][d])/2 for d in range(len(v[0])))
    m02 = tuple((v[0][d]+v[2][d])/2 for d in range(len(v[0])))
    m12 = tuple((v[1][d]+v[2][d])/2 for d in range(len(v[0])))
    return [(v[0], m01, m02), (m01, v[1], m12),
            (m02, m12, v[2]), (m01, m12, m02)]


def systems(k=10):
    world = [(F(0), F(0)), (F(4), F(0)), (F(0), F(4))]
    children = split(world)
    mats, rhs, vertices, ratios = [], [], [], []
    for view in range(2):
        def project(p):
            x, y = p
            if view == 0:
                return x/(1-y/(4*k)), F(3,4)*y/(1-y/(4*k))
            return F(3,4)*x/(1-x/(4*k)), y/(1-x/(4*k))
        vp = [project(p) for p in world]
        vc = [[project(p) for p in c] for c in children]
        area, mu = triangle(vp)
        cm = [triangle(c) for c in vc]
        a = [x[0]/area for x in cm]
        assert sum(a) == 1
        mats.append([[I(x) for x in a]] +
                    [[a[j]*cm[j][1][d] for j in range(4)] for d in range(2)])
        rhs.append([I(F(1,2)), mu[0]/2, mu[1]/2])
        vertices.append({'parent': [[str(x) for x in p] for p in vp],
                         'children': [[[str(x) for x in p] for p in c] for c in vc]})
        ratios.append([str(x) for x in a])
    return mats, rhs, vertices, ratios


def det3(a):
    out = I(0)
    for p in permutations(range(3)):
        sign = (-1)**sum(p[i] > p[j] for i in range(3) for j in range(i+1, 3))
        out += sign*a[0][p[0]]*a[1][p[1]]*a[2][p[2]]
    return out


def verify(cert):
    assert isinstance(cert['k'], int) and cert['k'] > 1
    assert cert['sigma'] == 1 and cert['parent_opacity'] == '1/2'
    mats, rhs, vertices, ratios = systems(cert['k'])
    mat, b = mats[0]+mats[1], rhs[0]+rhs[1]
    y = [F(x) for x in cert['farkas_y']]
    assert len(y) == 6
    aty = [sum(y[i]*mat[i][j] for i in range(6)) for j in range(4)]
    bty = sum(y[i]*b[i] for i in range(6))
    assert all(x.lo > 0 for x in aty), 'failed nonnegative-cone separation'
    assert bty.hi < 0, 'failed negative RHS separation'
    # Dual test functions l_v(x)=y_(3v)+y_(3v+1)*x+y_(3v+2)*y.
    # On a triangle their absolute maxima occur at vertices (all rational).
    # For relative continuous L1 error e_v, |y.r| <= o*sum(C_v*e_v).
    test_max=[]
    for view_id in range(2):
        yy=y[3*view_id:3*view_id+3]
        test_max.append(max(abs(yy[0]+yy[1]*F(p[0])+yy[2]*F(p[1]))
                            for p in vertices[view_id]['parent']))
    margin=-bty.hi
    residual_bound=margin/sum(abs(x) for x in y)
    relative_l1_bound=margin/(F(1,2)*sum(test_max))
    assert residual_bound > 0 and relative_l1_bound > 0
    feasible = []
    for m, r in zip(mats, rhs):
        z4 = I(F(17,31))
        a = [row[:3] for row in m]
        b3 = [r[i]-m[i][3]*z4 for i in range(3)]
        determinant = det3(a)
        assert not determinant.lo <= 0 <= determinant.hi
        z = []
        for j in range(3):
            aj = [[b3[i] if c == j else a[i][c] for c in range(3)] for i in range(3)]
            z.append(det3(aj)/determinant)
        z.append(z4)
        assert all(x.lo > 0 and x.hi < 1 for x in z)
        feasible.append({'exact_solution_definition': 'Cramer rule, z4=17/31',
                         'opacity_enclosures': [x.record() for x in z],
                         'determinant': determinant.record()})
    return {'passed': True, 'proof': 'rational interval Farkas + Cramer rule',
            'sqrt_decimal_digits': 30, 'projected_vertices': vertices,
            'child_area_ratios': ratios, 'At_y': [x.record() for x in aty],
            'bt_y': bty.record(), 'separate_view_feasibility': feasible,
            'dual_test_abs_maxima':[str(x) for x in test_max],
            'residual_infinity_lower_bound':str(residual_bound),
            'worst_view_relative_continuous_L1_lower_bound':str(relative_l1_bound),
            'display_L1_lower_bound':float(relative_l1_bound),
            'L1_scope':'at least one of the two ideal continuous windows, any shared nonnegative child opacities; no clipping/culling/occlusion'}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('certificate', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = verify(json.loads(args.certificate.read_text()))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': result['passed'],
                      'At_y_midpoints': [x['display_midpoint'] for x in result['At_y']],
                      'bt_y_midpoint': result['bt_y']['display_midpoint']}))
