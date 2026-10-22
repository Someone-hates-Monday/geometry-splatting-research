"""Exact fixed-radius SH transport identities and independent scalar quadrature; no GPU."""
import datetime
import hashlib
import json
from pathlib import Path
import sympy as s
import mpmath as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/tgs_sh_shell_v13'
a, t, u = s.symbols('a t u', positive=True)


def exact_coefficient(old_degree, new_degree):
    # u=sqrt(1+2*a*t+a*a), 0<a<1, so endpoints are exactly 1-a,1+a.
    tt = (u*u-1-a*a)/(2*a)
    e = (tt+a)/u
    integrand = s.expand(s.cancel(s.legendre(old_degree, e) * s.legendre(new_degree, tt)*u/a))
    primitive = 0
    for term in s.Add.make_args(integrand):
        power = term.as_powers_dict().get(u, s.Integer(0))
        coefficient = s.cancel(term/u**power)
        assert not coefficient.has(u)
        primitive += coefficient*(s.log(u) if power == -1 else u**(power+1)/(power+1))
    assert s.cancel(s.diff(primitive, u)-integrand) == 0
    return s.factor(s.Rational(2*new_degree+1, 2) *
                    (primitive.subs(u, 1+a)-primitive.subs(u, 1-a)))


def main():
    OUT.mkdir(exist_ok=True)
    target = OUT/'math_results.json'
    assert not target.exists()
    mp.mp.dps = 60
    rows = []
    for n in range(8):
        formula = 2*a/3 if n == 0 else (-a)**(n-1)*(s.Rational(n,2*n-1)-s.Rational(n+2,2*n+3)*a*a)
        calculated = exact_coefficient(1, n)
        assert s.simplify(calculated-formula) == 0
        rows.append(dict(degree=n, exact_coefficient=str(calculated)))
    first_order = []
    for ell in range(1, 9):
        lhs = (1-t*t)*s.diff(s.legendre(ell,t),t)
        rhs = s.Rational(ell*(ell+1),2*ell+1)*(s.legendre(ell-1,t)-s.legendre(ell+1,t))
        assert s.expand(lhs-rhs) == 0
        first_order.append(dict(old_degree=ell, coefficient_down=str(s.Rational(ell*(ell+1),2*ell+1)),
                                coefficient_up=str(-s.Rational(ell*(ell+1),2*ell+1))))
    p3_to_p4 = exact_coefficient(3,4)
    assert s.limit(p3_to_p4/a,a,0,dir='+') == -s.Rational(12,7)
    lower = s.simplify(abs(rows and exact_coefficient(1,4).subs(a,s.Rational(1,2)))/12)
    assert lower == s.Rational(67,14784)
    # Independent mpmath quadrature in the ORIGINAL t coordinate, not transformed u.
    numeric = []
    for aa in ['0.01','0.1','0.25','0.5','0.75']:
        am = mp.mpf(aa)
        f = lambda x:(x+am)/mp.sqrt(1+2*am*x+am*am)
        coefficients = []
        for n in range(8):
            numerical = (2*n+1)/2*mp.quad(lambda x:f(x)*mp.legendre(n,x),[-1,0,1])
            exact = s.N(exact_coefficient(1,n).subs(a,s.Rational(aa)),65)
            residual = abs(numerical-mp.mpf(str(exact)))
            assert residual < mp.mpf('1e-48')
            coefficients.append(numerical)
        proj = lambda x:sum(coefficients[n]*mp.legendre(n,x) for n in range(4))
        e2 = mp.quad(lambda x:(f(x)-proj(x))**2,[-1,0,1])/2
        tail_one = abs(coefficients[4])/3
        assert mp.sqrt(e2) >= tail_one
        uniform_projection = 1+sum(coefficients[n]*(-1)**n for n in range(4))
        gridmax = max(abs(f(mp.mpf(i)/1000)-proj(mp.mpf(i)/1000)) for i in range(-1000,1001))
        assert abs(gridmax-uniform_projection) < mp.mpf('1e-48')
        # A degree-3 source already leaks to degree4 at first order in displacement.
        p3 = lambda x:mp.legendre(3,f(x))
        b4 = mp.mpf('4.5')*mp.quad(lambda x:p3(x)*mp.legendre(4,x),[-1,0,1])
        b4_exact = mp.mpf(str(s.N(p3_to_p4.subs(a,s.Rational(aa)),65)))
        assert abs(b4-b4_exact) < mp.mpf('1e-48')
        numeric.append(dict(relative_displacement=aa,
            degree1_to_degree3_L2_error=str(mp.sqrt(e2)),
            degree1_to_degree3_one_mode_lower_bound=str(tail_one),
            degree1_to_degree3_exact_projection_uniform_error=str(uniform_projection),
            degree3_to_degree3_P4_leakage=str(b4),
            degree3_to_degree3_one_mode_L2_lower_bound=str(abs(b4)/3)))
        print(json.dumps(dict(stage='quadrature_verified',relative_displacement=aa)),flush=True)
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        exact_checks_pass=True, sympy=s.__version__, mpmath=mp.__version__,
        degree1_coefficients=rows, first_order_zonal_transport=first_order,
        degree3_to_degree4_exact_coefficient=str(p3_to_p4),
        degree3_to_degree4_leading_coefficient='-12/7',
        color_amplitude_quarter_L3_half_displacement_lower_bound=str(lower),
        lower_bound_decimal=str(s.N(lower,40)), numeric_checks=numeric,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Real arithmetic, full uniform camera shell, 0<a<1, scalar unclamped fields; exact quadrature checks are not image/training results.',
        novelty_established=False, training=False, CUDA=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(exact_checks_pass=True,P3_to_P4=str(p3_to_p4),result=str(target))),flush=True)


if __name__=='__main__':
    main()
