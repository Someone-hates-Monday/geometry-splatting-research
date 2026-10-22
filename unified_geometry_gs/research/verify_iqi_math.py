"""Exact interpolation identities and independent smooth-function order check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Fresh output required.")
    y,y0,y1,y2,k = sp.symbols("y y0 y1 y2 k",real=True)
    polynomial = (y-y0)*(y-y2)/((y1-y0)*(y1-y2)) + k*(y-y0)*(y-y1)/((y2-y0)*(y2-y1))
    candidate = y0*y2/((y1-y0)*(y1-y2)) + k*y0*y1/((y2-y0)*(y2-y1))
    checks = [sp.cancel(polynomial.subs(y,y0)),sp.cancel(polynomial.subs(y,y1)-1),
        sp.cancel(polynomial.subs(y,y2)-k),sp.cancel(polynomial.subs(y,0)-candidate)]
    assert checks == [0,0,0,0]
    t = sp.symbols("t",real=True)
    T = sp.Function("T")(t)
    first = 1/sp.diff(T,t)
    second = sp.diff(first,t)/sp.diff(T,t)
    third = sp.diff(second,t)/sp.diff(T,t)
    expected = (3*sp.diff(T,t,2)**2-sp.diff(T,t)*sp.diff(T,t,3))/sp.diff(T,t)**5
    assert sp.cancel(third-expected) == 0
    rows=[]
    for width in (.2,.1,.05,.025):
        nodes=width*np.array([-.37,.63,-1.37])
        values=np.exp(-nodes)-1
        a,b,c=values
        linear=nodes[0]+width*a/(a-b)
        iq=nodes[0]+width*((a/(b-a))*(c/(b-c))-(a/(c-a))*(b/(c-b)))
        rows.append({"interval_width":width,"linear_error":abs(float(linear)),"iqi_error":abs(float(iq))})
    for i in range(1,len(rows)):
        rows[i]["linear_observed_order"]=float(np.log2(rows[i-1]["linear_error"]/rows[i]["linear_error"]))
        rows[i]["iqi_observed_order"]=float(np.log2(rows[i-1]["iqi_error"]/rows[i]["iqi_error"]))
    assert rows[-1]["iqi_observed_order"] > 2.9
    result={"symbolic_identities_pass":True,"inverse_third_derivative":str(expected),
        "smooth_exp_check":rows,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions":["Three distinct T values for the Lagrange identity.",
            "Inverse-function remainder requires T to be C3 with T' nonzero on the full three-node interval.",
            "GGGS peaks and critical roots can violate those assumptions; CUDA does not certify them.",
            "Tests verify formulas/order on a smooth independent function, not GS quality or GPU accuracy."]}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
