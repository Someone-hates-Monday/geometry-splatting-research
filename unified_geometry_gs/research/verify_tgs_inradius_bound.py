"""Exact algebra and bounded numerical checks for the perspective radius bound.

The universal proof is in TGS_PERSPECTIVE_INRADIUS_BOUND.md. Numerical cases
are checks, not a substitute for that proof or a novelty claim.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import sympy as sp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_real_event_v1'


def stats(v):
    lengths=np.linalg.norm(np.roll(v,-1,axis=0)-v,axis=1)
    area=abs(np.linalg.det(np.stack([v[1]-v[0],v[2]-v[0]])))/2
    return area,float(lengths.sum()),float(2*area/lengths.sum())


def midpoint_projected(v,z):
    m01=(z[0]*v[0]+z[1]*v[1])/(z[0]+z[1])
    m02=(z[0]*v[0]+z[2]*v[2])/(z[0]+z[2])
    m12=(z[1]*v[1]+z[2]*v[2])/(z[1]+z[2])
    return np.array([[v[0],m01,m02],[v[1],m01,m12],[v[2],m02,m12],[m01,m02,m12]])


def main():
    x,y,z=sp.symbols('x y z',positive=True)
    v=[sp.Matrix([0,0]),sp.Matrix([1,0]),sp.Matrix([0,1])]
    m01=(x*v[0]+y*v[1])/(x+y);m02=(x*v[0]+z*v[2])/(x+z);m12=(y*v[1]+z*v[2])/(y+z)
    area_ratio=sp.factor(sp.det(sp.Matrix.hstack(m01-m02,m12-m02)))
    expected=2*x*y*z/((x+y)*(x+z)*(y+z))
    assert sp.simplify(area_ratio-expected)==0
    k,t=sp.symbols('k t',positive=True)
    q=sp.expand(((1+t)*(t+k)*(k+1))/(t*k))
    q_form=2+k+1/k+(1+1/k)*t+(1+k)/t
    assert sp.simplify(q-q_form)==0
    assert sp.simplify(sp.diff(q_form,t,2)-2*(1+k)/t**3)==0
    assert sp.simplify(q_form.subs(t,1)-2*(k+1)**2/k)==0
    assert sp.simplify(q_form.subs(t,k)-2*(k+1)**2/k)==0
    rng=np.random.default_rng(20261020)
    checked=0;skipped=0;min_margin=float('inf');max_perimeter_ratio=0.
    for _ in range(1500):
        vertices=rng.normal(size=(3,2));depth=np.exp(rng.uniform(-3,3,size=3))
        area,perimeter,radius=stats(vertices)
        if area<1e-5:
            skipped+=1;continue
        contrast=float(depth.max()/depth.min());lower=radius/(1+contrast)
        children=midpoint_projected(vertices,depth)
        child_stats=[stats(c) for c in children]
        actual=min(c[2] for c in child_stats)
        margin=actual/lower-1
        assert margin>=-1e-9
        assert child_stats[3][0]/area>=contrast/(1+contrast)**2-1e-10
        assert child_stats[3][1]/perimeter<=contrast/(1+contrast)+1e-10
        min_margin=min(min_margin,margin);max_perimeter_ratio=max(max_perimeter_ratio,child_stats[3][1]/perimeter)
        checked+=1
    sharp=[]
    for contrast in [1,2,10,100]:
        vertices=np.array([[0.,0.],[4.,0.],[0.,3.]])
        children=midpoint_projected(vertices,np.array([contrast,1.,1.]))
        ratio=stats(children[0])[2]/stats(vertices)[2]
        assert abs(ratio-1/(contrast+1))<1e-12
        sharp.append({'depth_contrast':contrast,'corner_radius_ratio':ratio,'predicted':1/(contrast+1)})
    results=json.loads((OUT/'results.json').read_text());real=[]
    for row in results['cases']:
        p=row['continuous_parent']
        if p is None:continue
        raw=np.load(OUT/f"event_{row['event_index']}_input.npz")
        camera=np.load(OUT/f"baseline_{row['camera']}.npz")
        full=camera['projection'].astype(float).T
        depth=(np.c_[raw['points'].astype(float),np.ones(3)]@full.T)[:,3]+1e-7
        assert (depth>0).all()
        contrast=float(depth.max()/depth.min())
        c=midpoint_projected(np.array(p['vertices']),depth)
        ideal_min=min(stats(v)[2] for v in c)
        lower=p['inradius']/(1+contrast)
        assert ideal_min>=lower*(1-1e-7)
        actual=[x['inradius'] for x in row['continuous_children'] if x is not None]
        native_cull=row['parent_radii']>0 and max(row['child_radii'])==0
        record={'event':row['event_index'],'camera':row['camera'],'native_parent_survives':row['parent_radii']>0,
                'parent_inradius_cpu':p['inradius'],'depth_contrast':contrast,'child_radius_lower_bound':lower,
                'ideal_projected_midpoint_child_radius_min':ideal_min,
                'actual_rounded_world_midpoint_child_radius_min':min(actual) if len(actual)==4 else None,
                'sufficient_no_inradius_cull_condition':lower>=1,
                'native_surviving_parent_all_children_culled':native_cull}
        if native_cull:
            assert len(actual)==4 and max(actual)<1 and lower<1
            # Other preprocess conditions are numerically checked separately
            # from the inradius inequality; no sufficiency claim for them.
            view=camera['view'].astype(float).T
            geometry=np.concatenate([raw['points'][None],raw['children']])
            conditions=[]
            for tri in geometry:
                center=tri.mean(0);vcenter=(view@np.r_[center,1])[:3]
                normal=view[:3,:3]@np.cross(tri[1]-tri[0],tri[2]-tri[0])
                normal/=max(np.linalg.norm(normal),1e-4)
                cosine=abs(float(normal@vcenter/max(np.linalg.norm(vcenter),1e-4)))
                conditions.append({'view_center_depth':float(vcenter[2]),'abs_cosine_with_native_normal_clamp':cosine})
                assert vcenter[2]>.2 and cosine>.001
            record['other_preprocess_conditions']=conditions
            record['opacity_above_preprocess_threshold']=float(raw['opacity'])>=.01
        real.append(record)
    output={'passed':True,'symbolic_central_area_ratio':str(area_ratio),
            'bound':'all four projected child inradii >= parent_inradius/(1+max_depth/min_depth)',
            'sharp_for_corner':True,'random_seed':20261020,'random_checks':checked,'skipped_near_degenerate':skipped,
            'random_min_relative_slack':min_margin,'random_max_central_perimeter_ratio':max_perimeter_ratio,
            'sharp_examples':sharp,'real_case_checks':real,
            'source_results_sha256':hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),
            'scope':'ideal projective geometry; native rounding checked descriptively; other culling/quality not guaranteed'}
    (OUT/'inradius_bound_checks.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'passed':True,'random_checks':checked,'real_positive_depth_cases':len(real),
                      'cull_witness':[x for x in real if x['native_surviving_parent_all_children_culled']]}),flush=True)


if __name__=='__main__':
    main()
