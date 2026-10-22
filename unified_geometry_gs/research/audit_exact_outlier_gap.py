"""Independent MP180 direct-H check of all eighteen peak-free gap bounds."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    run=ROOT/'research/experiments/exact_outlier_gap_v1'
    summary=read(run/'summary.json');figure=read(run/'figure_manifest.json')
    assert summary['status']=='complete'
    for name,h in summary['source_sha256'].items():assert sha(ROOT/name)==h
    for name,h in figure['artifacts_sha256'].items():assert sha(run/name)==h
    assert figure['source_summary_sha256']==sha(run/'summary.json')
    mp.mp.dps=180
    rows=[];points_checked=0;inconclusive=0
    tol=mp.mpf('1e-44')  # Stored bounds/interval endpoints carry 50 digits.
    for row in summary['rows']:
        case=row['case'];npz=ROOT/f'research/experiments/exact_outlier_capture_v3/ray_{case:02d}.npz'
        assert sha(npz)==summary['input_sha256'][f'ray_{case:02d}.npz']
        lo,hi=map(mp.mpf,row['central_60_percent_gap']);mid=mp.mpf(row['midpoint'])
        with np.load(npz,allow_pickle=False) as z:
            for arm,result in row['arms'].items():
                alpha,peak,rho=[[mp.mpf(float(v)) for v in z[f'{arm}_{name}']] for name in ('alpha','peak','rho')]
                assert sum(lo<t<hi for t in peak)==result['interior_peaks']==0
                def direct_H(t):
                    terms=[]
                    for a,q,r in zip(alpha,peak,rho):
                        g=a*mp.exp(-((t-q)*r)**2/2)
                        near=-mp.log1p(-g)/2
                        terms.append(near if t<=q else -mp.log1p(-a)-near)
                    return mp.fsum(terms)
                lower=mp.mpf(result['uniform_residual_lower_bound']);upper=mp.mpf(result['uniform_residual_upper_bound'])
                samples=[]
                for k in range(7):
                    t=lo+(hi-lo)*k/6
                    value=direct_H(t)-mp.log(2)
                    assert lower-tol<=value<=upper+tol,(case,arm,k)
                    samples.append(mp.nstr(value,60));points_checked+=1
                actual_mid=direct_H(mid)-mp.log(2)
                assert abs(actual_mid-mp.mpf(result['midpoint_residual']))<tol
                assert mp.sign(actual_mid)==mp.sign(mp.mpf(result['midpoint_residual']))
                slope=mp.fsum(a*mp.exp(-((mid-q)*r)**2/2)*abs((mid-q)*r)*r/(2*(1-a*mp.exp(-((mid-q)*r)**2/2))) for a,q,r in zip(alpha,peak,rho))
                assert abs(slope/mp.mpf(result['midpoint_H_slope'])-1)<mp.mpf('1e-40')
                classification=result['sufficient_classification']
                if classification=='root_left_of_gap':assert lower>0 and samples[0] and direct_H(lo)>mp.log(2)
                elif classification=='root_right_of_gap':assert upper<0 and direct_H(hi)<mp.log(2)
                else:
                    assert classification=='bound_inconclusive' and lower<=0<=upper
                    inconclusive+=1
                rows.append({'case':case,'arm':arm,'classification':classification,'direct_H_residual_samples':samples,
                             'direct_midpoint_residual':mp.nstr(actual_mid,60),'direct_midpoint_H_slope':mp.nstr(slope,60)})
    from PIL import Image
    with Image.open(run/'gap_evidence.png') as im:
        im.verify()
    with Image.open(run/'gap_evidence.png') as im:
        dimensions=list(im.size);dpi=im.info.get('dpi')
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','dps':180,'models':len(rows),
            'direct_H_point_checks':points_checked,'bound_inconclusive_retained':inconclusive,'rows':rows,
            'PNG_dimensions':dimensions,'PNG_dpi':dpi,'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
            'checked_summary_sha256':sha(run/'summary.json'),'checked_figure_manifest_sha256':sha(run/'figure_manifest.json'),
            'limits':['Direct-H implementation does not import the original Coefficients or bounds function.',
                      'Seven samples per model are numerical cross-checks; the full-interval inequality rests on the analytic nonnegative-tail monotonicity argument.',
                      'MP signs and tolerance are not outward-directed interval proof.',
                      'PNG integrity and dimensions are checked; publisher compliance and full accessibility are not established.']}
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','limits','source_sha256')},indent=2))


if __name__=='__main__':main()
