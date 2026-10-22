"""Replay sources, memberships, baseline roots and factorial/path arithmetic."""
from __future__ import annotations
import argparse,hashlib,itertools,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from research.median_solver import log_fields

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def stats(v):return dict(zip(('min','median','max'),map(float,np.quantile(v,[0,.5,1])))) if len(v) else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    m=read(args.run/'manifest.json');assert m['status']=='complete'
    with zipfile.ZipFile(args.run/'sources.zip') as archive:
        assert set(archive.namelist())==set(m['source_sha256'])
        for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest and hashlib.sha256(archive.read(name)).hexdigest()==digest
    for name,digest in m['artifacts_sha256'].items():assert sha(args.run/name)==digest
    rows=[];rooterrors=[]
    for i in range(24):
        row=read(args.run/f'ray_{i:02d}.json');old=read(ROOT/f'research/experiments/world_affine_rays_v1/ray_{i:02d}.json')
        assert row['scan']==old['scan'] and row['camera']==old['camera'] and row['pixel']==old['pixel']
        with np.load(args.run/f'ray_{i:02d}.npz',allow_pickle=False) as data:
            n=int(data['model_count']);assert n==row['gaussian_count']
            masks={name:np.unpackbits(data['membership_'+name],bitorder='little',count=n).astype(bool) for name in row['stage_membership_counts']}
            for name,mask in masks.items():assert sum(mask)==row['stage_membership_counts'][name]
            assert all(np.all(~masks[a]|masks[b]) for a,b in zip(('selected','eligible','tile','valid'),('eligible','tile','valid','near')))
            assert np.array_equal(np.sort(data['selected_ids']),np.flatnonzero(masks['selected']))
            assert len(np.unique(data['selected_ids']))==sum(masks['selected'])
            for mask in range(8):
                a=np.minimum(.99,data['world_alpha_selected']) if mask&4 else data['affine_alpha']
                peaks=data['world_peaks_selected'] if mask&1 else data['affine_peaks']
                widths=data['world_widths_selected'] if mask&2 else data['affine_widths']
                arm=row['factorial_arms'][mask]
                assert arm['finite']==(-np.log1p(-a).sum()>np.log(2))
                if arm['finite']:
                    t=arm['depth'];assert abs(log_fields(t,a,peaks,widths)[0]+np.log(2))<=1e-8
                    span=max(1.,abs(t)*2)
                    while log_fields(-span,a,peaks,widths)[0]<-np.log(2) or log_fields(span,a,peaks,widths)[0]>-np.log(2):span*=2
                    independent=brentq(lambda x:log_fields(x,a,peaks,widths)[0]+np.log(2),-span,span,xtol=5e-13,rtol=1e-13)
                    error=abs(independent-t);assert error<=1e-9*max(1.,abs(t));rooterrors.append(error)
            values=np.zeros(3)
            for order in itertools.permutations(range(3)):
                previous=0
                for factor in order:
                    nxt=previous|(1<<factor);values[factor]+=row['factorial_arms'][nxt]['depth']-row['factorial_arms'][previous]['depth'];previous=nxt
            values/=6
            assert values.tolist()==[row['permutation_average_allocation'][key] for key in ('peak','width','amplitude')]
        stages=row['world_stages'];names=['selected_cap','selected','eligible','tile','valid','near','full','full_camera']
        for key,before,after in zip(row['ordered_increments'],names,names[1:]):
            assert row['ordered_increments'][key]==stages[after]['depth']-stages[before]['depth']
        vector=values.tolist()+list(row['ordered_increments'].values());total=stages['full_camera']['depth']-row['factorial_arms'][0]['depth']
        assert abs(sum(vector)-total)<=1e-9
        assert stages['full_camera']['depth']==old['world_camera_median']
        rows.append(row)
    summary=read(args.run/'summary.json');assert summary['points']==summary['complete_factorial_cases']==summary['complete_telescope_cases']==24
    for key,item in summary['component_summaries'].items():
        if key.startswith('factor_'):values=[abs(r['permutation_average_allocation'][key.removeprefix('factor_')])/abs(r['legacy_affine_depth']) for r in rows]
        else:values=[abs(r['ordered_increments'][key])/abs(r['legacy_affine_depth']) for r in rows]
        assert item['defined']==len(values) and item['absolute_increment_over_legacy_depth']==stats(values)
    for scan in (37,55,63):
        old=read(ROOT/f'research/results/median_cpu_scan{scan}.json')
        ply=ROOT/f'output_compare/ggs_pipeline/scan{scan}/point_cloud/iteration_30000/point_cloud.ply'
        assert sha(ply)==next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','source_sha256':sha(Path(__file__)),
      'verified_sources_and_zip':len(m['source_sha256']),'verified_artifacts':len(m['artifacts_sha256']),'verified_PLYs':3,
      'memberships_npz_and_all_records':24,'factorial_roots_independently_resolved_with_original_log_fields':len(rooterrors),
      'max_independent_factorial_root_error':max(rooterrors),'factorial_and_path_arithmetic':True,
      'limits':['No independent re-evaluation of the large world selection-stage roots.',
                'CPU reference identity does not prove installed CUDA parity or reconstruction relevance.',
                'No GT, causal allocation, novelty, quality or acceleration claim.']}
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
