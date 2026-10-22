"""CPU arithmetic/archive replay of one-view inference smoke; no rerender."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def hostpath(p):return Path(p.replace('/mnt/c/','C:/',1)) if p.startswith('/mnt/c/') else Path(p)
def quantile(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return dict(zip(('min','p50','p90','p99','max'),np.quantile(x,(0,.5,.9,.99,1)).tolist()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();out=a.out.resolve()
    if out.exists():raise ValueError('Fresh report required.')
    manifest=read(run/'manifest.json');summary=read(run/'summary.json');assert manifest['status']=='complete'
    for name,digest in manifest['original_source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in manifest['artifacts_sha256'].items():assert sha(run/name)==digest,name
    generated_count=0
    with zipfile.ZipFile(run/'sources.zip') as z:
        for name,digest in manifest['original_source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
        for variant in ('affine','exact_packet','exact_factor'):
            meta=read(run/f'backend_{variant}.json')
            for name,digest in meta['generated_sha256'].items():
                assert hashlib.sha256(z.read(f'generated/{variant}/{name}')).hexdigest()==digest,name;generated_count+=1
            binary=hostpath(next(x for x in [str((ROOT/'research/build').resolve())] if x))
            # Locate content-addressed build without using untrusted stored paths.
            build=ROOT/'research/build'/f'exact_perspective_{variant}_{meta["signature"]}'
            binaries=list(build.glob('*.so'));assert len(binaries)==1 and sha(binaries[0])==meta['binary_sha256']
            for name,digest in meta['generated_sha256'].items():assert sha(build/'src'/name)==digest,name
    frames={}
    for name in ('installed','affine','exact_packet','exact_factor'):
        with np.load(run/f'image_{name}.npz',allow_pickle=False) as npz:frames[name]={k:npz[k] for k in npz.files}
        for k,value in frames[name].items():assert int(np.sum(~np.isfinite(value)))==summary['outputs'][name]['nonfinite_components'][k]
        assert int(np.sum(frames[name]['median_depth']>0))==summary['outputs'][name]['positive_depth_pixels']
    fields=list(frames['affine']);replayed_pairs=0
    for pair,record in summary['comparisons'].items():
        left,right=pair.split('_vs_');x,y=frames[left],frames[right]
        for k in fields:
            assert bool(np.array_equal(x[k],y[k]))==record[k]['bitwise_equal']
            delta=abs(x[k]-y[k]);assert float(np.max(delta[np.isfinite(delta)]))==record[k]['max_absolute_finite']
        positive=(x['median_depth']>0)&(y['median_depth']>0)
        assert int(np.sum((x['median_depth']>0)!=(y['median_depth']>0)))==record['depth_mask_disagreement_pixels']
        assert int(np.sum(positive))==record['depth_common_positive_pixels']
        relative=np.max(abs(x['median_depth'][positive]-y['median_depth'][positive])/np.maximum(1e-4,abs(y['median_depth'][positive])))
        assert float(relative)==record['max_relative_depth_error_common_positive'];replayed_pairs+=1
    b=frames['exact_packet'];c=frames['exact_factor'];depthcommon=(b['median_depth']>0)&(c['median_depth']>0)
    slopeb=b['interface_slope'].squeeze().astype(float);slopec=c['interface_slope'].squeeze().astype(float)
    slopediff=abs(slopeb-slopec);slopecommon=(slopeb>0)&(slopec>0)
    worst=np.argsort(slopediff.ravel(),kind='stable')[-5:][::-1];worstrows=[]
    for idx in worst:
        y,x=np.unravel_index(idx,slopediff.shape)
        worstrows.append({'pixel':[int(x),int(y)],'absolute_slope_difference':float(slopediff[y,x]),
            'packet_depth':float(b['median_depth'][0,y,x]),'factor_depth':float(c['median_depth'][0,y,x]),
            'packet_slope':float(slopeb[y,x]),'factor_slope':float(slopec[y,x])})
    path=ROOT/'output_compare/ggs_pipeline/scan37/point_cloud/iteration_30000/point_cloud.ply'
    old=read(ROOT/'research/results/median_cpu_scan37.json');plyhash=sha(path)
    assert plyhash==next(v for k,v in old['source_sha256'].items() if k.endswith('point_cloud.ply'))
    preserved=all(np.array_equal(frames['affine'][k],frames[v][k]) for v in ('exact_packet','exact_factor') for k in ('render','mask','normal','radii'))
    control=all(np.array_equal(frames['installed'][k],frames['affine'][k]) for k in fields)
    pfrac=summary['comparisons']['exact_packet_vs_exact_factor']['max_relative_depth_error_common_positive']
    gate=preserved and control and all(np.all(np.isfinite(v)) for f in frames.values() for v in f.values()) and np.array_equal(b['median_depth']>0,c['median_depth']>0) and pfrac<=1e-3
    assert bool(gate)==summary['overall_gate_pass']
    result={'status':'complete','source_hashes':len(manifest['original_source_sha256']),'artifact_hashes':len(manifest['artifacts_sha256']),
        'generated_archive_and_actual_hashes':generated_count,'binaries_checked':3,'pairs_replayed':replayed_pairs,
        'current_model_ply_sha256_matches_old_frozen_reference':plyhash,'overall_gate_pass_replayed':bool(gate),
        'packet_factor_slope_absolute_error_all_pixels':quantile(slopediff),
        'packet_factor_slope_relative_error_common_positive_floor1e-6':quantile(slopediff[slopecommon]/np.maximum(1e-6,abs(slopec[slopecommon]))),
        'posthoc_worst_slope_pixels':worstrows,
        'limits':['Archive/hash and saved-array arithmetic replay, not an independent CUDA implementation or rerender.',
            'Original smoke manifest did not hash the model PLY at time of rendering. Current PLY matches older frozen reference; this check cannot independently certify time-of-use bytes.',
            'Slope differences are descriptive posthoc diagnostics, without a newly invented pass threshold.',
            'Depth agreement does not establish interface-slope, Jacobian, GT quality, efficiency, NCC or normal consistency.']}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
