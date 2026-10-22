"""CPU reconstruction/loss audit of all24 joint images, including overlapping pairs."""
import json,math
import numpy as np
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,exact
from tgs_image_metrics_v11 import png8,ssim_map
from tgs_batch_score_common_v22 import cpu_sse
from tgs_joint_edit_common_v27 import OUT,paint,round_bound
V20=ROOT/'experiments/tgs_local_stress_v20';V23=ROOT/'experiments/tgs_view_transfer_v23'

def main():
    p=json.loads((OUT/'plan.json').read_text());r=json.loads((OUT/'results.json').read_text());assert r['all_passed'] and r['new_native_forwards']==32 and len(r['rows'])==24
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    assert r['source_sha256']==sha(ROOT/'run_tgs_joint_edit_v27.py')
    refs=json.loads((V23/'analysis.json').read_text());rows=[];max_error=0.
    for vp in p['views']:
        scene,view,W,H=vp['scene'],vp['view'],vp['W'],vp['H'];B=np.load(V20/scene/'baseline.npz')['rgb'] if view=='fresh' else np.load(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy')
        gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy' if view=='fresh' else ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy');G=gt.astype(np.float64)/255.;b=B.transpose(1,2,0).astype(np.float64);be=(b-G)**2;ba=np.abs(b-G);sb=ssim_map(b,G);sse=cpu_sse(png8(B),gt);n=gt.size;singles=[]
        for candidate in vp['candidates']:
            en=next(x for x in refs['rows'] if (x['scene'],x['view'],x['event'])==(scene,view,candidate['event']))
            path=V20/scene/f"parent_{en['parent_index']}"/'moved/query_patch.npz' if view=='fresh' else V23/scene/f"proposal_{en['event']}_rgb_patch.npz";patch=np.load(path);x0,y0,x1,y1=patch['bbox'].tolist();rgb=B.copy();rgb[:,y0:y1,x0:x1]=patch['rgb'];assert ahash(rgb)==en['rgb_sha256']
            x=rgb.transpose(1,2,0).astype(np.float64);ss=ssim_map(x,G);terms=[(x-G)**2-be,np.abs(x-G)-ba,ss-sb];scores=[float(t.mean()) for t in terms]
            singles.append(dict(rgb=rgb,ss=ss,terms=terms,scores=scores,sse=en['delta_SSE'],changed=not exact(rgb,B)))
        for prediction in vp['pairs']:
            i,j=prediction['i'],prediction['j'];en=next(x for x in r['rows'] if (x['scene'],x['view'],x['i'],x['j'])==(scene,view,i,j));assert en['classification']==prediction
            patch=np.load(OUT/f'{scene}_{view}/pair_{i}_{j}_rgb_patch.npz');R=patch['bbox'].tolist();assert R==en['ROI'];x0,y0,x1,y1=R;rgb=B.copy();rgb[:,y0:y1,x0:x1]=patch['rgb'];assert ahash(rgb)==en['full_rgb_hash']
            t=rgb.transpose(1,2,0).astype(np.float64);ss=ssim_map(t,G);terms=[(t-G)**2-be,np.abs(t-G)-ba,ss-sb];scores=[float(v.mean()) for v in terms]
            err=np.abs(np.array(scores)-en['joint_score']);assert np.all(err<=p['score_atol']+p['score_rtol']*np.abs(scores));max_error=max(max_error,float(err.max()))
            dsse=cpu_sse(png8(rgb),gt)-sse;assert dsse==en['delta_PNG8_SSE'] and dsse-singles[i]['sse']-singles[j]['sse']==en['PNG8_interaction']
            outsideR=np.ones((H,W),dtype=bool);outsideA=np.ones((H,W),dtype=bool)
            for k in [i,j]:
                x0,y0,x1,y1=vp['candidates'][k]['ROI'];outsideR[y0:y1,x0:x1]=False
                x0,y0,x1,y1=vp['candidates'][k]['A'];outsideA[y0:y1,x0:x1]=False
            assert exact(rgb.transpose(1,2,0)[outsideR],B.transpose(1,2,0)[outsideR]) and exact(ss[outsideA],sb[outsideA])
            if prediction['point_disjoint']:
                expected=B.transpose(1,2,0).copy()
                for k in [i,j]:paint(expected,vp['candidates'][k]['ROI'],singles[k]['rgb'].transpose(1,2,0))
                assert exact(rgb.transpose(1,2,0),expected) and dsse==singles[i]['sse']+singles[j]['sse']
            if prediction['ssim_disjoint']:
                expected=sb.copy()
                for k in [i,j]:paint(expected,vp['candidates'][k]['A'],singles[k]['ss'])
                assert exact(ss,expected)
            eta=[math.fsum([scores[k],-singles[i]['scores'][k],-singles[j]['scores'][k]]) for k in range(3)];bounds=[round_bound([terms[k],singles[i]['terms'][k],singles[j]['terms'][k]],n) for k in range(3)]
            required=[0,1] if prediction['point_disjoint'] else []
            if prediction['ssim_disjoint']:required.append(2)
            assert all(abs(eta[k])<=bounds[k] and abs(en['interactions'][k])<=en['roundoff_bounds'][k] for k in required)
            row=dict(scene=scene,view=view,i=i,j=j,classification=prediction,CPU_joint_score=scores,CPU_interactions=eta,CPU_roundoff_bounds=bounds,
                joint_score_absolute_error=err.tolist(),PNG8_interaction=dsse-singles[i]['sse']-singles[j]['sse'],
                point_mosaic_checked=prediction['point_disjoint'],SSIM_mosaic_checked=prediction['ssim_disjoint'],unchanged_outside_structural_union=True,
                both_single_RGB_effects_nonzero=singles[i]['changed'] and singles[j]['changed'])
            assert row['both_single_RGB_effects_nonzero']==en['both_single_RGB_effects_nonzero'];rows.append(row)
            print(json.dumps(dict(scene=scene,view=view,pair=[i,j],CPU_audit_passed=True,point_disjoint=prediction['point_disjoint'],SSIM_interaction=eta[2])),flush=True)
    assert len(rows)==24 and len({(x['scene'],x['view'],x['i'],x['j']) for x in rows})==24
    out=dict(all_passed=True,rows=rows,max_CPU_GPU_joint_score_error=max_error,training=False,new_native_forwards=0,
        point_disjoint_pairs=sum(x['classification']['point_disjoint'] for x in rows),ssim_disjoint_pairs=sum(x['classification']['ssim_disjoint'] for x in rows),
        nontrivial_ssim_disjoint_pairs=sum(x['classification']['ssim_disjoint'] and x['both_single_RGB_effects_nonzero'] for x in rows),
        inputs={str(path.relative_to(ROOT)):sha(path) for path in [OUT/'plan.json',OUT/'results.json',Path(__file__)]})
    write_new(OUT/'independent_audit.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['rows','inputs']}),flush=True)

if __name__=='__main__':main()
