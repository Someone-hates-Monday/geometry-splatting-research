"""CPU lossless reconstruction and descriptive4-proposal cross-view contrasts."""
import json,traceback
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from PIL import Image
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,exact
from tgs_image_metrics_v11 import png8,ssim_map
from tgs_batch_score_common_v22 import cpu_sse
OUT=ROOT/'experiments/tgs_view_transfer_v23';V20=ROOT/'experiments/tgs_local_stress_v20';V22=ROOT/'experiments/tgs_batch_score_v22'
METRICS=['PNG8_MSE','raw_MSE','raw_L1','L1_DSSIM_form','L2_DSSIM_form']

def signs(x):return np.sign(np.array(x)).astype(int).tolist()

def ordering(a,b):
    ra=rankdata(a);rb=rankdata(b)
    rho=None if np.std(ra)==0 or np.std(rb)==0 else float(np.corrcoef(ra,rb)[0,1])
    relations=[(int(np.sign(a[j]-a[i])),int(np.sign(b[j]-b[i]))) for i in range(len(a)) for j in range(i+1,len(a))]
    return dict(n=len(a),rank_rho=rho,signs_first=signs(a),signs_second=signs(b),same_sign=sum(x==y for x,y in zip(signs(a),signs(b))),
        pairwise_concordant=sum(x*y>0 for x,y in relations),pairwise_discordant=sum(x*y<0 for x,y in relations),pairwise_tied=sum(x*y==0 for x,y in relations))

def main():
    plan=json.loads((OUT/'plan.json').read_text())
    for n,h in plan['inputs'].items():assert sha(ROOT/n)==h,n
    r=json.loads((OUT/'results.json').read_text());assert r['all_passed'] and r['native_forwards']==20 and len(r['rows'])==16
    known=json.loads((V22/'results.json').read_text());rows=[];baselines=[]
    for scene in ['flowers','room']:
        for view in ['fresh','old_0']:
            if view=='fresh':
                base=np.load(V20/scene/'baseline.npz')['rgb'];gtfile=ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy';gtpath=ROOT/f'sources/tgs_sh_isolation_v12/{scene}/fresh';refs=[x for x in known['rows'] if x['scene']==scene]
            else:
                base=np.load(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy');gtfile=ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy';gtpath=ROOT/f'sources/tgs_gt_identity_v11/{scene}';refs=[x for x in r['rows'] if x['scene']==scene]
            gt=np.load(gtfile)
            for name in ['original.JPG','published_gt.png']:assert np.array_equal(gt,np.asarray(Image.open(gtpath/name).convert('RGB')))
            b=base.transpose(1,2,0).astype(np.float64);g=gt.astype(np.float64)/255.;d=b-g;be=d*d;ba=np.abs(d);bs=ssim_map(b,g);sse=cpu_sse(png8(base),gt)
            baselines.append(dict(scene=scene,view=view,raw_MSE=float(be.mean()),raw_L1=float(ba.mean()),raw_SSIM=float(bs.mean()),PNG8_SSE=sse,raw_RGB_min=float(b.min()),raw_RGB_max=float(b.max())))
            for e,en in enumerate(refs):
                assert en['event']==e
                pf=V20/scene/f"parent_{en['parent_index']}"/en['appearance']/'query_patch.npz' if view=='fresh' else OUT/scene/f'proposal_{e}_rgb_patch.npz'
                patch=np.load(pf);x0,y0,x1,y1=patch['bbox'].tolist();trial=base.copy();trial[:,y0:y1,x0:x1]=patch['rgb']
                expected=en['reference_hashes']['rgb'] if view=='fresh' else en['full_rgb_sha256'];assert ahash(trial)==expected
                delta=cpu_sse(png8(trial),gt)-sse;assert delta==en['delta_SSE']==en['batch_score']==en['full_gpu_score']
                t=trial.transpose(1,2,0).astype(np.float64);err=t-g
                dmse=float(np.mean(err*err-be));dl1=float(np.mean(np.abs(err)-ba));dss=float(np.mean(ssim_map(t,g)-bs))
                changes=dict(PNG8_MSE=delta/(gt.size*255**2),raw_MSE=dmse,raw_L1=dl1,raw_SSIM=dss,L1_DSSIM_form=.8*dl1-.2*dss,L2_DSSIM_form=.8*dmse-.2*dss)
                row=dict(scene=scene,view=view,event=e,parent_id=en['parent_id'],parent_index=en['parent_index'],appearance=en['appearance'],delta_SSE=delta,changes=changes,rgb_sha256=expected)
                write_new(OUT/f'loss_{scene}_{view}_{e}.json',row);rows.append(row)
                print(json.dumps(dict(scene=scene,view=view,event=e,changes=changes)),flush=True)
    summaries=[]
    for scene in ['flowers','room']:
        for app in ['moved','hold']:
            fresh=[x for x in rows if x['scene']==scene and x['appearance']==app and x['view']=='fresh'];old=[x for x in rows if x['scene']==scene and x['appearance']==app and x['view']=='old_0'];assert len(fresh)==len(old)==4
            assert [x['parent_id'] for x in fresh]==[x['parent_id'] for x in old]
            cross={m:ordering([x['changes'][m] for x in fresh],[x['changes'][m] for x in old]) for m in METRICS}
            alignment={v:{m:ordering([x['changes']['PNG8_MSE'] for x in cases],[x['changes'][m] for x in cases]) for m in METRICS[1:]} for v,cases in [('fresh',fresh),('old_0',old)]}
            summaries.append(dict(scene=scene,appearance=app,parent_ids=[x['parent_id'] for x in fresh],cross_view=cross,within_view_alignment=alignment))
    write_new(OUT/'analysis.json',dict(all_verified=True,baselines=baselines,rows=rows,summaries=summaries,descriptive_only=True,no_independent_scene_test=True,
        training=False,actual_checkpoint_training_loss_branch_known=False,source_sha256=sha(Path(__file__)),render_results_sha256=sha(OUT/'results.json')))
    print(json.dumps(dict(primary=[x for x in summaries if x['appearance']=='moved'])),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        with (OUT/'analysis_failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
