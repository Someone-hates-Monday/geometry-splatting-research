"""Freeze and render every known moved-parent pair;32 native forward calls total."""
import argparse,gc,itertools,json,math,traceback
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,HELPER,sha,ahash,write_new,module,pack,colors,CrossWorkload,exact
from tgs_aux_ablation_common_v21 import control,view_list,geometry_equal
from tgs_batch_score_common_v22 import backend as planner_backend,bank,cpu_sse,child_only
from tgs_local_image_loss_common_v24 import layout,device_layout
from tgs_shared_image_loss_common_v26 import backend as loss_backend,evaluate
from run_tgs_local_split_v19 import cache,same_bits
from trace_tgs_window_scene_v15 import call
from tgs_image_metrics_v11 import png8,ssim_map
from tgs_joint_edit_common_v27 import OUT,empty,expand,overlaps,union,paint,round_bound,split_pair
V20=ROOT/'experiments/tgs_local_stress_v20';V23=ROOT/'experiments/tgs_view_transfer_v23';V25=ROOT/'experiments/tgs_packed_query_v25';V26=ROOT/'experiments/tgs_shared_loss_v26'

def prepare():
    OUT.mkdir();old=json.loads((V26/'results.json').read_text());assert old['all_passed'];refs=json.loads((V23/'analysis.json').read_text());views=[]
    for scene in ['flowers','room']:
        for view in ['fresh','old_0']:
            H,W=(828,1256) if scene=='flowers' else (1038,1557);candidates=[]
            for ref in [r for r in refs['rows'] if r['scene']==scene and r['view']==view and r['appearance']=='moved']:
                en=next(r for r in old['rows'] if (r['scene'],r['view'],r['event'])==(scene,view,ref['event']))
                candidates.append(dict(parent_id=ref['parent_id'],parent_index=ref['parent_index'],event=ref['event'],ROI=en['ROI'],A=expand(en['ROI'],W,H)))
            assert len(candidates)==4 and [r['parent_index'] for r in candidates]==list(range(4))
            pairs=[dict(i=i,j=j,point_disjoint=not overlaps(candidates[i]['ROI'],candidates[j]['ROI']),ssim_disjoint=not overlaps(candidates[i]['A'],candidates[j]['A'])) for i,j in itertools.combinations(range(4),2)]
            views.append(dict(scene=scene,view=view,W=W,H=H,candidates=candidates,pairs=pairs))
    files=[Path(__file__),ROOT/'tgs_joint_edit_common_v27.py',ROOT/'TGS_JOINT_EDIT_PROTOCOL_V27.md',ROOT/'TGS_JOINT_EDIT_LOCALITY_NOTE_V27.md',V26/'results.json',V26/'evidence_manifest.json',V26/'build_receipt.json',V23/'analysis.json']
    for source in [V25/'plan.json',V25/'runtime_dependency_addendum.json',V26/'plan.json']:
        files.append(source);files.extend(ROOT/n for n in json.loads(source.read_text())['inputs'])
    files=sorted(set(files));write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},views=views,new_native_forward_cap=32,
        score_atol=1e-13,score_rtol=1e-9,training=False,accepted_mutations=0,source_conditioned_sufficient_not_necessary=True))
    print(json.dumps(dict(plan_frozen=True,pairs=24,point_disjoint=sum(x['point_disjoint'] for v in views for x in v['pairs']),ssim_disjoint=sum(x['ssim_disjoint'] for v in views for x in v['pairs']))),flush=True)

def frozen():
    p=json.loads((OUT/'plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    return p

def fixture_losses(B,I,J,X,G):
    arrays=[a.astype(np.float64) for a in [B,I,J,X]];b,i,j,x=arrays;n=b.size;err=b-G;be=err*err;ba=np.abs(err);bs=ssim_map(b,G);maps=[]
    for v in [i,j,x]:maps.append([(v-G)**2-be,np.abs(v-G)-ba,ssim_map(v,G)-bs])
    eta=[math.fsum([float(maps[2][k].mean()),-float(maps[0][k].mean()),-float(maps[1][k].mean())]) for k in range(3)]
    bounds=[round_bound([m[k] for m in maps],n) for k in range(3)];return eta,bounds

@torch.no_grad()
def fixtures():
    frozen();points=torch.arange(27,device='cuda',dtype=torch.float32).reshape(3,3,3);state=pack(points,torch.ones((3,1),device='cuda'),torch.full((3,1),.5,device='cuda'),torch.zeros((3,16,3),device='cuda'))
    col=torch.eye(3,device='cuda');children=[child_only(state,2),child_only(state,0)];joint,cc,keep=split_pair(state,[2,0],children,[col[2:3].repeat(4,1),col[:1].repeat(4,1)],col)
    assert same_bits(joint['points'][:1],state['points'][1:2]) and same_bits(joint['points'][1:5],children[0]['points']) and same_bits(joint['points'][5:],children[1]['points']) and same_bits(cc[:1],col[1:2])
    H=W=64;B=np.full((H,W,3),.25,dtype=np.float64);G=np.linspace(0,1,H*W*3).reshape(H,W,3);rows=[]
    for name,r1,r2 in [('separated',[5,5,9,9],[40,40,44,44]),('halo_overlap',[10,10,11,11],[12,10,13,11]),('pixel_overlap',[10,10,11,11],[10,10,11,11])]:
        I=B.copy();J=B.copy();X=B.copy()
        for image,r in [(I,r1),(J,r2),(X,r1),(X,r2)]:x0,y0,x1,y1=r;image[y0:y1,x0:x1]+=.125
        eta,bounds=fixture_losses(B,I,J,X,G);pd=not overlaps(r1,r2);sd=not overlaps(expand(r1,W,H),expand(r2,W,H))
        if pd:assert all(abs(eta[k])<=bounds[k] for k in [0,1])
        if sd:assert all(abs(eta[k])<=bounds[k] for k in range(3))
        if name=='halo_overlap':assert abs(eta[2])>bounds[2]
        if name=='pixel_overlap':assert abs(eta[0])>bounds[0]
        rows.append(dict(case=name,point_disjoint=pd,ssim_disjoint=sd,interactions=eta,bounds=bounds))
    a=Fraction(49,50);c=Fraction(99,100);v=Fraction(98,99);layers=[(a,Fraction(1)),(a,Fraction(1)),(c,v)]
    def comp(ls):
        T=Fraction(1);C=Fraction(0)
        for opacity,color in ls:
            test=T*(1-opacity)
            if test<Fraction(1,10000):break
            C+=T*opacity*color;T=test
        return C
    original=comp(layers);one=comp(layers[1:]);two=comp([layers[0],layers[2]]);both=comp(layers[2:]);assert original==one==two==Fraction(2499,2500) and both==Fraction(49,50)
    write_new(OUT/'fixtures.json',dict(all_passed=True,software_only=True,simultaneous_parent_order=True,rows=rows,
        rational_zero_observed_support_counterexample=dict(original=str(original),single_i=str(one),single_j=str(two),joint=str(both),generic_deletion_not_real_triangle_split=True)))
    print('JOINT_SOFTWARE_FIXTURES_PASSED',flush=True)

def visible_joint_gate(bg,cg,fg,keep,parents,i,j):
    ids=torch.nonzero(keep&(bg[6]>0)).flatten();newids=ids-(ids>parents[0]).long()-(ids>parents[1]).long();assert same_bits(bg[6][keep],fg[6][:-8])
    for field in range(9):
        if field!=6:assert same_bits(bg[field].index_select(0,ids),fg[field].index_select(0,newids)),field
    start=len(bg[6])-2
    for which,index in enumerate([i,j]):geometry_equal([g[4*index:4*index+4] for g in cg],[g[start+4*which:start+4*which+4] for g in fg])

def full_loss(m,base,gt,trial,full_layout,shared):
    out=evaluate(m,base,gt,trial.reshape(-1),full_layout,shared,True);torch.cuda.synchronize();assert not any(out[1].cpu().tolist())
    H,W=base.shape[1:];ss=out[2][:,1].reshape(3,H,W).contiguous();return out[0][0].cpu().numpy(),ss

@torch.no_grad()
def run():
    p=frozen();assert json.loads((OUT/'fixtures.json').read_text())['all_passed'];ctrl=control();planner=planner_backend();losser=loss_backend();helper=module(HELPER)
    prior=json.loads((V23/'analysis.json').read_text());prior_gpu=json.loads((V26/'results.json').read_text());forwards=0;rows=[]
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh)
        for vp in [v for v in p['views'] if v['scene']==scene]:
            view=vp['view'];H,W=vp['H'],vp['W'];folder=OUT/f'{scene}_{view}';folder.mkdir();cam=next(v['cam'] for v in view_list(w,scene) if v['label']==view);col=colors(helper,state,cam)
            original=call(ctrl,state,cam,col);torch.cuda.synchronize();forwards+=1;bg,_,_,_=cache(original,state,cam);base=original[1];B=base.cpu().numpy()
            oldB=np.load(V20/scene/'baseline.npz')['rgb'] if view=='fresh' else np.load(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy');assert exact(B,oldB)
            gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy' if view=='fresh' else ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy');gt8=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda();G=gt.astype(np.float64)/255.;b=B.transpose(1,2,0).astype(np.float64);be=(b-G)**2;ba=np.abs(b-G);n=gt.size
            fl=device_layout(layout([[0,0,W,H]],W,H));prepared=losser.prepare(base,gt8,fl['weights']);assert not any(prepared[1].cpu().tolist());shared=prepared[0];sb=shared[:,2].reshape(3,H,W).contiguous().cpu().numpy().transpose(1,2,0).copy()
            refs=[r for r in prior['rows'] if r['scene']==scene and r['view']==view and r['appearance']=='moved'];cases=[dict(parent_id=r['parent_id'],appearance='moved') for r in refs]
            bs,cc,parents,small,cc_list=bank(state,col,cam,cases,helper);tiny=call(ctrl,bs,cam,cc);torch.cuda.synchronize();forwards+=1;cg,_,_,_=cache(tiny,bs,cam);boxes,_=planner.plan(bg,cg,parents,W,H)
            actual_rois=[[x0*16,y0*16,min(W,x1*16),min(H,y1*16)] for x0,y0,x1,y1 in boxes.cpu().tolist()];assert actual_rois==[r['ROI'] for r in vp['candidates']]
            singles=[];base_sse=cpu_sse(png8(B),gt)
            for e,ref in enumerate(refs):
                old=V20/scene/f"parent_{ref['parent_index']}"/'moved/query_patch.npz' if view=='fresh' else V23/scene/f"proposal_{ref['event']}_rgb_patch.npz";data=np.load(old);x0,y0,x1,y1=data['bbox'].tolist();rgb=B.copy();rgb[:,y0:y1,x0:x1]=data['rgb'];assert ahash(rgb)==ref['rgb_sha256']
                raw=torch.from_numpy(rgb).cuda();score,ssg=full_loss(losser,base,gt8,raw,fl,shared);ss=ssg.cpu().numpy().transpose(1,2,0).copy();gpu_ref=next(r for r in prior_gpu['rows'] if (r['scene'],r['view'],r['event'])==(scene,view,ref['event']))
                assert exact(score,np.array(gpu_ref['full_gpu_score'],dtype=np.float64));v=rgb.transpose(1,2,0).astype(np.float64)
                singles.append(dict(rgb=rgb,score=score,ss=ss,terms=[(v-G)**2-be,np.abs(v-G)-ba,ss-sb],changed=not exact(rgb,B),sse=ref['delta_SSE']))
                del raw,ssg
            for pair in vp['pairs']:
                i,j=pair['i'],pair['j'];ps=[refs[i]['parent_id'],refs[j]['parent_id']];joint,jcol,keep=split_pair(state,ps,[small[i],small[j]],[cc_list[i],cc_list[j]],col)
                result=call(ctrl,joint,cam,jcol);torch.cuda.synchronize();forwards+=1;print(json.dumps(dict(scene=scene,view=view,pair=[i,j],joint_forward_complete=True,native_forwards=forwards)),flush=True)
                fg,_,_,_=cache(result,joint,cam);visible_joint_gate(bg,cg,fg,keep,ps,i,j);rgb=result[1].cpu().numpy();score,ssg=full_loss(losser,base,gt8,result[1],fl,shared);ss=ssg.cpu().numpy().transpose(1,2,0).copy()
                R=union(vp['candidates'][i]['ROI'],vp['candidates'][j]['ROI']);x0,y0,x1,y1=R;rebuilt=B.copy();rebuilt[:,y0:y1,x0:x1]=rgb[:,y0:y1,x0:x1];assert exact(rebuilt,rgb)
                outsideR=np.ones((H,W),dtype=bool);outsideA=np.ones((H,W),dtype=bool)
                for e in [i,j]:
                    rx0,ry0,rx1,ry1=vp['candidates'][e]['ROI'];outsideR[ry0:ry1,rx0:rx1]=False
                    ax0,ay0,ax1,ay1=vp['candidates'][e]['A'];outsideA[ay0:ay1,ax0:ax1]=False
                assert exact(rgb.transpose(1,2,0)[outsideR],B.transpose(1,2,0)[outsideR]) and exact(ss[outsideA],sb[outsideA])
                if pair['point_disjoint']:
                    expected=B.transpose(1,2,0).copy()
                    for e in [i,j]:paint(expected,vp['candidates'][e]['ROI'],singles[e]['rgb'].transpose(1,2,0))
                    assert exact(rgb.transpose(1,2,0),expected)
                if pair['ssim_disjoint']:
                    expectedS=sb.copy()
                    for e in [i,j]:paint(expectedS,vp['candidates'][e]['A'],singles[e]['ss'])
                    assert exact(ss,expectedS)
                v=rgb.transpose(1,2,0).astype(np.float64);terms=[(v-G)**2-be,np.abs(v-G)-ba,ss-sb];eta=[math.fsum([float(score[k]),-float(singles[i]['score'][k]),-float(singles[j]['score'][k])]) for k in range(3)]
                bounds=[round_bound([terms[k],singles[i]['terms'][k],singles[j]['terms'][k]],n) for k in range(3)]
                required=[0,1] if pair['point_disjoint'] else []
                if pair['ssim_disjoint']:required.append(2)
                assert all(abs(eta[k])<=bounds[k] for k in required),(scene,view,i,j,eta,bounds)
                dsse=cpu_sse(png8(rgb),gt)-base_sse;isum=singles[i]['sse']+singles[j]['sse']
                if pair['point_disjoint']:assert dsse==isum
                blends=lambda s:[.8*s[1]-.2*s[2],.8*s[0]-.2*s[2]]
                pair_blends=blends(score);predicted=[a+b for a,b in zip(blends(singles[i]['score']),blends(singles[j]['score']))]
                np.savez_compressed(folder/f'pair_{i}_{j}_rgb_patch.npz',bbox=np.array(R,dtype=np.int32),rgb=rgb[:,y0:y1,x0:x1])
                row=dict(scene=scene,view=view,i=i,j=j,parents=ps,classification=pair,ROI=R,full_rgb_hash=ahash(rgb),joint_score=score.tolist(),single_i_score=singles[i]['score'].tolist(),single_j_score=singles[j]['score'].tolist(),
                    interactions=eta,roundoff_bounds=bounds,required_zero_metric_indices=required,point_RGB_mosaic_literal=pair['point_disjoint'],SSIM_mosaic_literal=pair['ssim_disjoint'],
                    delta_PNG8_SSE=dsse,unary_PNG8_sum=isum,PNG8_interaction=dsse-isum,unary_blend_sum=predicted,joint_blends=pair_blends,
                    blend_interactions=[math.fsum([pair_blends[k],-predicted[k]]) for k in range(2)],both_single_RGB_effects_nonzero=singles[i]['changed'] and singles[j]['changed'],
                    external_child_visible_fields_literal=True,RGB_and_SSIM_unchanged_outside_structural_union=True,native_forwards=forwards)
                write_new(folder/f'pair_{i}_{j}.json',row);rows.append(row);del joint,jcol,keep,result,fg,rgb,score,ssg,ss,terms,rebuilt,v;gc.collect()
            assert initial==w.input_hashes();write_new(folder/'receipt.json',dict(pair_count=6,model_inputs_unchanged=True,native_forwards=forwards))
            del original,bg,base,B,gt8,b,be,ba,fl,prepared,shared,sb,bs,cc,parents,small,cc_list,tiny,cg,boxes,singles,col;gc.collect();torch.cuda.empty_cache()
        del w,state;gc.collect();torch.cuda.empty_cache()
    assert forwards==p['new_native_forward_cap']==32 and len(rows)==24;frozen()
    write_new(OUT/'results.json',dict(all_passed=True,rows=rows,new_native_forwards=forwards,training=False,accepted_mutations=0,source_sha256=sha(Path(__file__))))
    print('ALL24_JOINT_CASES_RECORDED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','fixtures','run']);args=parser.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
