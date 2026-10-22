"""Provisional scientific plot: calibration choices and every heldout observation."""
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prepare_tgs_policy_data_v28 import ROOT,OUT,sha,write
a=json.loads((OUT/'analysis.json').read_text());assert a['all_verified']
labels=['Signed mixed','Signed MSE','Perturbation','Largest area','Hash'];rows=[]
with plt.rc_context({'font.size':10,'pdf.fonttype':42}):
    fig,axes=plt.subplots(2,1,figsize=(10,7.8),layout='constrained')
    for ax,sc in zip(axes,a['scenes']):
        for x,p in enumerate(sc['policies'][:5]):
            ax.scatter(x-.16,p['calibration']['mean_mixed'],marker='x',s=52,color='#0072B2',label='Calibration mean' if x==0 else None,zorder=4)
            ax.scatter(x+.16,p['heldout']['mean_mixed'],marker='D',s=42,color='#D55E00',label='Heldout mean' if x==0 else None,zorder=4)
            ax.scatter(x+np.array([.09,.23]),p['heldout']['per_view_mixed'],marker='o',s=35,facecolors='none',edgecolors='#222222',label='Both heldout views' if x==0 else None,zorder=3)
            for cohort in ['calibration','heldout']:
                for vi,v in enumerate(p[cohort]['per_view_mixed']):rows.append(dict(scene=sc['scene'],policy=p['policy'],parent_id=p['parent_id'],cohort=cohort,view=vi,delta_mixed=v))
        ax.axhline(0,color='#555555',ls='--',lw=.9,label='No edit' if sc is a['scenes'][0] else None)
        ax.set_yscale('symlog',linthresh=1e-10,linscale=1);ax.set_xticks(range(5),labels)
        ax.set_title(sc['scene']+' | one frozen edit per forced policy (+3 triangles)',loc='left',fontsize=11)
        ax.set_ylabel('Change in image form\n0.8 L1 + 0.2 (1 - SSIM)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='outside upper center',ncols=4,fontsize=9)
    fig.supxlabel('Negative improves the fixed image form. Signed logarithmic y scale; linear within +/-1e-10.\nMeans use two views per cohort; circles are observations, not confidence intervals.\nTwo existing scenes, new views; fixed external FP32 colors with child-only SH (v28 r1).\nNo training, geometry evaluation, final-quality or population-superiority claim.',fontsize=9)
    for suffix in ['png','pdf']:
        p=OUT/f'policy_transfer.{suffix}';assert not p.exists();fig.savefig(p,dpi=180,facecolor='white')
    plt.close(fig)
p=OUT/'figure_data.csv'
with p.open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(OUT/'figure_provenance.json',dict(inputs={p.relative_to(ROOT).as_posix():sha(p) for p in [OUT/'analysis.json',ROOT/'plot_tgs_policy_transfer_v28.py',OUT/'figure_data.csv']},all_forced_policies=True,per_view_rows=len(rows),scale='symlog, linear threshold1e-10; presentation only, not selection threshold',means='two calibration or two heldout views, equal weights',uncertainty='none; dependent views displayed separately',fixed_external_FP32_color_contract=True,publisher_requirements='unspecified; provisional research figure',alt_text='Two scene panels compare calibration means, heldout means, and both individual heldout observations for five frozen one-edit rules. Negative changes improve the fixed image objective. Zero is no edit; this is not a training or final-quality result.'))
