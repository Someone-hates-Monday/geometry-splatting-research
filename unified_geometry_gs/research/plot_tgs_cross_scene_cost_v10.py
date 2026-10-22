"""All primary paired points, descriptive intervals, and different-numerics context."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tgs_cross_scene_common_v10 import OUT,sha

def main():
    cost=OUT/'cost';assert not (cost/'paired_cost.png').exists();a=json.loads((cost/'analysis.json').read_text());audit=json.loads((cost/'independent_audit.json').read_text());assert audit['all_verified']
    with (cost/'paired_ratios.csv').open() as f:raw=list(csv.DictReader(f))
    style={'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42};data=[]
    with plt.rc_context(style):
        fig,(left,right)=plt.subplots(1,2,figsize=(10.0,4.7),sharey=True);fig.subplots_adjust(left=.29,bottom=.30,top=.75,right=.985,wspace=.16);labels=[];all_primary=[];context_upper=[]
        for ci,c in enumerate(a['cameras']):
            labels.append(c['scene']+' / '+c['image']+' ('+c['split']+')');values=[float(r['coarse_oracle_pair_wall_ms']) for r in raw if int(r['camera'])==ci];all_primary.extend(values);left.scatter(values,ci+np.linspace(-.12,.12,len(values)),s=13,color='#777777',alpha=.65,label='All paired blocks' if ci==0 else None)
            q=c['paired_ratios']['coarse_oracle']['pair_wall_ms'];med=q['median'];lo,hi=q['paired_block_bootstrap_percentiles_95'];left.errorbar(med,ci,xerr=np.array([[med-lo],[hi-med]]),fmt='o',color='#0072B2',capsize=3,ms=5,lw=1.5,label='Median + descriptive 95%' if ci==0 else None)
            for offset,v,marker,color in [(-.10,'coarse','o','#0072B2'),(.10,'oracle','s','#333333')]:
                r=c['paired_ratios'][v+'_piecewise']['pair_wall_ms'];m=r['median'];l,h=r['paired_block_bootstrap_percentiles_95'];context_upper.append(h);right.errorbar(m,ci+offset,xerr=np.array([[m-l],[h-m]]),fmt=marker,color=color,capsize=3,ms=4.5,label=v+' / piecewise' if ci==0 else None)
            data.append(dict(scene=c['scene'],camera=c['image'],split=c['split'],coarse_oracle_median=med,bootstrap_025=lo,bootstrap_975=hi,coarse_piecewise_median=c['paired_ratios']['coarse_piecewise']['pair_wall_ms']['median'],oracle_piecewise_median=c['paired_ratios']['oracle_piecewise']['pair_wall_ms']['median']))
        left.set_yticks(range(4),labels);left.invert_yaxis();left.set_ylabel('Scene / camera (split)');left.set_xlim(min(.60,min(all_primary)-.03),max(1.03,max(all_primary)+.03));right.set_xlim(.5,max(context_upper)+.4)
        for ax in [left,right]:ax.axvline(1,color='#555555',ls='--',lw=1);ax.grid(axis='x',alpha=.18);ax.set_xlabel('Within-block wall F+B time ratio (unitless)')
        left.set_title('A  coarse / native64\nSame numerical forward',loc='left',fontsize=10);right.set_title('B  Historical piecewise context\nDifferent numerical forward',loc='left',fontsize=10)
        h,l=left.get_legend_handles_labels();hr,lr=right.get_legend_handles_labels();fig.legend(h+hr,l+lr,loc='upper center',bbox_to_anchor=(.62,.21),ncol=2,frameon=False,fontsize=8);fig.suptitle('v10 cross-scene fixed-input operator diagnostic | 2026-10-20',y=.97,fontsize=11)
        fig.text(.5,.035,'Source: cost/results.json; 216 measured F+B pairs, all retained. 20,000 paired-block resamples.\nTwo scene phases in one unlocked-clock session; intervals are descriptive, not scene/seed confidence intervals.\nPanel axes differ. Native API includes allocation and host synchronization; no GT or training evaluation.',ha='center',va='bottom',fontsize=8)
        fig.savefig(cost/'paired_cost.png',dpi=300,facecolor='white');fig.savefig(cost/'paired_cost.pdf',facecolor='white');plt.close(fig)
    with (cost/'figure_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    out=dict(script_sha256=sha(Path(__file__)),analysis_sha256=sha(cost/'analysis.json'),audit_sha256=sha(cost/'independent_audit.json'),files={s:sha(cost/s) for s in ['paired_cost.png','paired_cost.pdf','figure_data.csv']},size_inches=[10.,4.7],dpi=300,matplotlib=matplotlib.__version__,scope='Provisional internal scientific diagnostic; all primary observations retained, deterministic vertical offsets only, no journal compliance claim.',alt_text='Four preselected cameras from flowers and room; raw paired coarse/native64 timing ratios, medians and descriptive bootstrap intervals. A separate panel retains ratios to the numerically different historical piecewise implementation.');(cost/'figure_metadata.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(figure=str(cost/'paired_cost.png'))),flush=True)

if __name__=='__main__':main()
