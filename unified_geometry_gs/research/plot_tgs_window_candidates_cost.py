"""Provisional diagnostic figure: paired raw points and explicitly different context."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tgs_window_candidates_common import OUT,sha

def main():
    cost=OUT/'cost';target=cost/'paired_cost.png';assert not target.exists();a=json.loads((cost/'analysis.json').read_text());audit=json.loads((cost/'independent_audit.json').read_text());assert audit['all_verified'];assert list(a['engineering_screen'])==['coarse']
    with (cost/'paired_ratios.csv').open() as fp:raw=list(csv.DictReader(fp))
    style={'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42}
    with plt.rc_context(style):
        fig,(left,right)=plt.subplots(1,2,figsize=(9.8,4.9),sharey=True,layout='constrained');labels=[];data=[]
        for ci,c in enumerate(a['cameras']):
            labels.append(c['image']+(' (new camera)' if ci==6 else ''))
            vals=[float(r['coarse_oracle_pair_wall_ms']) for r in raw if int(r['camera'])==ci];yy=ci+np.linspace(-.13,.13,len(vals));left.scatter(vals,yy,s=12,color='#777777',alpha=.65,zorder=1,label='All 18 paired blocks' if ci==0 else None)
            q=c['paired_ratios']['coarse_oracle']['pair_wall_ms'];lo,hi=q['paired_block_bootstrap_percentiles_95'];m=q['median'];left.errorbar(m,ci,xerr=np.array([[m-lo],[hi-m]]),fmt='o',color='#0072B2',capsize=3,ms=5,lw=1.6,label='Median; descriptive 95% bootstrap' if ci==0 else None)
            for offset,v,marker,color in [(-.10,'coarse','o','#0072B2'),(.10,'oracle','s','#333333')]:
                q2=c['paired_ratios'][v+'_piecewise']['pair_wall_ms'];m2=q2['median'];lo2,hi2=q2['paired_block_bootstrap_percentiles_95'];right.errorbar(m2,ci+offset,xerr=np.array([[m2-lo2],[hi2-m2]]),fmt=marker,color=color,capsize=3,ms=4.5,lw=1.2,label=v+' / historical piecewise' if ci==0 else None)
            data.append(dict(camera=c['image'],coarse_oracle_median=m,bootstrap_025=lo,bootstrap_975=hi,coarse_piecewise_median=c['paired_ratios']['coarse_piecewise']['pair_wall_ms']['median'],oracle_piecewise_median=c['paired_ratios']['oracle_piecewise']['pair_wall_ms']['median']))
        for ax in [left,right]:ax.axvline(1,color='#555555',ls='--',lw=1);ax.grid(axis='x',alpha=.18);ax.set_xlabel('Within-block wall F+B time ratio (unitless)')
        left.set_yticks(range(7),labels);left.invert_yaxis();left.set_ylabel('Camera (one stump7000 checkpoint)');left.set_xlim(.60,1.03);right.set_xlim(.5,8.1)
        left.set_title('A  Same numerical forward: coarse / native64',loc='left',fontsize=10);right.set_title('B  Context: different piecewise numerical forward',loc='left',fontsize=10)
        left.legend(loc='lower left',bbox_to_anchor=(0,1.04),frameon=False,fontsize=7.5);right.legend(loc='lower left',bbox_to_anchor=(0,1.04),frameon=False,fontsize=7.5)
        fig.get_layout_engine().set(rect=(0,.17,1,.93));fig.suptitle('v9 fixed-input operator diagnostic | 2026-10-20',fontsize=11)
        fig.text(.5,.035,'Source: cost/results.json; 378 measured F+B pairs, all retained. 20,000 paired-block resamples.\nOne unlocked-clock session; intervals are descriptive, not scene/seed confidence intervals. No GT or training evaluation.\nOnly precision-passing coarse was timed. Panel axes differ; point positions (not bar lengths) encode ratios.',ha='center',va='bottom',fontsize=8)
        fig.savefig(target,dpi=300,facecolor='white');fig.savefig(cost/'paired_cost.pdf',facecolor='white');plt.close(fig)
    with (cost/'figure_data.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    meta=dict(script_sha256=sha(Path(__file__)),analysis_sha256=sha(cost/'analysis.json'),audit_sha256=sha(cost/'independent_audit.json'),files={s:sha(cost/s) for s in ['paired_cost.png','paired_cost.pdf','figure_data.csv']},size_inches=[9.8,4.9],dpi=300,matplotlib=matplotlib.__version__,destination='Provisional internal research diagnostic; no journal format claim.',alt_text='Seven camera ratios of coarse/native64 have paired medians about 0.734 to 0.749, with all raw paired blocks shown. Context panel shows coarse still about 4.94 to 5.37 times historical piecewise latency; the latter has different numerical semantics.',transforms='Paired ratios, camera median, frozen descriptive bootstrap; deterministic vertical offsets only, no exclusions.',uncertainty='Single-session descriptive 95 percent paired-block bootstrap, not independent scene/seed inference.')
    (cost/'figure_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(dict(figure=str(target),files=meta['files'])),flush=True)

if __name__=='__main__':main()
