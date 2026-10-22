"""All raw calls/ratios; provisional scientific figure, no journal compliance claim."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_phase_cost_v5/cost'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'figure_metadata.json';assert not target.exists();data=json.loads((OUT/'results.json').read_text());analysis=json.loads((OUT/'analysis.json').read_text());rows=data['rows'];rng=np.random.default_rng(202610208)
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(11.2,4.4),layout='constrained');ax=axes[0];labels=[]
        for ci,c in enumerate(analysis['cameras']):
            values=[]
            for block in sorted(set(r['block'] for r in rows if r['camera']==ci)):
                m={r['variant']:r for r in rows if r['block']==block};values.append(m['guard']['pair_wall_ms']/m['oracle']['pair_wall_ms'])
            x=ci+rng.uniform(-.075,.075,len(values));ax.scatter(x,values,s=19,color='#0072B2',alpha=.65,label='18 paired blocks/view' if ci==0 else None)
            stats=c['paired_ratios']['guard_oracle']['pair_wall_ms'];lo,hi=stats['paired_block_bootstrap_percentiles_95'];med=stats['median'];ax.errorbar(ci,med,yerr=[[med-lo],[hi-med]],fmt='D',ms=5,color='#111111',capsize=4,label='Median; bootstrap percentiles' if ci==0 else None)
            labels.append(c['image'].replace('_DSC','')+'\n'+c['split'])
        ax.axhline(1,color='#555555',linestyle='--',linewidth=1);ax.set(xticks=range(4),xticklabels=labels,ylabel='Guard / oracle paired API wall ratio',title='A  No observed median acceleration');ax.legend(loc='lower right',fontsize=7.8)
        ax=axes[1];styles=[('piecewise',-.23,'#777777','s','Historical piecewise*'),('oracle',0.,'#D55E00','^','Double-window oracle'),('guard',.23,'#0072B2','o','Guard')]
        for v,offset,color,marker,label in styles:
            for ci in range(4):
                y=[r['pair_wall_ms'] for r in rows if r['camera']==ci and r['variant']==v];assert all(t>0 for t in y)
                ax.scatter(ci+offset+rng.uniform(-.035,.035,len(y)),y,s=15,c=color,marker=marker,alpha=.6,label=label if ci==0 else None);ax.hlines(np.median(y),ci+offset-.07,ci+offset+.07,color=color,linewidth=2)
        ax.set(yscale='log',xticks=range(4),xticklabels=labels,ylabel='Forward + backward API wall time (ms, log10)',title='B  All 216 measured call pairs');ax.legend(loc='lower right',fontsize=7.8)
        fig.suptitle('Fixed stump7000 model, one unlocked-clock device session',fontsize=11)
        fig.supxlabel('*Historical piecewise has different numerics. Bootstrap intervals describe this session; blocks may be dependent.',fontsize=8)
        for fmt in ['png','pdf']:fig.savefig(OUT/('cost_comparison.'+fmt),dpi=300)
        plt.close(fig)
    meta=dict(script_sha256=sha(Path(__file__)),results_sha256=sha(OUT/'results.json'),analysis_sha256=sha(OUT/'analysis.json'),matplotlib=matplotlib.__version__,numpy=np.__version__,dimensions_inches=[11.2,4.4],png_dpi=300,
        output_sha256={fmt:sha(OUT/('cost_comparison.'+fmt)) for fmt in ['png','pdf']},raw_n=216,ratios_per_camera=18,estimator='Paired ratio median; 95% within-session percentile bootstrap, 20000 resamples',jitter_seed=202610208,exclusions=[],
        limitations='Provisional internal research figure; publisher requirements pending. No scene/seed population uncertainty, training latency, GT quality or equivalent historical-forward claim.',
        alt_text='Four views show guard/oracle paired median ratios above one. Raw API call pairs show guard and oracle far slower than the historical different-numerics piecewise module. All samples are plotted; device clocks were not locked.')
    target.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(dict(png=str(OUT/'cost_comparison.png'),pdf=str(OUT/'cost_comparison.pdf'))),flush=True)
if __name__=='__main__':main()
