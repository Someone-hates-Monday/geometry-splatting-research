"""Within-run matched-scope attribution with all measured matched blocks visible."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tgs_split_color_common_v14 import ROOT
OUT=ROOT/'experiments/tgs_aux_ablation_v21';a=json.loads((OUT/'timing/independent_analysis.json').read_text());raw=json.loads((OUT/'timing/results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
for ax,scene in zip(axes,['flowers','room']):
    for appearance,marker in [('hold','o'),('moved','D')]:
        rows=[r for r in a['rows'] if r['scene']==scene and r['appearance']==appearance]
        x=[100*r['coverage'] for r in rows]
        ax.plot(x,[r['ratios']['original_full_over_original_local'] for r in rows],ls='--',marker=marker,color='#0072B2',markerfacecolor='white',label=f'Original comparison, {appearance}')
        ax.plot(x,[r['ratios']['matched_full_over_local'] for r in rows],ls='-',marker=marker,color='#009E73',markerfacecolor='#009E73' if appearance=='hold' else 'white',label=f'Matched arithmetic, {appearance}')
        for row in rows:
            q=next(z for z in raw['rows'] if z['scene']==scene and z['appearance']==appearance and z['parent_index']==row['parent_index'])
            ax.scatter(np.full(8,100*row['coverage']),[b['ratios']['matched_full_over_local'] for b in q['blocks']],s=11,color='#009E73',alpha=.25)
    ax.axhline(1,color='black',ls=':',lw=1);ax.set_xscale('log');ax.set_ylim(bottom=.8);ax.set_title(scene+'; same16-case study, eight blocks/case' if scene=='flowers' else scene+'; matched full vs cached query')
    ax.set_xlabel('Affected image area (%)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
axes[0].set_ylabel('Case median within-block full / local wall latency');axes[1].set_ylabel('Case median within-block full / local wall latency')
fig.suptitle('Auxiliary arithmetic explains part of prior speedup; API allocations remain in all arms',fontsize=11)
fig.savefig(OUT/'aux_attribution.png',dpi=180);fig.savefig(OUT/'aux_attribution.pdf');plt.close(fig)
