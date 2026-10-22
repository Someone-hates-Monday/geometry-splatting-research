"""Stress cases and setup-inclusive estimates; no fitted crossover or dropped outcomes."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tgs_split_color_common_v14 import ROOT
OUT=ROOT/'experiments/tgs_local_stress_v20';a=json.loads((OUT/'timing/independent_analysis.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
for scene,color in [('flowers','#0072B2'),('room','#D55E00')]:
    for appearance,marker in [('hold','o'),('moved','D')]:
        rows=[r for r in a['rows'] if r['scene']==scene and r['appearance']==appearance]
        ax[0].plot([100*r['coverage'] for r in rows],[r['paired_wall_ratio'] for r in rows],marker=marker,color=color,
            ls='-' if appearance=='hold' else '--',ms=6,markerfacecolor=color if appearance=='hold' else 'white',label=f'{scene}, {appearance}')
ax[0].axhline(1,color='black',ls=':',lw=1);ax[0].set_xscale('log');ax[0].set_xlabel('Affected image area (%)');ax[0].set_ylabel('Case median paired full / local wall latency')
ax[0].set_title('Fixed query code; all16 stress cases retained');ax[0].legend(fontsize=8)
views=a['views'];x=np.arange(2);width=.35;full=np.array([v['full_8_sum_medians_ms'] for v in views]);setup=np.array([v['setup_wall_ms'] for v in views]);local=np.array([v['local_8_sum_medians_plus_setup_ms']-v['setup_wall_ms'] for v in views])
ax[1].bar(x-width/2,full,width,color='#0072B2',label='Full: sum of8 event medians')
ax[1].bar(x+width/2,local,width,color='#009E73',label='Local: sum of8 event medians')
ax[1].bar(x+width/2,setup,width,bottom=local,color='#E69F00',label='One observed cache setup')
ax[1].set_xticks(x,['flowers fresh','room fresh']);ax[1].set_ylabel('Estimated8-query cost (ms)');ax[1].set_title('Setup may remove the short-batch advantage');ax[1].legend(fontsize=8)
for axis in ax:axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
fig.suptitle('Coverage reaches23%; bit-identical RGB/alpha/paths, no training or quality claim',fontsize=11)
fig.savefig(OUT/'stress_query_cost.png',dpi=180);fig.savefig(OUT/'stress_query_cost.pdf');plt.close(fig)
