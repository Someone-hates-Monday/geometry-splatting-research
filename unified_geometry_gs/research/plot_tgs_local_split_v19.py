"""Descriptive measured-pair plot; setup-inclusive panel is an explicit cost estimate."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tgs_split_color_common_v14 import ROOT
OUT=ROOT/'experiments/tgs_local_split_v19';raw=json.loads((OUT/'timing/results.json').read_text());a=json.loads((OUT/'timing/independent_analysis.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
for i,(row,summary) in enumerate(zip(raw['rows'],a['case_summaries'])):
    color='#0072B2' if row['scene']=='flowers' else '#D55E00';ys=[p['full_over_local_wall'] for p in row['pairs']]
    ax[0].scatter(i+np.linspace(-.18,.18,8),ys,s=11,color=color,alpha=.3)
    ax[0].scatter(i,summary['paired_wall_ratio'],s=39,marker='o' if row['affected_tiles'] else 's',facecolor=color if row['affected_tiles'] else 'white',edgecolor=color,zorder=3)
ax[0].axhline(1,color='black',ls='--',lw=1);ax[0].set_ylim(bottom=.8);ax[0].set_ylabel('Full / local synchronized wall latency');ax[0].set_xlabel('Four views, eight frozen events per view')
ax[0].set_xticks([3.5,11.5,19.5,27.5],['flowers test','flowers train','room test','room train']);ax[0].tick_params(axis='x',labelsize=8)
for x in [7.5,15.5,23.5]:ax[0].axvline(x,color='gray',lw=.6,alpha=.5)
ax[0].set_title('Cached prepared queries: 256 balanced pairs')
ax[0].legend(handles=[Line2D([],[],marker='o',ls='',color='#0072B2',label='Affected-region case median'),Line2D([],[],marker='s',ls='',markerfacecolor='white',color='#D55E00',label='Zero affected tiles'),Line2D([],[],marker='.',ls='',color='gray',alpha=.4,label='All eight measured pairs')],fontsize=8)
views=a['views'];x=np.arange(4);width=.35;full=np.array([v['full_8_prepared_ms'] for v in views]);setup=np.array([v['setup_wall_ms'] for v in views]);local=np.array([v['local_8_plus_observed_setup_ms']-v['setup_wall_ms'] for v in views])
ax[1].bar(x-width/2,full,width,label='Full: sum of 8 event medians',color='#0072B2')
ax[1].bar(x+width/2,local,width,label='Local: sum of 8 event medians',color='#009E73')
ax[1].bar(x+width/2,setup,width,bottom=local,label='One observed cache setup',color='#E69F00')
ax[1].set_xticks(x,['flowers test','flowers train','room test','room train']);ax[1].tick_params(axis='x',labelsize=8)
ax[1].set_ylabel('Estimated 8-query cost (ms)');ax[1].set_title('Setup-inclusive estimate; not a separately timed batch');ax[1].legend(fontsize=8)
for axis in ax:axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
fig.suptitle('Bit-identical counterfactual split queries; fixed models, no training speed claim',fontsize=12)
fig.savefig(OUT/'query_cost.png',dpi=180);fig.savefig(OUT/'query_cost.pdf');plt.close(fig)
