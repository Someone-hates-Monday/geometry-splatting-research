"""All four score endpoints and every measured block; no scene-population CI."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tgs_split_color_common_v14 import ROOT

OUT=ROOT/'experiments/tgs_batch_score_v22'
a=json.loads((OUT/'timing/independent_analysis.json').read_text())
r=json.loads((OUT/'timing/results_recovered.json').read_text())
arms=['A_full_score','B_cached_image_score','C_sequential_scalar','D_batch_scalar']
labels=['A: full render + score','B: cached image + score','C: sequential scalar','D: batched scalar']
colors=['#0072B2','#D55E00','#CC79A7','#009E73']
fig,axes=plt.subplots(1,2,figsize=(11.6,4.9),layout='constrained')
for ax,scene in zip(axes,['flowers','room']):
    rows=[x for x in a['rows'] if x['scene']==scene]
    for j,(arm,label,color) in enumerate(zip(arms,labels,colors)):
        ax.plot([x['K'] for x in rows],[x['methods'][arm]['wall_ms'] for x in rows],marker='o',color=color,label=label,lw=1.6)
        for row in rows:
            cell=next(x for x in r['rows'] if x['scene']==scene and x['K']==row['K'])
            ax.scatter(np.full(8,row['K'])+(j-1.5)*.045,[b['measurements'][arm]['wall_ms'] for b in cell['blocks']],s=12,color=color,alpha=.3,edgecolors='none')
    ax.set_title(scene);ax.set_xticks([1,2,4,8]);ax.set_ylim(bottom=0);ax.set_xlabel('Prepared proposal prefix size K');ax.set_ylabel('Whole K-score endpoint wall time (ms)')
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Fixed-state PNG8 SSE queries: medians and all 8 blocks per cell',fontsize=11)
fig.supxlabel('Warm cache; child forward, ROI scan and reduction included. Prefix coverage mix changes with K.\nOriginal timer exit 1 at final call-count check; all eight raw cells recovered, no timing rerun.',fontsize=8)
fig.savefig(OUT/'batch_score_cost.png',dpi=180);fig.savefig(OUT/'batch_score_cost.pdf');plt.close(fig)
