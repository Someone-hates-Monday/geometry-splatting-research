"""Four known views, every timing sample, plus the batching memory tradeoff."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_packed_query_v25';TIM=OUT/'timing'
a=json.loads((TIM/'independent_analysis.json').read_text());raw=json.loads((TIM/'results.json').read_text());assert a['all_verified']
arms=['A_full_score','B_cached_image','C_sequential_packed','D_batch_packed'];names=['A: full scene + loss','B: cached image + loss','C: sequential packed','D: batch packed']
colors=['#0072B2','#D55E00','#CC79A7','#009E73'];hatches=['','//','..','xx'];x=np.arange(4);width=.18
fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
for j,(arm,name,color,hatch) in enumerate(zip(arms,names,colors,hatches)):
    xx=x+(j-1.5)*width
    for ax,field,scale in [(axes[0],'wall_ms',1),(axes[1],'peak_increment_bytes',2**20)]:
        ax.bar(xx,[r['methods'][arm][field]/scale for r in a['rows']],width=width*.95,color=color,hatch=hatch,edgecolor='#333333',linewidth=.5,label=name,alpha=.85)
        for i,row in enumerate(raw['rows']):
            ax.scatter(xx[i]+np.linspace(-.025,.025,8),[b['measurements'][arm][field]/scale for b in row['blocks']],s=10,color='black',alpha=.4,zorder=3)
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
labels=[r['scene']+'\n'+('fresh' if r['view']=='fresh' else 'old test') for r in a['rows']]
for ax in axes:ax.set_xticks(x,labels)
axes[0].set_ylabel('Whole 8-score wall time (ms)');axes[1].set_ylabel('Additional peak allocation (MiB)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncols=4,fontsize=8,
    title='Complete candidate-score endpoints; same FP64 oracle, eight prepared proposals',title_fontsize=10)
fig.supxlabel('Bars: method medians; points: all 8 blocks per view (horizontal offsets for visibility).\nWarm baseline; child forward, dynamic ROI CPU/GPU layout, replay and loss included. Model/GT/proposal preparation excluded.\nAdditional allocation excludes resident cache/models/proposals. Four known views; no training-speed or population claim.',fontsize=8)
fig.savefig(OUT/'packed_pipeline_cost.png',dpi=180);fig.savefig(OUT/'packed_pipeline_cost.pdf');plt.close(fig)
write_new(OUT/'figure_provenance.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in [TIM/'independent_analysis.json',TIM/'results.json',ROOT/'plot_tgs_packed_query_v25.py']},
    raw_endpoint_samples=128,plotted_metrics=['wall_ms','peak_increment_bytes'],medians=True,no_intervals=True,
    palette='Okabe-Ito-derived plus hatching and positional labels',publisher_requirements='unspecified; provisional research figure',
    alt_text='Four-arm whole-batch score time and extra allocation across four known views. Batch packed is faster in all measured blocks, while its extra allocation is larger than sequential packed.'))
