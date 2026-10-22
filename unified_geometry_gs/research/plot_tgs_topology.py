"""Display the failed selector's full predefined group comparisons."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_topology_v1'


def main():
    audit=json.loads((OUT/'summary_audit.json').read_text());assert audit['passed']
    stats=audit['stats'];groups=[('general','train'),('general','test'),('stress','train'),('stress','test')]
    styles=[('red4','Original midpoint','#555555'),('world_longest4','Classical longest-edge','#0072B2'),
            ('train_maximin4','Train-view maximin','#D55E00')]
    with (OUT/'group_statistics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    with plt.rc_context({'font.size':10,'figure.facecolor':'white'}):
        fig,axes=plt.subplots(1,2,figsize=(11.2,4.5),layout='constrained')
        for j,(style,label,color) in enumerate(styles):
            values=[next(x for x in stats if (x['stratum'],x['split'],x['style'])==(*g,style)) for g in groups]
            pos=np.arange(4)+(j-1)*.23
            bars=axes[0].bar(pos,[x['all_children_culled'] for x in values],width=.21,color=color,label=label)
            axes[0].bar_label(bars,fontsize=9,padding=2)
            axes[1].plot(pos,[x['median_SH_full_image_MAE'] for x in values],linestyle='none',marker=['o','s','x'][j],
                         color=color,label=label,markersize=7)
        labels=[]
        for stratum,split in groups:
            count=next(x['parent_preprocess_survives'] for x in stats if (x['stratum'],x['split'],x['style'])==(stratum,split,'red4'))
            labels.append(f'{stratum}\n{split} (n={count})')
        for ax in axes:
            ax.set_xticks(np.arange(4),labels);ax.set_xlabel('Frozen stratum and camera split');ax.grid(axis='y',alpha=.2)
        axes[1].set_xticks(np.arange(4),[f'{a}\n{b} (16 cases)' for a,b in groups])
        axes[0].set(title='All four children culled after a surviving parent',ylabel='Event-camera count',ylim=(0,9))
        axes[0].legend(fontsize=8)
        axes[1].set(title='Median full-image RGB perturbation',ylabel='Mean absolute RGB change',ylim=(0,None))
        axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        fig.suptitle('Four-child capacity matched: the maximin selector fails its gate',fontsize=13)
        fig.supxlabel('16 new events; every rule adds 3 primitives. Left n = surviving-parent cases; right median includes all 16 cases per group. No training.',fontsize=8)
        fig.savefig(OUT/'topology_comparison.png',dpi=220,facecolor='white');plt.close(fig)
    manifest={'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'summary_audit.json',OUT/'group_statistics.csv']},
              'scope':'frozen diagnostic, no publisher target or quality claim',
              'encoding':'counts include baseline zero; medians include all 16 cases per group; no connecting lines',
              'replication':'one scene, eight events per stratum, repeated across cameras; aliases not independent execution',
              'output_sha256':hashlib.sha256((OUT/'topology_comparison.png').read_bytes()).hexdigest(),
              'matplotlib':matplotlib.__version__}
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figure':str(OUT/'topology_comparison.png'),'gate':audit['advance_selector_as_training_candidate']}))


if __name__=='__main__':main()
