"""All 32 actual-camera diagnostics, zero values retained; no inferential error bars."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tgs_split_color_common_v14 import OUT,sha

def main():
    target=OUT/'split_components.png';assert not target.exists()
    audit=json.loads((OUT/'independent_audit_v2.json').read_text());assert audit['all_verified']
    result=json.loads((OUT/'results.json').read_text());rows=[]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axes=plt.subplots(2,1,figsize=(11,7.8),sharey=True)
        fig.subplots_adjust(left=.10,right=.98,top=.86,bottom=.22,hspace=.65)
        for ax,scene in zip(axes,['flowers','room']):
            cases=[r for r in result['cases'] if r['scene']==scene];x=np.arange(16)
            for shift,name,color,marker,label in [(-.14,'noncolor','#0072B2','s','G: all non-color changes'),(0,'total','#222222','x','D: total change'),(.14,'appearance','#A34D00','D','A: color-direction change')]:
                ax.scatter(x+shift,[r['metrics'][name]['RMS'] for r in cases],c=color,marker=marker,s=24,label=label,zorder=3)
            ax.set_xticks(x,[f"{r['event']}/"+('test' if r['camera']==0 else 'train') for r in cases],rotation=55,ha='right')
            ax.set_yscale('symlog',linthresh=1e-9,linscale=.6);ax.set_ylim(-1e-10,5e-4)
            ax.set_yticks([0,1e-9,1e-7,1e-5,1e-3],['0','1e-9','1e-7','1e-5','1e-3'])
            ax.set_ylabel('Full-image RGB RMS\n(symlog; linear below 1e-9)');ax.grid(axis='y',alpha=.2)
            zero=sum(r['metrics']['noncolor']['RMS']<=1e-14 for r in cases)
            ax.set_title(f'{scene} | 8 offline midpoint splits x 2 captured cameras | {zero} zero/tiny G cases retained',loc='left',fontsize=10)
            for r in cases:
                rows.append(dict(scene=scene,event=r['event'],camera=r['camera_name'],split='test' if r['camera']==0 else 'train',
                    parent_id=r['parent_id'],A_RMS=r['metrics']['appearance']['RMS'],G_RMS=r['metrics']['noncolor']['RMS'],
                    D_RMS=r['metrics']['total']['RMS'],A_over_G_RMS=r['metrics']['A_over_G_RMS'],
                    signed_energy_cross_term=r['metrics']['signed_energy_cross_term']))
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
        fig.suptitle('TGS split perturbations: color is smaller than the remaining geometry/rendering changes',y=.99,fontsize=12)
        fig.text(.5,.045,'All 32 event-camera cases retained. Zero is displayed at zero; no pixel/scene confidence intervals.\nG includes window, opacity, support, culling and sorting effects; these mechanisms are not separated here.\nPhysical-center SH colors on frozen public7000 models; offline simulations, no training or GT-quality comparison.\nSource: results.json and independent_audit_v2.json. X labels: event index / camera split.',ha='center',va='bottom',fontsize=8)
        fig.savefig(target,dpi=300,facecolor='white');fig.savefig(OUT/'split_components.pdf',facecolor='white');plt.close(fig)
    with (OUT/'split_components.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'figure_metadata.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),
        results_sha256=sha(OUT/'results.json'),audit_sha256=sha(OUT/'independent_audit_v2.json'),
        files={n:sha(OUT/n) for n in ['split_components.png','split_components.pdf','split_components.csv']},
        size_inches=[11,7.8],dpi=300,all_cases_retained=True,
        alt_text='All 32 offline split-camera cases on flowers and room. Color-direction perturbation is smaller than non-color perturbation in every case with nonzero changes. Thirteen zero/tiny non-color cases remain at zero. No training or image-quality benefit claim.'),indent=2)+'\n')
    print(json.dumps(dict(figure=str(target))),flush=True)

if __name__=='__main__':main()
