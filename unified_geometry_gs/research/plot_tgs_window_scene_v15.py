"""All32 window cases plus ALL3 mechanism pixels; no extrapolation to quality/population."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tgs_split_color_common_v14 import ROOT,sha
OUT=ROOT/'experiments/tgs_window_scene_v15'

def main():
    target=OUT/'window_scene.png';assert not target.exists()
    a=json.loads((OUT/'independent_audit.json').read_text());t=json.loads((OUT/'trace/independent_audit.json').read_text())
    assert a['all_saved_arrays_and_controls_verified'] and t['all_verified']
    rows=a['rows'];tr=t['rows']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,(left,right)=plt.subplots(1,2,figsize=(11,5.7));fig.subplots_adjust(left=.09,right=.98,bottom=.29,top=.80,wspace=.38)
        for scene,color,marker in [('flowers','#0072B2','s'),('room','#A34D00','o')]:
            rs=[r for r in rows if r['scene']==scene]
            left.scatter([r['isolated_alpha_RMS'] for r in rs],[r['full_scene_G_RMS'] for r in rs],
                c=color,marker=marker,s=32,alpha=.85,label=scene)
        violated=[r for r in rows if r['fixed_slot_bound_violating_pixels']>0]
        left.scatter([r['isolated_alpha_RMS'] for r in violated],[r['full_scene_G_RMS'] for r in violated],
            facecolors='none',edgecolors='#222222',marker='o',s=95,lw=1.2,label='Conditional-bound violation case')
        left.set_xscale('symlog',linthresh=1e-9);left.set_yscale('symlog',linthresh=1e-9)
        left.set_xlim(-1e-10,None);left.set_ylim(-1e-10,None);left.grid(alpha=.15)
        left.set_xlabel('Isolated white-alpha RMS |Q-P|');left.set_ylabel('Full-scene RGB RMS |G|')
        left.set_title('A  All32 cases, same captured cameras',loc='left',fontsize=10)
        left.annotate('13 zero cases overlap here',(0,0),xytext=(15,12),textcoords='offset points',fontsize=8)
        left.legend(loc='upper left',fontsize=7,frameon=False)
        x=np.arange(3)
        right.scatter(x-.08,[r['native_perturbation_peak'] for r in tr],marker='s',color='#0072B2',s=40,label='Native changed order')
        right.scatter(x+.08,[r['fixed_group_perturbation_peak'] for r in tr],marker='D',color='#A34D00',s=35,label='Children at parent slot')
        right.set_yscale('log');right.set_xticks(x,[f"event{r['event']}\n({r['pixel'][0]},{r['pixel'][1]})" for r in tr])
        right.set_ylabel('RGB perturbation peak relative to before');right.set_title('B  All3 frozen mechanism pixels',loc='left',fontsize=10)
        right.grid(axis='y',alpha=.15);right.legend(fontsize=8,frameon=False,loc='upper left')
        fig.suptitle('TGS subdivision: window redistribution and a local ordering mechanism',y=.97,fontsize=12)
        fig.text(.5,.045,'A: symlog axes, linear below 1e-9; ratio also includes color contrast and transmittance, not an occlusion fraction.\nB: same cached alphas/colors; 6 native endpoints reproduced literally. No early stop on these3 pixels.\nSame v14 cases reused post-hoc; no independent scene statistics, GT quality, training or efficiency claim.\nSources: independent_audit.json and trace/independent_audit.json; all raw cases preserved in CSV/NPZ.',ha='center',va='bottom',fontsize=8)
        fig.savefig(target,dpi=300,facecolor='white');fig.savefig(OUT/'window_scene.pdf',facecolor='white');plt.close(fig)
    flat=[{k:v for k,v in r.items() if k!='fixed_slot_Cmax'} for r in rows]
    with (OUT/'window_scene_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    flat_trace=[{k:v for k,v in r.items() if k not in ['exact_three_channel_decomposition','pixel']}|dict(x=r['pixel'][0],y=r['pixel'][1]) for r in tr]
    with (OUT/'window_scene_trace.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat_trace[0]));w.writeheader();w.writerows(flat_trace)
    (OUT/'figure_metadata.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),
        audit_sha256=sha(OUT/'independent_audit.json'),trace_audit_sha256=sha(OUT/'trace/independent_audit.json'),
        files={n:sha(OUT/n) for n in ['window_scene.png','window_scene.pdf','window_scene_cases.csv','window_scene_trace.csv']},
        size_inches=[11,5.7],dpi=300,all_cases_retained=True,
        alt_text='All32 cases show isolated alpha redistribution versus full-scene noncolor RGB changes;13 zero cases retained. All3 frozen bound-violation pixels show much smaller perturbation when changed children are moved to the parent ordering slot. No early stop, training or quality claim.'),indent=2)+'\n')
    print(json.dumps(dict(figure=str(target))),flush=True)

if __name__=='__main__':main()
