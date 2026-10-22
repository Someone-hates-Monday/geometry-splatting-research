"""Transparent scientific figures from all saved real-checkpoint cases."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_real_event_v1'


def main():
    results=json.loads((OUT/'results.json').read_text())
    audit=json.loads((OUT/'summary_audit.json').read_text());assert audit['audit_passed']
    rows=results['cases'];assert len(rows)==32
    modes=[('original','Original native SH','#0072B2','o'),
           ('fixed','Corrected native SH','#D55E00','x'),('white','Constant white','#555555','^')]
    with plt.rc_context({'font.size':10,'figure.facecolor':'white'}):
        fig,axes=plt.subplots(1,3,figsize=(13.4,4.1),layout='constrained')
        for mode,label,color,marker in modes:
            data=[r['metrics'][mode] for r in rows]
            for ax,key in zip(axes,['rgb_max_abs','rgb_pixels_above_tolerance','rgb_mean_abs']):
                ax.plot(np.arange(32),[x[key] for x in data],color=color,marker=marker,
                        markersize=3.5,linestyle='none',clip_on=False,label=label)
        labels=['Peak absolute RGB change','Pixels above 3e-5 RGB change','Full-image mean absolute RGB change']
        units=['RGB channel value','Pixel count per case','RGB channel value']
        for ax,title,unit in zip(axes,labels,units):
            ax.set(title=title,xlabel='Event (four camera cases per event)',ylabel=unit,ylim=(0,None),xlim=(-.5,31.5))
            ax.set_xticks(np.arange(8)*4+1.5,[str(i) for i in range(8)])
            ax.grid(axis='y',alpha=.2)
        axes[0].legend(fontsize=8)
        axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        fig.suptitle('Stump checkpoint 7000: all 32 frozen split simulations',fontsize=13)
        fig.supxlabel('8 events, 2 test + 2 train cameras; zero changes retained. Perturbation metrics, not training-quality comparisons.',fontsize=9)
        fig.savefig(OUT/'all_event_effects.png',dpi=200,facecolor='white');plt.close(fig)
        # The largest observed peak is explicitly a posthoc illustration;
        # no sample or result is excluded from the all-case figure above.
        example=max(rows,key=lambda r:r['metrics']['original']['rgb_max_abs'])
        ei,ci=example['event_index'],example['camera']
        raw=np.load(OUT/f'event_{ei}_camera_{ci}_original.npz')
        baseline=np.load(OUT/f'baseline_{ci}.npz')
        x0,y0,x1,y1=raw['bbox'];extent=[x0-.5,x1-.5,y1-.5,y0-.5]
        fig,axes=plt.subplots(2,2,figsize=(9.2,7),layout='constrained')
        axes[0,0].imshow(np.clip(baseline['original_rgb'].transpose(1,2,0),0,1),interpolation='nearest')
        axes[0,0].add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='#D55E00',linewidth=1.2))
        axes[0,0].set_title('Frozen real model, original native SH')
        for ax,array,title in [(axes[0,1],raw['before_rgb'],'Before split: crop'),
                               (axes[1,0],raw['after_rgb'],'After split: same crop')]:
            ax.imshow(np.clip(array.transpose(1,2,0),0,1),extent=extent,interpolation='nearest')
            ax.set_title(title)
        change=np.abs(raw['after_rgb']-raw['before_rgb']).max(0)
        image=axes[1,1].imshow(change,extent=extent,interpolation='nearest',cmap='magma',vmin=0,vmax=.05)
        axes[1,1].set_title('Maximum absolute channel change')
        fig.colorbar(image,ax=axes[1,1],label='Absolute RGB change')
        for ax in axes.flat:ax.set(xlabel='Column (pixel)',ylabel='Row (pixel)')
        fig.suptitle(f'Posthoc illustration: largest peak, event {ei}, camera {ci}',fontsize=12)
        fig.supxlabel('RGB display clipped to [0,1]; numeric differences use unmodified floats. All 32 cases reported separately.',fontsize=8)
        fig.savefig(OUT/'real_event_example.png',dpi=200,facecolor='white');plt.close(fig)
    inputs=[OUT/'results.json',OUT/'summary_audit.json',OUT/f'event_{ei}_camera_{ci}_original.npz',OUT/f'baseline_{ci}.npz']
    manifest={'audience':'research diagnostic; no publisher target','matplotlib':matplotlib.__version__,
              'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'all_case_inclusions':'all 32, including 13 zero changes',
              'example_rule':'posthoc largest original RGB peak; illustrative only',
              'transformations':['no smoothing/interpolation','RGB display clipping [0,1] only',
                                  'maximum absolute channel difference, common [0,.05] scale',
                                  'global MAE denominator 3*1245*825; no ROI normalization'],
              'outputs':{s:hashlib.sha256((OUT/s).read_bytes()).hexdigest() for s in ['all_event_effects.png','real_event_example.png']}}
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figures':list(manifest['outputs']),'example_event':ei,'camera':ci}))


if __name__=='__main__':
    main()
