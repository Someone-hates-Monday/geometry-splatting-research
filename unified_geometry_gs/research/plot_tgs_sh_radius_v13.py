"""Post-hoc descriptive visualization; all four prespecified displacement levels retained."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_tgs_sh_real_fields_v13 import ROOT,OUT,sha
DEST=OUT.parent/'radius_stress'

def main():
    target=DEST/'distance_transfer.png';assert not target.exists()
    main=json.loads((OUT/'results.json').read_text());stress=json.loads((DEST/'results.json').read_text())
    assert json.loads((OUT/'independent_audit.json').read_text())['all_verified']
    assert json.loads((DEST/'independent_audit.json').read_text())['all_raw_errors_verified']
    rows=[]
    for source,k in [(main,1),(stress,None)]:
        for r in source['rows']:
            rho=r['relative_displacement'] if k==1 else r['fit_a']
            factor=1 if k==1 else r['radius_factor']
            m={v:r['metrics'][('copy' if v=='copy' else v+'_fit32')]['raw_RMS'] for v in ['copy','L3','L4']} if k==1 else r['raw_RMS']
            for method in ['L3','L4']:
                rows.append(dict(scene=r['scene'],axis=r['axis'],primitive=r['primitive'],rho=rho,
                    radius_factor=factor,method=method,copy_RMS=m['copy'],method_RMS=m[method],
                    ratio=m[method]/m['copy'] if m['copy']>1e-14 else None))
    colors=['#000000','#0072B2','#009E73','#A34D00'];markers=['o','s','^','D']
    summary=[]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axs=plt.subplots(2,2,figsize=(10,6.5),sharex=True,sharey=True)
        fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.84,wspace=.12,hspace=.24)
        for si,scene in enumerate(['flowers','room']):
            for mi,method in enumerate(['L3','L4']):
                ax=axs[si,mi]
                for ri,rho in enumerate([.01,.05,.1,.25]):
                    ys=[]
                    for k in [.5,1,2,4]:
                        data=[r['ratio'] for r in rows if r['scene']==scene and r['method']==method and r['rho']==rho and r['radius_factor']==k and r['ratio'] is not None]
                        value=float(np.median(data));ys.append(value)
                        summary.append(dict(scene=scene,method=method,rho=rho,radius_factor=k,median_ratio=value,
                            retained_ratio_cases=len(data),all_cases=192))
                    ax.plot([.5,1,2,4],ys,marker=markers[ri],color=colors[ri],ls=['-','--','-.',':'][ri],label=f'a={rho:g}',lw=1.3,ms=4)
                ax.axhline(1,color='#666666',ls='--',lw=.8);ax.axvline(1,color='#999999',ls=':',lw=.8)
                ax.set_xscale('log',base=2);ax.set_xticks([.5,1,2,4],[.5,1,2,4]);ax.set_ylim(0,3.35)
                ax.set_title(f'{scene} | frozen degree {method[1:]} conversion',loc='left',fontsize=10)
                ax.grid(alpha=.12)
                if mi==0:ax.set_ylabel('RMS error / copying RMS')
                if si==1:ax.set_xlabel('Camera radius / fitting radius')
        handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False)
        fig.suptitle('Fixed-shell SH conversion loses its advantage across distances',y=.99,fontsize=12)
        fig.text(.5,.04,'Medians over 64 sampled primitives x 3 axes per scene; repeated cases, not independent scenes.\nAll four displacement levels retained. Copy RMS <= 1e-14 has no ratio; raw cases remain in JSON/CSV.\nConstructed camera shells using real public SH coefficients. No image, geometry, training or efficiency claim.',ha='center',va='bottom',fontsize=8)
        fig.savefig(target,dpi=300,facecolor='white');fig.savefig(DEST/'distance_transfer.pdf',facecolor='white');plt.close(fig)
    for name,data in [('distance_transfer_cases.csv',rows),('distance_transfer_summary.csv',summary)]:
        with (DEST/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (DEST/'figure_metadata.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'results.json',DEST/'results.json',OUT/'independent_audit.json',DEST/'independent_audit.json']},
        files={n:sha(DEST/n) for n in ['distance_transfer.png','distance_transfer.pdf','distance_transfer_cases.csv','distance_transfer_summary.csv']},
        size_inches=[10,6.5],dpi=300,posthoc_descriptive=True,
        alt_text='Degree3 and degree4 fixed-shell projections improve color approximation at the fitted radius, but both become worse than copying at four times that radius on these real coefficient cases. All four displacement levels shown; synthetic cameras, no image claims.'),indent=2)+'\n')
    print(json.dumps(dict(figure=str(target),all_displacement_levels_retained=True)),flush=True)

if __name__=='__main__':main()
