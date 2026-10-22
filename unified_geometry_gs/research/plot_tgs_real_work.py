"""All-ray empirical survival curves, no latency or population-inference claim."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_real_work_v7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=json.loads((OUT/'independent_audit.json').read_text());assert a['all_gates'];assert not (OUT/'K_survival.png').exists()
    colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00'];styles=['-','--','-.',':','-'];markers=['o','s','^','D','v'];raw=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.5,3.9),layout='constrained')
    for ci,r in enumerate(a['rows']):
        hist=np.load(OUT/f'camera_{ci}_K_histogram.npy').astype(np.int64);ks=np.arange(len(hist));tail=hist.sum()-np.cumsum(hist);survival=tail/hist.sum();mask=tail>0
        label=r['name'].replace('_DSC','')+(' (new camera)' if r['new_holdout'] else '')
        ax.plot(ks[mask],100*survival[mask],color=colors[ci],ls=styles[ci],lw=1.6,label=label,marker=markers[ci],markevery=25,ms=3.3)
        raw.extend(dict(camera=r['name'],K=int(k),pixel_count=int(hist[k]),pixels_over_K=int(tail[k]),fraction_over_K=float(survival[k])) for k in ks)
    ax.axvline(128,color='#555555',lw=1,ls='--');ax.text(130,75,'capacity 128',rotation=90,va='top',ha='left',fontsize=8,color='#444444')
    ax.set_yscale('log');ax.set_ylim(0.00008,110);ax.set_xlim(0,220);ax.set_xlabel('Capacity K (included contributions per ray)');ax.set_ylabel('Pixels with more than K contributions (%)');ax.grid(axis='y',alpha=.2)
    ax.legend(loc='lower left',frameon=False,ncol=2,fontsize=8);ax.set_title('Actual oracle traversal: five cameras, one fixed checkpoint',fontsize=10)
    fig.get_layout_engine().set(rect=(0,.09,1,.91));fig.text(.5,.015,'All pixels retained, including K=0; counts do not establish runtime or a universal capacity bound.',ha='center',fontsize=8)
    fig.savefig(OUT/'K_survival.png',dpi=300,metadata={'Title':'Actual TGS ray contribution-count survival'});fig.savefig(OUT/'K_survival.pdf');plt.close(fig)
    with (OUT/'K_histograms.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    (OUT/'figure_metadata.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),audit_sha256=sha(OUT/'independent_audit.json'),pixels_per_camera=1027125,all_zero_pixels_retained=True,quantities='included accepted contributions, not sorted ordinals',log_y=True,zero_survival_points_not_plotted_but_retained_in_CSV=True,size_inches=[6.5,3.9],dpi=300,figure_files={p:sha(OUT/p) for p in ['K_survival.png','K_survival.pdf','K_histograms.csv']}),indent=2)+'\n')
    print(json.dumps(dict(figure=str(OUT/'K_survival.png'),rows=len(raw))),flush=True)
if __name__=='__main__':main()
