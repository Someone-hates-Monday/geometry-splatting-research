"""Show all64 samples, including51 exact zeros; distinguish development pixels."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tgs_split_color_common_v14 import ROOT
OUT=ROOT/'experiments/tgs_interleaving_v17'
a=json.loads((OUT/'validation/independent_audit_v2.json').read_text());rows=a['rows']
d=json.loads((OUT/'development_audit.json').read_text())['rows']
fig,ax=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for stopped,color,label in [(False,'#0072B2','No early stop'),(True,'#D55E00','Early stop')]:
    selected=[r for r in rows if any(r['early_stop'])==stopped]
    ax[0].scatter([r['actual_peak'] for r in selected],[r['bound_peak'] for r in selected],s=32,color=color,alpha=.65,label=f'{label} (n={len(selected)})')
ax[0].scatter([r['actual_peak'] for r in d],[r['bound_peak'] for r in d],marker='s',s=55,facecolors='none',edgecolors='#009E73',label='Development pixels (n=3)')
x=np.r_[0,np.geomspace(1e-7,.009,200)];ax[0].plot(x,x,'k--',lw=1,label='Bound = observed change')
ax[0].set_xscale('symlog',linthresh=5e-7);ax[0].set_yscale('symlog',linthresh=5e-7)
ax[0].set_xticks([0,1e-5,1e-4,1e-3,1e-2]);ax[0].set_yticks([0,1e-5,1e-4,1e-3,1e-2])
ax[0].set_xlabel('Observed max-channel RGB change');ax[0].set_ylabel('Conditional max-channel upper bound')
ax[0].set_title('All samples retained; zero points may overlap');ax[0].legend(fontsize=8)
nonzero=[r for r in rows if r['bound_over_actual'] is not None]
ratios=np.array([r['bound_over_actual'] for r in nonzero]);colors=['#D55E00' if any(r['early_stop']) else '#0072B2' for r in nonzero]
ax[1].bar(np.arange(len(nonzero)),ratios-1,bottom=1,color=colors);ax[1].axhline(1,color='black',ls='--',lw=1)
ax[1].set_yscale('log');ax[1].set_ylabel('Upper bound / observed RGB change');ax[1].set_xlabel('13 nonzero samples in frozen input order')
ax[1].set_title('Nonzero only: median 5.27, maximum 50.70')
for axy in ax:axy.grid(axis='y',alpha=.2)
fig.suptitle('Conditional fixed-cache subdivision bound: coverage passes, tightness remains limited',fontsize=11)
fig.savefig(OUT/'bound_coverage.png',dpi=180);fig.savefig(OUT/'bound_coverage.pdf');plt.close(fig)
