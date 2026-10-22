"""Primary moved-color contrasts, no independent-scene inference or hidden cases."""
import csv,io,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_view_transfer_v23'
a=json.loads((OUT/'analysis.json').read_text());assert a['all_verified']
rows=[r for r in a['rows'] if r['appearance']=='moved'];assert len(rows)==16
metrics=['PNG8_MSE','L1_DSSIM_form'];labels=['PNG8 MSE change','0.8 raw L1 + 0.2 DSSIM change']
fig,axes=plt.subplots(2,2,figsize=(11.5,7.4),layout='constrained')
for col,scene in enumerate(['flowers','room']):
    fresh=[x for x in rows if x['scene']==scene and x['view']=='fresh'];old=[x for x in rows if x['scene']==scene and x['view']=='old_0'];x=np.arange(4)
    for row,(metric,label) in enumerate(zip(metrics,labels)):
        ax=axes[row,col];f=[r['changes'][metric] for r in fresh];o=[r['changes'][metric] for r in old]
        for j in range(4):ax.plot([j-.12,j+.12],[f[j],o[j]],color='#888888',lw=.8,zorder=1)
        ax.scatter(x-.12,f,color='#0072B2',marker='o',s=40,label='Fresh view',zorder=2)
        ax.scatter(x+.12,o,color='#D55E00',marker='s',s=36,label='Old test view',zorder=2)
        ax.axhline(0,color='black',lw=.8);ax.set_yscale('symlog',linthresh=1e-10,base=10);ax.set_ylim(-2e-4,2e-4)
        ax.set_xticks(x,[str(r['parent_id']) for r in fresh]);ax.set_xlabel('Fixed parent ID');ax.set_ylabel(label);ax.set_title(scene)
        ax.grid(axis='y',alpha=.18)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncols=2,fontsize=9,
    title='Same edits, two known views; inherited SH reevaluated',title_fontsize=10)
fig.supxlabel('Lower is better. Signed log10 scale, linear within +/-1e-10; common limits.\nFour moved proposals per scene; all observations shown. Gray segments pair views of one edit.\nFloat64 image-objective form; no training or independent-scene generalization test.',fontsize=8)
fig.savefig(OUT/'view_and_loss_alignment.png',dpi=180);fig.savefig(OUT/'view_and_loss_alignment.pdf');plt.close(fig)
buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=['scene','view','parent_id','PNG8_MSE','L1_DSSIM_form']);writer.writeheader()
for r in rows:writer.writerow(dict(scene=r['scene'],view=r['view'],parent_id=r['parent_id'],**{m:r['changes'][m] for m in metrics}))
data=OUT/'view_and_loss_alignment_data.csv'
if data.exists():assert data.read_bytes()==buffer.getvalue().encode()
else:data.write_bytes(buffer.getvalue().encode())
provenance=dict(analysis_sha256=sha(OUT/'analysis.json'),script_sha256=sha(ROOT/'plot_tgs_view_transfer_v23.py'),
    primary_moved_only=True,all_hold_cases_in_analysis=True,observations=16,normalization='whole-image 3HW; PNG8 also divided by255^2',
    transform='base10 symlog, linear +/-1e-10, common +/-2e-4 limits',inference='descriptive; no intervals',publisher_requirements='unspecified; provisional research figure',
    alt_text='Paired fresh and old-view error changes for four fixed parent edits per scene. Several pairs cross zero; PNG8 and L1 plus DSSIM changes can also have different signs.')
target=OUT/'figure_provenance.json'
if target.exists():
    assert json.loads(target.read_text())['analysis_sha256']==provenance['analysis_sha256']
    target.write_text(json.dumps(provenance,indent=2)+'\n')
else:write_new(target,provenance)
