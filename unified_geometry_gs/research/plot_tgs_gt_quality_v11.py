"""Full verified views, fixed PNG display convention, shared explicit difference scale."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_gt_quality_v11'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'quality_views.png';assert not target.exists();audit=json.loads((OUT/'independent_quality_audit.json').read_text());assert audit['all_verified'];results=json.loads((OUT/'quality_results.json').read_text());identity=json.loads((OUT/'identity_audit.json').read_text());info=[]
    style={'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42}
    with plt.rc_context(style):
        fig,axes=plt.subplots(2,5,figsize=(15.6,6.3));fig.subplots_adjust(left=.06,right=.955,bottom=.15,top=.88,wspace=.08,hspace=.19);last=None
        for i,scene in enumerate(['flowers','room']):
            cam=next(r['camera'] for r in identity['rows'] if r['scene']==scene);gt=np.load(OUT/scene/'gt_rgb.npy');axes[i,0].imshow(gt,interpolation='nearest');axes[i,0].set_title(scene+' / '+cam+'\nVerified original photograph');axes[i,0].set_ylabel(scene,fontsize=10)
            by={r['variant']:r for r in results['rows'] if r['scene']==scene}
            for j,v in enumerate(['archived_native','piecewise','coarse'],start=1):
                image=np.load(OUT/scene/v/'png_rgb.npy');axes[i,j].imshow(image,interpolation='nearest');q=by[v]['primary_PNG8'];label='coarse = native64 oracle' if v=='coarse' else v;axes[i,j].set_title(f"{label}\nPSNR {q['PSNR_dB']:.4f} dB; SSIM {q['SSIM']:.5f}")
            x=np.load(OUT/scene/'coarse/png_rgb.npy').astype(float)/255;y=np.load(OUT/scene/'archived_native/png_rgb.npy').astype(float)/255;delta=np.abs(x-y).mean(2);last=axes[i,4].imshow(delta,cmap='viridis',vmin=0,vmax=.1,interpolation='nearest');axes[i,4].set_title('|coarse - archived_native|\nMean absolute RGB difference');info.append(dict(scene=scene,max_difference=float(delta.max()),fraction_above_display_scale=float((delta>.1).mean()),full_pixels=int(delta.size)))
            for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
        cb=fig.colorbar(last,ax=axes[:,4].tolist(),fraction=.03,pad=.02);cb.set_label('RGB difference (0–1); saturated above0.10',fontsize=8);fig.suptitle('v11 verified GT: fixed7000-step weights, zero-step operator compatibility',y=.97,fontsize=12)
        fig.text(.5,.035,'All full images retained; display uses the fixed PNG8 saving convention. No crop, alignment, exposure or color adjustment.\nDifference scale is shared across both rows; values above0.10 use the top color and saturation counts are retained in metadata.\nTwo diagnostic views, not a dataset score or a retrained-method comparison. Source: quality_results.json and identity_audit.json.',ha='center',va='bottom',fontsize=8)
        fig.savefig(target,dpi=300,facecolor='white');fig.savefig(OUT/'quality_views.pdf',facecolor='white');plt.close(fig)
    meta=dict(script_sha256=sha(Path(__file__)),results_sha256=sha(OUT/'quality_results.json'),identity_sha256=sha(OUT/'identity_audit.json'),audit_sha256=sha(OUT/'independent_quality_audit.json'),files={s:sha(OUT/s) for s in ['quality_views.png','quality_views.pdf']},size_inches=[15.6,6.3],dpi=300,difference_display_range=[0,.1],rows=info,alt_text='Two verified photographs and three different renderer outputs on the same fixed weights. Coarse exactly matches native64; its quality is nearly equal to piecewise and below the archived-source control, especially room. The full-frame last column shows coarse versus archived RGB differences, with a disclosed shared scale.',destination='Provisional internal research diagnostic, no journal compliance claim.');(OUT/'figure_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(dict(figure=str(target),difference_rows=info)),flush=True)
if __name__=='__main__':main()
