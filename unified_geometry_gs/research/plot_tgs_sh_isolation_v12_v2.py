"""Deterministic fixed-weight source effects; no inferential error bars or pooled score."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tgs_sh_isolation_common_v12 import OUT,sha
def main():
    target=OUT/'source_effects_v2.png';assert not target.exists();audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['all_verified'] and audit['all_isolation_gates'];quality=json.loads((OUT/'quality_results.json').read_text());rows=[]
    for c in quality['contrasts']:rows.append(dict(scene=c['scene'],cohort=c['cohort'],camera=c['camera'],SH_only_minus_original_PSNR=c['contrasts']['sh_only_minus_original']['primary_PNG8']['PSNR_dB'],SH_only_minus_original_SSIM=c['contrasts']['sh_only_minus_original']['primary_PNG8']['SSIM'],coarse_minus_SH_only_PSNR=c['contrasts']['coarse_minus_sh_only']['primary_PNG8']['PSNR_dB']))
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42}):
        fig,(a,b)=plt.subplots(1,2,figsize=(10,4.8),sharey=True);fig.subplots_adjust(left=.28,right=.98,top=.77,bottom=.29,wspace=.28);labels=[]
        for i,r in enumerate(rows):
            labels.append(r['scene']+' / '+r['camera']+' ('+('fresh' if r['cohort']=='fresh' else 'old')+')');marker='s' if r['cohort']=='fresh' else 'o';color='#0072B2' if r['cohort']=='fresh' else '#444444';x=r['SH_only_minus_original_PSNR'];z=r['coarse_minus_SH_only_PSNR'];a.plot(x,i,marker=marker,color=color,ms=6,ls='none');b.plot(z,i,marker=marker,color=color,ms=6,ls='none');a.annotate(f'{x:+.6f}',(x,i),xytext=(0,-13),textcoords='offset points',ha='center',fontsize=8);b.annotate(f'{z:+.6f}',(z,i),xytext=(0,-13),textcoords='offset points',ha='center',fontsize=8)
        a.set_yticks(range(4),labels);a.invert_yaxis();a.set_xlim(-2.4,.35);b.set_xlim(-.0057,.002);a.set_ylim(3.5,-.5)
        for ax in [a,b]:ax.axvline(0,color='#777777',ls='--',lw=1);ax.grid(axis='x',alpha=.15);ax.set_xlabel('PNG8 PSNR difference (dB)')
        a.set_title('A  SH-only - archived native\nIsolated source intervention',loc='left',fontsize=10);b.set_title('B  coarse - SH-only\nSubsequent combined changes',loc='left',fontsize=10);fig.suptitle('v12 zero-step compatibility | four fixed inputs, two scenes',y=.96,fontsize=12)
        fig.text(.5,.055,'Circles: old post-hoc views. Squares: prespecified fresh views. Different x-axis scales; all four inputs retained.\nNon-color outputs and degree0/precomputed-color controls matched literally. No confidence intervals or training-effect claim.\nSource: quality_results.json and independent_audit.json; PNG8 convention and full-image metrics fixed before scoring.',ha='center',va='bottom',fontsize=8);fig.savefig(target,dpi=300,facecolor='white');fig.savefig(OUT/'source_effects_v2.pdf',facecolor='white');plt.close(fig)
    with (OUT/'source_effects_v2.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'figure_metadata_v2.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),quality_sha256=sha(OUT/'quality_results.json'),audit_sha256=sha(OUT/'independent_audit.json'),files={p:sha(OUT/p) for p in ['source_effects_v2.png','source_effects_v2.pdf','source_effects_v2.csv']},size_inches=[10,4.8],dpi=300,alt_text='All four isolated SH-constructor source changes reduce frozen-weight PSNR, with larger losses in room. Two new views confirm the direction; later coarse changes are much smaller. Values are deterministic case diagnostics, not training or population effects.',no_journal_compliance_claim=True),indent=2)+'\n');print(json.dumps(dict(figure=str(target))),flush=True)
if __name__=='__main__':main()
