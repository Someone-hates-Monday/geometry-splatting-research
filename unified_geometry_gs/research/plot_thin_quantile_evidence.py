"""Provisional manuscript figure from saved fixed-ray diagnostics (not scene data)."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT/'research/figures/thin_quantile_evidence_v1'
    targets = [out.with_suffix(x) for x in ('.png','.svg','.manifest.json')]
    if any(p.exists() for p in targets):
        raise ValueError('New output names required')
    thin_path = ROOT/'research/results/thin_quantile_limit_v2.json'
    band_path = ROOT/'research/results/quantile_band_v2.json'
    thin = json.loads(thin_path.read_text(encoding='utf-8'))
    band = json.loads(band_path.read_text(encoding='utf-8'))
    assert thin['all_declared_checks_pass'] and band['all_declared_checks_pass']
    blue, orange, gray = '#0072B2', '#A35A00', '#444444'
    style = {'font.size':8.5, 'axes.titlesize':9, 'axes.labelsize':8.5,
             'legend.fontsize':7.5,'xtick.labelsize':8,'ytick.labelsize':8,
             'axes.spines.top':False,'axes.spines.right':False,
             'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2,2,figsize=(180/25.4,150/25.4),layout='constrained')
        fig.suptitle('Fixed-ray mathematical diagnostics (synthetic)',fontsize=11)
        ax=axes[0,0]
        eps=[float(x['epsilon']) for x in thin['critical_cases']]
        logs=[x['log10_abs_opacity_derivative'] for x in thin['critical_cases']]
        band_logs=[np.log10(abs(x['opacity_gradient'][1])) for x in band['critical_cases']]
        ax.plot(eps,logs,'o-',color=blue,label='Single median')
        ax.plot(eps,band_logs,'s--',color=orange,label='Band [0.45, 0.55]')
        ax.set(title='(a) Critical-family opacity sensitivity',xlabel=r'Thickness factor $\varepsilon$ (dimensionless)',
               ylabel=r'$\log_{10}|\partial\,\mathrm{depth}/\partial\alpha_2|$')
        ax.set_xlim(.52,.025); ax.legend(loc='upper left',frameon=False)
        ax.grid(axis='y',color='#DDDDDD',linewidth=.5)
        ax=axes[0,1]
        median=[float(x) for x in thin['critical_cases'][-1]['position_weights']]
        averaged=band['critical_cases'][-1]['position_gradient']
        x=np.arange(3)
        ax.bar(x-.18,median,.36,color=blue,label='Single median')
        ax.bar(x+.18,averaged,.36,color=orange,hatch='//',label='Band [0.45, 0.55]')
        ax.set_xticks(x,['Left (−1)','Middle (0)','Right (+1)'])
        ax.set(title=r'(b) Position Jacobian at $\varepsilon=0.075$',xlabel='Peak position (arbitrary depth units)',
               ylabel='Position derivative (dimensionless)',ylim=(0,1.18))
        ax.legend(loc='upper right',frameon=False)
        ax=axes[1,0]
        for widths,marker in (([1.,1.],'o'),([1.,2.],'s')):
            for interval,ls,color in (([.45,.55],'-',blue),([.4,.55],'--',orange)):
                rows=[r for r in band['plateau_cases'] if r['width_factors']==widths and r['band']==interval]
                label=f'[{interval[0]:.2f}, {interval[1]:.2f}], s={int(widths[0])}:{int(widths[1])}'
                ax.plot([r['epsilon'] for r in rows],[r['mean'] for r in rows],marker=marker,ls=ls,color=color,label=label)
        ax.axhline(0,color=gray,ls=':',lw=1)
        ax.axhline(1/3,color=gray,ls='-.',lw=1)
        ax.text(.043,0.015,'limit 0',fontsize=7,ha='right')
        ax.text(.043,1/3+.012,'limit +1/3',fontsize=7,ha='right')
        ax.set(title='(c) Bands spanning a two-layer plateau',xlabel=r'Thickness factor $\varepsilon$ (dimensionless)',
               ylabel='Band depth (arbitrary units)',ylim=(-1,1),xlim=(.52,.025))
        ax.text(.50,-.90,'Surfaces at −1 and +1',fontsize=7)
        ax.legend(loc='lower right',frameon=False,fontsize=6.5)
        ax=axes[1,1]
        for case in thin['regular_cases']:
            rows=case['checks']; j=case['selected_zero_based']
            ax.plot([float(r['epsilon']) for r in rows],[float(r['position_weights'][j]) for r in rows],
                    color=gray,lw=.8,marker='.',ms=3,alpha=.55)
        ax.set(title='(d) Strict, noncritical selected-layer weights',xlabel=r'Thickness factor $\varepsilon$ (dimensionless)',
               ylabel='Selected position derivative (dimensionless)',ylim=(0,1.05),xlim=(.42,.0125))
        ax.text(.40,.12,'All 20 prescribed-domain cases\nTwo margins ≥ 0.03; no error bars',fontsize=7)
        out.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(targets[0],dpi=300,facecolor='white')
        fig.savefig(targets[1],facecolor='white')
        plt.close(fig)
    with Image.open(targets[0]) as im:
        metadata={'pixels':list(im.size),'dpi':list(im.info['dpi']),'mode':im.mode,
                  'alpha_fully_opaque':im.getextrema()[-1]==(255,255) if im.mode=='RGBA' else True}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),
              'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (thin_path,band_path)},
              'output_sha256':{str(p.relative_to(ROOT)):sha(p) for p in targets[:2]},
              'matplotlib_version':matplotlib.__version__,'raster_metadata':metadata,
              'intended_size_mm':[180,150],'publisher_requirements':'Unspecified; provisional general scientific figure, not a compliance claim.',
              'transformations':['Panel a: log10 of absolute saved middle-opacity derivatives; all seven epsilon values.',
                                 'Panel b: last prescribed epsilon, all three position derivatives for both functions.',
                                 'Panel c: all 12 plateau/band cases; dotted reference limits from theorem; peaks at -1,+1.',
                                 'Panel d: all 20 included regular cases and all five epsilon values; nine analytic-domain exclusions retained in source JSON.'],
              'replication':'Deterministic synthetic families, not statistical replicates; no uncertainty bands or significance tests.',
              'line_semantics':'Segments connect calculated samples to guide reading; uncomputed intermediate values are not observations.',
              'caption':'Continuous median depths can converge without Jacobian convergence. A finite quantile band has bounded conditional gradients but can mix surfaces. Ideal fixed-ray model, not real-scene reconstruction or runtime evidence. Read protocol erratum for corrected +1/3 prediction.',
              'alt_text':'Four panels show an exponentially growing single-median opacity derivative versus a finite band derivative; median position weights on the two outer peaks versus band weight on the middle peak; band depths between two surfaces; and selected-layer weights approaching one in twenty regular cases.'}
    targets[2].write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'outputs':[str(p) for p in targets],'raster_metadata':metadata},indent=2))


if __name__=='__main__':
    main()
