"""Plot original native arrays without interpolation or image adjustment."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_subdivision_v1'


def main():
    inputs=[]
    with plt.rc_context({'font.size':10,'axes.titlesize':11,'figure.facecolor':'white'}):
        fig,axes=plt.subplots(2,3,figsize=(11.4,6.8),layout='constrained')
        for row,(name,limits) in enumerate([('large_sigma1.0',(30,211,30,181)),('inradius1.5',(110,122,110,122))]):
            paths=[OUT/(name+'_'+s+'.npz') for s in ['parent','children']]
            for p in paths:
                inputs.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            parent,child=[np.load(p)['native'][0] for p in paths]
            x0,x1,y0,y1=limits
            arrays=[parent,child,child-parent]
            titles=['Parent alpha','Four-child alpha','Children minus parent']
            for col,(array,title) in enumerate(zip(arrays,titles)):
                ax=axes[row,col]
                image=ax.imshow(array[y0:y1,x0:x1],origin='lower',interpolation='nearest',
                    extent=[x0-.5,x1-.5,y0-.5,y1-.5],
                    cmap='RdBu_r' if col==2 else 'viridis',vmin=-.5 if col==2 else 0,vmax=.5)
                ax.set(title=title,xlabel='Column (pixel)',ylabel='Row (pixel)')
                ax.set_aspect('equal')
                fig.colorbar(image,ax=ax,label='Alpha change' if col==2 else 'Alpha',shrink=.88)
            axes[row,0].text(.02,.98,'Large triangle: r = 40 px' if row==0 else 'Small triangle: r = 1.5 px',
                transform=axes[row,0].transAxes,va='top',fontsize=9,
                color='black',bbox={'facecolor':'white','edgecolor':'none','alpha':1.})
        fig.suptitle('Pinned TGS native forward: synthetic midpoint subdivision',fontsize=13)
        fig.supxlabel('One frozen scene per row; constant white appearance on black; no training or scene-quality claim',fontsize=9)
        fig.savefig(OUT/'subdivision_diagnostic.png',dpi=240,facecolor='white')
        fig.savefig(OUT/'subdivision_diagnostic.svg',facecolor='white')
        plt.close(fig)
    manifest={'audience':'research diagnostic, provisional general figure; no journal target',
              'inputs':inputs,'matplotlib':matplotlib.__version__,'numpy':np.__version__,
              'transformations':['native channel zero: white-on-black equals alpha',
                                 'explicit rectangular crops: large cols 30..210 rows 30..180; small 110..121',
                                 'signed difference children-parent; no smoothing/interpolation',
                                 'common alpha limits [0,.5], difference limits [-.5,.5]'],
              'replication':'single deterministic input per row; not independent scene replicates',
              'caption':'The large triangle redistributes alpha while its sum remains nearly constant. The small parent survives but all four children are culled.',
              'outputs':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['subdivision_diagnostic.png','subdivision_diagnostic.svg']}}
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figure':str(OUT/'subdivision_diagnostic.png'),'manifest':True}))


if __name__=='__main__':
    main()
