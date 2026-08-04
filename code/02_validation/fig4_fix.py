# -*- coding: utf-8 -*-
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
outdir=os.path.join(REPO,'figures'); os.makedirs(outdir,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.edgecolor':'#555'})

cats=['Known AMP','Known non-AMP\nprotein','No significant hit']
vals=[0,34,27]; cols=['#4daf4a','#b2182b','#bdbdbd']
fig,ax=plt.subplots(figsize=(9,5.4))
y=np.array([2,1,0])
ax.barh(y,vals,color=cols,edgecolor='white',height=0.6)
ax.set_yticks(y); ax.set_yticklabels(cats)
ax.set_xlabel('Number of priority candidates (of 61)'); ax.set_xlim(0,52); ax.set_ylim(-0.6,2.7)
# value labels
ax.text(0.7,2,'0  (0%)',va='center',fontsize=11)
ax.text(35,1,'34  (56%)',va='center',fontsize=11,fontweight='bold')
ax.text(28,0,'27  (44%)',va='center',fontsize=11)
# green callout beside Known-AMP bar
ax.text(6.5,2,'← none are known AMPs',va='center',fontsize=11,color='#1a7d1a',fontweight='bold')
# representative-hits box in top-right empty area
ax.text(0.58,0.93,'Representative hits (non-AMP):\n • EF-Tu  (85% id, e=1e-17)\n • DnaJ chaperone (Zn-finger)\n • DNA-binding protein HU\n • DNA primase / RNA polymerase\n = housekeeping & nucleic-acid-binding',
        transform=ax.transAxes,ha='left',va='top',fontsize=9.2,color='#333',
        bbox=dict(boxstyle='round',fc='#fff4f4',ec='#b2182b',alpha=0.95))
ax.set_title('Figure 4.  Priority candidates vs Swiss-Prot (575,503 proteins):\nnone are known AMPs; most are known non-AMP proteins',
             fontweight='bold',fontsize=12,loc='left')
fig.savefig(os.path.join(outdir,'fig4_functional_annotation.png'),dpi=200,bbox_inches='tight'); plt.close(fig)
print("fig4 regenerated:",os.path.join(outdir,'fig4_functional_annotation.png'))
