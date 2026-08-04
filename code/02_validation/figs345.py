# -*- coding: utf-8 -*-
"""Figures 3 (FPR) · 4 (functional annotation) · 5 (novelty + cysteine). English labels. Audit-confirmed numbers."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
outdir=os.path.join(REPO,'figures'); os.makedirs(outdir,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.edgecolor':'#555'})

# ================= Figure 3: FPR / base rate =================
thr=['≥0.7','≥0.8','≥0.9']
fpr=[2.88,0.84,0.24]           # AMPlify negatives FPR (%)
vir=[2.08,0.44,0.036]          # polar virome pass rate (%)
tpr=[70.7,58.2,41.1]           # sensitivity for reference
x=np.arange(3); w=0.36
fig,ax=plt.subplots(figsize=(8,5.2))
b1=ax.bar(x-w/2,fpr,w,label='False-positive rate (AMPlify non-AMPs)',color='#9e9e9e',edgecolor='white')
b2=ax.bar(x+w/2,vir,w,label='Polar virome pass rate',color='#b2182b',edgecolor='white')
ax.set_yscale('log'); ax.set_ylim(0.02,10)
ax.set_xticks(x); ax.set_xticklabels(thr); ax.set_xlabel('AMP-score threshold')
ax.set_ylabel('% of sequences passing (log scale)')
for i,(f,v) in enumerate(zip(fpr,vir)):
    ax.text(i-w/2,f*1.1,f'{f:.2f}',ha='center',fontsize=9)
    ax.text(i+w/2,v*1.1,f'{v:.2f}',ha='center',fontsize=9,color='#b2182b')
ax.set_title('Figure 3.  Polar virome shows no AMP signal above the noise floor',fontweight='bold',fontsize=12,loc='left')
ax.text(0.5,-0.20,'At every threshold the virome pass rate is at or BELOW the classifier\'s false-positive rate\n→ AMP-like sequences are not enriched above noise.  (Sensitivity for reference: 71 / 58 / 41%)',
        transform=ax.transAxes,ha='center',fontsize=9,color='#444')
ax.legend(fontsize=9,frameon=False,loc='upper right')
fig.savefig(os.path.join(outdir,'fig3_fpr_baserate.png'),dpi=200,bbox_inches='tight'); plt.close(fig)

# ================= Figure 4: functional annotation =================
cats=['Known AMP','Known non-AMP protein','No significant hit']
vals=[0,34,27]; cols=['#4daf4a','#b2182b','#bdbdbd']
fig,ax=plt.subplots(figsize=(8.4,5.2))
y=np.arange(3)[::-1]
ax.barh(y,vals,color=cols,edgecolor='white')
ax.set_yticks(y); ax.set_yticklabels(cats)
ax.set_xlabel('Number of priority candidates (of 61)'); ax.set_xlim(0,46)
for yy,v in zip(y,vals):
    pct=100*v/61
    ax.text(v+0.7,yy,f'{v}  ({pct:.0f}%)',va='center',fontsize=10)
ax.text(0.5,0,'  ← 0 candidates match any known antimicrobial peptide',va='center',fontsize=9.5,color='#1a7d1a',transform=ax.get_yaxis_transform())
ax.set_title('Figure 4.  Priority candidates vs Swiss-Prot (575,503 proteins):\n                 none are known AMPs; most are known non-AMP proteins',fontweight='bold',fontsize=11.5,loc='left')
ax.text(0.98,0.42,'Representative hits (non-AMP):\n• EF-Tu  (85% id, e=1e-17)\n• DnaJ chaperone (Zn-finger)\n• DNA-binding protein HU\n• DNA primase / RNA polymerase\n= housekeeping & nucleic-acid-\n  binding proteins',
        transform=ax.transAxes,ha='right',va='center',fontsize=8.8,color='#333',
        bbox=dict(boxstyle='round',fc='#fff4f4',ec='#b2182b',alpha=0.9))
fig.savefig(os.path.join(outdir,'fig4_functional_annotation.png'),dpi=200,bbox_inches='tight'); plt.close(fig)

# ================= Figure 5: novelty + cysteine artifact =================
fig,(axA,axB)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'wspace':0.30})
# 5A novelty collapse
labA=['Author metric\n(matches / query len)','Standard tool\nvs DRAMP','Standard tool\nvs DRAMP + AMPSphere']
valA=[100,97,48]; colA=['#bdbdbd','#f4a582','#b2182b']
xa=np.arange(3)
axA.bar(xa,valA,color=colA,edgecolor='white')
axA.set_xticks(xa); axA.set_xticklabels(labA,fontsize=9); axA.set_ylim(0,112); axA.set_ylabel('% of 61 priority candidates called "novel"')
for i,v in enumerate(valA): axA.text(i,v+2,f'{v}%',ha='center',fontsize=10.5,fontweight='bold')
axA.set_title('A. Novelty collapses under standard tools + expanded DB',fontweight='bold',fontsize=11,loc='left')
# 5B cysteine artifact
labB=['Polar virome\n(low-tox subset)','Known real AMPs\n(non-polar, low-tox)','Random subsample\n(no gate)']
valB=[57,57,46]; colB=['#2166ac','#67a9cf','#bdbdbd']
xb=np.arange(3)
axB.bar(xb,valB,color=colB,edgecolor='white')
axB.set_xticks(xb); axB.set_xticklabels(labB,fontsize=9); axB.set_ylim(0,80); axB.set_ylabel('% cysteine-containing')
for i,v in enumerate(valB): axB.text(i,v+1.5,f'{v}%',ha='center',fontsize=10.5,fontweight='bold')
axB.axhline(46,ls='--',color='#888',lw=1)
axB.set_title('B. Cysteine enrichment = filter effect, not polar-specific',fontweight='bold',fontsize=11,loc='left')
axB.text(0.5,-0.26,'Same low-toxicity gate → same ~57% cysteine whether polar or non-polar known AMPs;\nrandom subsampling shows no enrichment.  Driver: GRAVY↔hemolysis r=+0.82.',
         transform=axB.transAxes,ha='center',fontsize=8.7,color='#444')
fig.suptitle('Figure 5.  Novelty is inflated (A) and the "cysteine chemical space" is a filter artifact (B)',fontsize=12,fontweight='bold',y=1.02)
fig.savefig(os.path.join(outdir,'fig5_novelty_cysteine.png'),dpi=200,bbox_inches='tight'); plt.close(fig)

print("saved:")
for f in ['fig3_fpr_baserate.png','fig4_functional_annotation.png','fig5_novelty_cysteine.png']:
    p=os.path.join(outdir,f); print("  ",f, os.path.exists(p), f"{os.path.getsize(p)//1024}KB")
