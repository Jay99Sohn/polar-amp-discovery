# -*- coding: utf-8 -*-
"""Figure 6: Arctic permafrost resistome drug-threat map (English labels; avoids Korean fonts)."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATASET=os.path.join(REPO,'dataset')   # raw KPDC data (not included in repo)
argdir=[e.path for e in os.scandir(DATASET) if '2608' in e.name][0]
df=pd.read_csv(os.path.join(argdir,"ARG.abund.table.txt"),sep="\t")
df['present']=df['counts']>0
df['layer']=np.where(df['Depth.cm']<=40,'active','permafrost')
nsamp=df.SampleID.nunique(); na=df[df.layer=='active'].SampleID.nunique(); npf=df[df.layer=='permafrost'].SampleID.nunique()

def samp_pct(mask, layer=None):
    d=df[mask & df.present]
    if layer: d=d[d.layer==layer]; tot=(na if layer=='active' else npf)
    else: tot=nsamp
    return 100*d.SampleID.nunique()/tot

# class -> (label, representative drug, tier)  tier: crit/key/other
CLASSES=[
 ('BETA-LACTAM','β-lactam (carbapenem)','crit'),
 ('GLYCOPEPTIDE','Glycopeptide (vancomycin)','crit'),
 ('RIFAMYCIN','Rifamycin (rifampin)','key'),
 ('AMINOGLYCOSIDE','Aminoglycoside (gentamicin)','key'),
 ('MACROLIDE','Macrolide (azithromycin)','key'),
 ('FOSFOMYCIN','Fosfomycin','other'),
 ('PHENICOL','Phenicol (chloramphenicol)','other'),
 ('TRIMETHOPRIM','Trimethoprim','other'),
 ('STREPTOGRAMIN','Streptogramin','other'),
 ('TETRACYCLINE','Tetracycline','other'),
]
rows=[]
for cls,lab,tier in CLASSES:
    rows.append((lab,tier,samp_pct(df.Class.eq(cls))))
rows.sort(key=lambda x:x[2])
labs=[r[0] for r in rows]; tiers=[r[1] for r in rows]; vals=[r[2] for r in rows]
col={'crit':'#b2182b','key':'#ef8a62','other':'#bdbdbd'}
cols=[col[t] for t in tiers]

plt.rcParams.update({'font.size':11,'axes.edgecolor':'#555'})
fig=plt.figure(figsize=(11,8.2))
gs=fig.add_gridspec(2,2,height_ratios=[1.35,1],width_ratios=[1.25,1],hspace=0.42,wspace=0.32)

# Panel A: detection rate by class
axA=fig.add_subplot(gs[0,:])
y=np.arange(len(labs))
axA.barh(y,vals,color=cols,edgecolor='white')
axA.set_yticks(y); axA.set_yticklabels(labs)
axA.set_xlabel('% of samples with resistance detected (n=69)')
axA.set_xlim(0,108)
for i,v in enumerate(vals): axA.text(v+1.5,i,f'{v:.0f}%',va='center',fontsize=10)
axA.set_title('A. Antibiotic-resistance detected in Arctic permafrost resistome',fontweight='bold',loc='left')
# WHO annotations
for i,lab in enumerate(labs):
    if 'carbapenem' in lab: axA.text(vals[i]+8,i,'WHO CRITICAL (CRE)',va='center',color='#b2182b',fontsize=8.5,style='italic')
    if 'vancomycin' in lab: axA.text(vals[i]+8,i,'WHO HIGH (VRE); vanA in permafrost',va='center',color='#b2182b',fontsize=8.5,style='italic')
axA.legend(handles=[Patch(color='#b2182b',label='Last-resort / WHO priority'),Patch(color='#ef8a62',label='Key clinical drug'),Patch(color='#bdbdbd',label='Other')],
           loc='lower right',fontsize=9,frameon=False)

# Panel B: active vs permafrost for key resistances
axB=fig.add_subplot(gs[1,0])
keys=[('blaOXA',df.geneID.eq('blaOXA')),('vanA',df.geneID.eq('vanA')),('Macrolide',df.Class.eq('MACROLIDE'))]
act=[samp_pct(m,'active') for _,m in keys]; per=[samp_pct(m,'permafrost') for _,m in keys]
x=np.arange(len(keys)); w=0.38
axB.bar(x-w/2,act,w,label=f'Active layer (n={na})',color='#92c5de',edgecolor='white')
axB.bar(x+w/2,per,w,label=f'Permafrost (n={npf})',color='#2166ac',edgecolor='white')
axB.set_xticks(x); axB.set_xticklabels([k for k,_ in keys]); axB.set_ylabel('% samples'); axB.set_ylim(0,110)
axB.set_title('B. Depth: thaw-locked reservoir',fontweight='bold',loc='left',fontsize=11)
axB.legend(fontsize=8.5,frameon=False)
for i,(a,p) in enumerate(zip(act,per)):
    axB.text(i-w/2,a+2,f'{a:.0f}',ha='center',fontsize=8); axB.text(i+w/2,p+2,f'{p:.0f}',ha='center',fontsize=8)

# Panel C: still effective + caveat (text box)
axC=fig.add_subplot(gs[1,1]); axC.axis('off')
axC.set_title('C. Still effective (no resistance detected)',fontweight='bold',loc='left',fontsize=11)
axC.text(0.02,0.80,'• Colistin  (no mcr genes)',fontsize=10.5,color='#1a7d1a',transform=axC.transAxes)
axC.text(0.02,0.66,'• Fluoroquinolones  (no qnr genes)',fontsize=10.5,color='#1a7d1a',transform=axC.transAxes)
axC.text(0.02,0.52,'• Newer β-lactams (cefiderocol …)',fontsize=10.5,color='#1a7d1a',transform=axC.transAxes)
axC.text(0.02,0.30,'Co-selection driver:\nHg & As resistance in 100% of samples',fontsize=9.5,color='#555',transform=axC.transAxes)
axC.text(0.02,0.06,'Caveat: derived 83-gene panel; environmental\nhazard, not proven clinical transfer.',fontsize=8,style='italic',color='#888',transform=axC.transAxes)

fig.suptitle('Figure 6.  Arctic permafrost resistome → clinical drug-threat map (KOPRI-KPDC-00002608)',fontsize=12.5,fontweight='bold',y=0.99)

# save to figures folder
outdir=os.path.join(REPO,'figures'); os.makedirs(outdir,exist_ok=True)
outp=os.path.join(outdir,'fig6_drug_threat_map.png')
fig.savefig(outp,dpi=200,bbox_inches='tight')
print("saved:",outp)
print("na,npf=",na,npf,"| vals:",[f'{l}:{v:.0f}' for l,v in zip(labs,vals)])
