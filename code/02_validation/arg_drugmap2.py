# -*- coding: utf-8 -*-
import os, pandas as pd, numpy as np
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATASET=os.path.join(REPO,'dataset')   # raw KPDC data (not included in repo)
argdir=[e.path for e in os.scandir(DATASET) if '2608' in e.name][0]
df=pd.read_csv(os.path.join(argdir,"ARG.abund.table.txt"),sep="\t")
df['present']=df['counts']>0
df['layer']=np.where(df['Depth.cm']<=40,'active','permafrost')
nsamp=df.SampleID.nunique()
na=df[df.layer=='active'].SampleID.nunique(); npf=df[df.layer=='permafrost'].SampleID.nunique()
print("total samples %d = active layer %d + permafrost %d"%(nsamp,na,npf))

def prev(mask,label):
    sub=df[mask & df.present]
    ov=sub.SampleID.nunique()
    a=sub[sub.layer=='active'].SampleID.nunique(); p=sub[sub.layer=='permafrost'].SampleID.nunique()
    print(f"  {label:34s}: overall {ov}/{nsamp}({100*ov/nsamp:3.0f}%) | active {100*a/na:3.0f}% | permafrost {100*p/npf:3.0f}%")

print("\n=== high-risk clinical resistance: sample detection rate (overall/active/permafrost) ===")
prev(df.geneID.eq('blaOXA'),"blaOXA (OXA beta-lactamase)")
prev(df['gene.description'].str.contains('metallo',case=False,na=False),"metallo-beta-lactamase (carbapenem)")
prev(df.Class.eq('BETA-LACTAM'),"beta-lactam class (any)")
prev(df.geneID.eq('vanA'),"vanA (high-level vancomycin)")
prev(df.Class.eq('GLYCOPEPTIDE'),"glycopeptide (vancomycin) class")
prev(df.geneID.eq('arr'),"arr (rifampin)")
prev(df.Class.eq('AMINOGLYCOSIDE'),"aminoglycoside")
prev(df.Class.eq('MACROLIDE'),"macrolide")
print("\nheavy metals (co-selection):")
prev(df.Class.eq('MERCURY'),"mercury (mer)"); prev(df.Class.eq('ARSENIC'),"arsenic (ars)")

# actual metallo gene list
mg=sorted(set(df[df['gene.description'].str.contains('metallo',case=False,na=False)].geneID))
print("\nmetallo-beta-lactamase genes:", mg)
ox=df[df.geneID.eq('blaOXA')]['gene.description'].iloc[0]
print("blaOXA description:", ox[:70])
