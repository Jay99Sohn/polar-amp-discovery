# -*- coding: utf-8 -*-
"""Shared module for the review re-analysis — reproduces the author pipeline (colab_full_run.py / p1 / p5) exactly.
Same 27 features, same stripM de-bias, same RandomForest(500, seed42)."""
import os, numpy as np, pandas as pd

KD=dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,
        K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
AAS="ACDEFGHIKLMNPQRSTVWY"
EIS=dict(A=.62,R=-2.53,N=-.78,D=-.90,C=.29,Q=-.85,E=-.74,G=.48,H=-.40,I=1.38,L=1.06,
         K=-1.5,M=.64,F=1.19,P=.12,S=-.18,T=-.05,W=.81,Y=.26,V=1.08)

def readfa(p):
    s=[];c=""
    for ln in open(p, encoding="utf-8", errors="ignore"):
        ln=ln.strip()
        if ln.startswith(">"):
            if c:s.append(c);c=""
        elif ln:c+=ln
    if c:s.append(c)
    return s

def stripM(s): return s[1:] if s.startswith("M") and len(s)>1 else s

def feats(seq):
    seq=[a for a in seq if a in KD]; L=len(seq)
    if L<5: return None
    pos=sum(seq.count(a) for a in "KR"); neg=sum(seq.count(a) for a in "DE")
    return dict(length=L,charge=pos-neg,charge_per_res=(pos-neg)/L,gravy=float(np.mean([KD[a] for a in seq])),
        frac_cationic=pos/L,frac_hydrophobic=sum(seq.count(a) for a in "AILMFVW")/L,
        aromaticity=sum(seq.count(a) for a in "FWY")/L,**{f"f_{a}":seq.count(a)/L for a in AAS})

def uH(seq,delta=100.):
    import math
    seq=[a for a in seq if a in EIS]
    if not seq: return 0.
    r=math.radians(delta); sx=sum(EIS[a]*math.cos(i*r) for i,a in enumerate(seq)); sy=sum(EIS[a]*math.sin(i*r) for i,a in enumerate(seq))
    return round(math.hypot(sx,sy)/len(seq),3)

def build_from_fasta(pa,pb,ya,yb,strip=True):
    A=[stripM(s) if strip else s for s in readfa(pa)]
    B=[stripM(s) if strip else s for s in readfa(pb)]
    seqs=A+B; y=np.array([ya]*len(A)+[yb]*len(B))
    rows=[feats(s) for s in seqs]; keep=[i for i,r in enumerate(rows) if r]
    return pd.DataFrame([rows[i] for i in keep]), y[keep]

def train_models(REF):
    """Reproduce AMP (de-biased) + HemoPI-2 + HemoPI-3."""
    from sklearn.ensemble import RandomForestClassifier
    Xa,ya=build_from_fasta(REF+'/AMPlify_AMP_train_common.fa', REF+'/AMPlify_non_AMP_train_balanced.fa',1,0)
    COLS=list(Xa.columns)
    amp=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=-1).fit(Xa,ya)
    H=REF+'/hemopi'
    Xh2,yh2=build_from_fasta(H+'/HemoPI-2_model_class1.fasta', H+'/HemoPI-2_model_class0.fasta',1,0)
    hemo2=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=-1).fit(Xh2[COLS],yh2)
    Xh3,yh3=build_from_fasta(H+'/HemoPI-3_model_class1.fasta', H+'/HemoPI-3_model_class0.fasta',1,0)
    hemo3=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=-1).fit(Xh3[COLS],yh3)
    return dict(amp=amp,hemo2=hemo2,hemo3=hemo3,cols=COLS)

def score_peptides(peps, M):
    """peps: list[str] -> DataFrame(pep,amp_score,hemo2,hemo3,cys,gravy,...)"""
    rows=[]; keep=[]
    for i,p in enumerate(peps):
        f=feats(p)
        if f is not None: rows.append(f); keep.append(i)
    X=pd.DataFrame(rows)[M['cols']]
    out=pd.DataFrame({'pep':[peps[i] for i in keep]})
    out['length']=[len(peps[i]) for i in keep]
    out['amp_score']=M['amp'].predict_proba(X)[:,1]
    out['hemo2']=M['hemo2'].predict_proba(X)[:,1]
    out['hemo3']=M['hemo3'].predict_proba(X)[:,1]
    out['cys_n']=out.pep.str.count('C')
    out['has_cys']=out.cys_n>0
    out['gravy']=[r['gravy'] for r in rows]
    out['charge']=[r['charge'] for r in rows]
    return out
