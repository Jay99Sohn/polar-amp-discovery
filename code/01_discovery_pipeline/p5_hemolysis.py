# -*- coding: utf-8 -*-
"""Task 2: hemolysis (toxicity) predictor from experimentally-validated peptides.
Trains on HemoPI-2 and HemoPI-3, whose NEGATIVE class is experimentally-validated
NON-hemolytic peptides (not random proteins) — so the model learns the true
hemolytic-vs-nonhemolytic boundary among peptide-like sequences. Two independently
trained models are saved; a candidate is called low-toxicity only if BOTH predict
non-hemolytic (conservative consensus). Same physicochemical features as the AMP
model; strip leading M for consistency with how virome ORF candidates are featurized.
"""
import os, numpy as np, pandas as pd, joblib
np.random.seed(42)
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H=os.path.join(BASE,"data","hemopi"); OUT=os.path.join(BASE,"outputs")
def rf(p):
    s=[];c=""
    for ln in open(p):
        ln=ln.strip()
        if ln.startswith(">"):
            if c:s.append(c);c=""
        elif ln: c+=ln
    if c:s.append(c)
    return s
def strip_M(s): return s[1:] if s.startswith("M") and len(s)>1 else s
KD=dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,
        K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
AAS="ACDEFGHIKLMNPQRSTVWY"
def feats(seq):
    seq=[a for a in seq if a in KD]; L=len(seq)
    if L<5: return None
    pos=sum(seq.count(a) for a in "KR"); neg=sum(seq.count(a) for a in "DE")
    return dict(length=L,charge=pos-neg,charge_per_res=(pos-neg)/L,gravy=np.mean([KD[a] for a in seq]),
        frac_cationic=pos/L,frac_hydrophobic=sum(seq.count(a) for a in "AILMFVW")/L,
        aromaticity=sum(seq.count(a) for a in "FWY")/L,**{f"f_{a}":seq.count(a)/L for a in AAS})
COLS=list(feats("ACDEFGHIKLMNPQRSTVWYACDEF").keys())
def load(ds,split):
    neg=[strip_M(s) for s in rf(os.path.join(H,f"{ds}_{split}_class0.fasta"))]  # non-hemolytic (experimentally validated)
    pos=[strip_M(s) for s in rf(os.path.join(H,f"{ds}_{split}_class1.fasta"))]  # hemolytic
    seqs=pos+neg; y=np.array([1]*len(pos)+[0]*len(neg))
    rows=[feats(s) for s in seqs]; keep=[i for i,r in enumerate(rows) if r]
    return pd.DataFrame([rows[i] for i in keep])[COLS], y[keep]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
for ds,fname in [("HemoPI-2","hemolysis_model.joblib"),("HemoPI-3","hemolysis_model_hemopi3.joblib")]:
    Xtr,ytr=load(ds,"model"); Xva,yva=load(ds,"validation")
    m=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=1).fit(Xtr,ytr)
    p=m.predict_proba(Xva)[:,1]
    print("%s train:%s pos(hemolytic)=%d | validation AUC=%.3f F1=%.3f"
          %(ds,Xtr.shape,ytr.sum(),roc_auc_score(yva,p),f1_score(yva,(p>=.5).astype(int))))
    joblib.dump({"model":m,"cols":COLS}, os.path.join(OUT,fname))
# sanity: known strongly-hemolytic vs known low-hemolytic AMP should separate
mods={n:joblib.load(os.path.join(OUT,f))["model"] for n,f in [("H2","hemolysis_model.joblib"),("H3","hemolysis_model_hemopi3.joblib")]}
for nm,seq in [("melittin(hemolytic)","GIGAVLKVLTTGLPALISWIKRKRQQ"),("magainin-2(low-hemolytic)","GIGKFLHSAKKFGKAFVGEIMNS")]:
    x=pd.DataFrame([feats(strip_M(seq))])[COLS]
    print("  sanity %-24s "%nm+"  ".join("%s=%.2f"%(n,mods[n].predict_proba(x)[:,1][0]) for n in mods))
print("saved hemolysis_model.joblib (HemoPI-2) + hemolysis_model_hemopi3.joblib ; DONE-HEMO")
