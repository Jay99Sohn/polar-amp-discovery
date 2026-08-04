# -*- coding: utf-8 -*-
"""AMP discovery Phase 1: train an antimicrobial-peptide classifier on the
AMPlify benchmark (pre-split, balanced). KEY HONESTY CHECK:
the full descriptor model MUST beat a trivial length+charge baseline on the
held-out test set — otherwise it adds nothing.
"""
import os, numpy as np, pandas as pd
np.random.seed(42)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data"); OUT = os.path.join(BASE, "outputs"); os.makedirs(OUT, exist_ok=True)

def read_fasta(path):
    seqs=[]; cur=""
    for ln in open(path):
        ln=ln.strip()
        if ln.startswith(">"):
            if cur: seqs.append(cur); cur=""
        else: cur+=ln
    if cur: seqs.append(cur)
    return seqs

def load(split):
    amp = read_fasta(os.path.join(D, f"AMPlify_AMP_{split}_common.fa"))
    non = read_fasta(os.path.join(D, f"AMPlify_non_AMP_{split}_balanced.fa"))
    seqs = amp+non; y = np.array([1]*len(amp)+[0]*len(non))
    return seqs, y

# ---- physicochemical descriptors (pharmacist-interpretable, no deps) ----
KD = dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,
          L=3.8,K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
AAS="ACDEFGHIKLMNPQRSTVWY"
def feats(seq):
    seq=[a for a in seq if a in KD]
    L=len(seq)
    if L==0: return None
    pos=sum(seq.count(a) for a in "KR"); neg=sum(seq.count(a) for a in "DE")
    charge=pos-neg
    gravy=np.mean([KD[a] for a in seq])
    arom=sum(seq.count(a) for a in "FWY")/L
    comp=[seq.count(a)/L for a in AAS]
    return dict(length=L, charge=charge, charge_per_res=charge/L, gravy=gravy,
                frac_cationic=pos/L, frac_hydrophobic=sum(seq.count(a) for a in "AILMFVW")/L,
                aromaticity=arom, **{f"f_{a}":c for a,c in zip(AAS,comp)})

def build(split):
    seqs,y=load(split)
    rows=[feats(s) for s in seqs]
    keep=[i for i,r in enumerate(rows) if r is not None]
    X=pd.DataFrame([rows[i] for i in keep]); yy=y[keep]
    return X, yy

Xtr,ytr=build("train"); Xte,yte=build("test")
print("train:",Xtr.shape,"pos=%d neg=%d"%(ytr.sum(),(ytr==0).sum()))
print("test :",Xte.shape,"pos=%d neg=%d"%(yte.sum(),(yte==0).sum()))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier

def ev(model, cols, name):
    model.fit(Xtr[cols], ytr); p=model.predict_proba(Xte[cols])[:,1]
    return dict(model=name, n_feat=len(cols), test_AUC=round(roc_auc_score(yte,p),3),
               test_F1=round(f1_score(yte,(p>=.5).astype(int)),3),
               test_ACC=round(accuracy_score(yte,(p>=.5).astype(int)),3))

trivial=["length","charge"]           # <-- the baseline the model MUST beat
allf=list(Xtr.columns)
res=[
  ev(Pipeline([("s",StandardScaler()),("l",LogisticRegression(max_iter=5000))]), trivial, "TRIVIAL length+charge (LR)"),
  ev(Pipeline([("s",StandardScaler()),("l",LogisticRegression(max_iter=5000))]), allf, "full descriptors (LR)"),
  ev(RandomForestClassifier(n_estimators=400,random_state=42,n_jobs=1), allf, "full descriptors (RF)"),
  ev(XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,subsample=0.9,
       colsample_bytree=0.8,eval_metric="logloss",random_state=42,n_jobs=1), allf, "full descriptors (XGB)"),
]
R=pd.DataFrame(res); print("\n=== held-out TEST performance ==="); print(R.to_string(index=False))
R.to_csv(os.path.join(OUT,"amp_phase1_metrics.csv"),index=False,encoding="utf-8-sig")
gain=R.loc[R.model.str.contains("XGB"),"test_AUC"].values[0]-R.loc[R.model.str.contains("TRIVIAL"),"test_AUC"].values[0]
print("\nAUC gain over trivial baseline: +%.3f  ->  %s"%(gain,"NON-TRIVIAL (model adds real value)" if gain>0.03 else "WEAK"))
print("DONE-AMP1")
