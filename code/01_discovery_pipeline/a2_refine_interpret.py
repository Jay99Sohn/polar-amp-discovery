# -*- coding: utf-8 -*-
"""AMP Phase 1 refinement:
  (1) cross-validated ROC-AUC with bootstrap 95% CI on the held-out test
  (2) permutation feature importance -> which physicochemical properties drive
      antimicrobial activity (pharmacist interpretation: cationicity, amphipathicity)
  (3) ROC curve figure + a threshold table for later virome screening.
"""
import os, numpy as np, pandas as pd
np.random.seed(42)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data"); OUT = os.path.join(BASE, "outputs"); FIG=os.path.join(OUT,"figures")
os.makedirs(FIG, exist_ok=True)

def read_fasta(p):
    seqs=[]; cur=""
    for ln in open(p):
        ln=ln.strip()
        if ln.startswith(">"):
            if cur: seqs.append(cur); cur=""
        else: cur+=ln
    if cur: seqs.append(cur)
    return seqs
KD=dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,
        K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
AAS="ACDEFGHIKLMNPQRSTVWY"
def feats(seq):
    seq=[a for a in seq if a in KD]; L=len(seq)
    if L==0: return None
    pos=sum(seq.count(a) for a in "KR"); neg=sum(seq.count(a) for a in "DE")
    return dict(length=L,charge=pos-neg,charge_per_res=(pos-neg)/L,gravy=np.mean([KD[a] for a in seq]),
        frac_cationic=pos/L,frac_hydrophobic=sum(seq.count(a) for a in "AILMFVW")/L,
        aromaticity=sum(seq.count(a) for a in "FWY")/L,**{f"f_{a}":seq.count(a)/L for a in AAS})
def build(split):
    amp=read_fasta(os.path.join(D,f"AMPlify_AMP_{split}_common.fa"))
    non=read_fasta(os.path.join(D,f"AMPlify_non_AMP_{split}_balanced.fa"))
    seqs=amp+non; y=np.array([1]*len(amp)+[0]*len(non))
    rows=[feats(s) for s in seqs]; keep=[i for i,r in enumerate(rows) if r]
    return pd.DataFrame([rows[i] for i in keep]), y[keep]
Xtr,ytr=build("train"); Xte,yte=build("test")

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rf=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=1).fit(Xtr,ytr)
p=rf.predict_proba(Xte)[:,1]; obs=roc_auc_score(yte,p)
rng=np.random.default_rng(42); bs=[]
yt=np.array(yte)
for _ in range(2000):
    idx=rng.integers(0,len(yt),len(yt))
    if len(np.unique(yt[idx]))<2: continue
    bs.append(roc_auc_score(yt[idx],p[idx]))
lo,hi=np.percentile(bs,[2.5,97.5])
print("Held-out test ROC-AUC = %.3f  (95%% CI %.3f-%.3f)"%(obs,lo,hi))

perm=permutation_importance(rf,Xte,yte,n_repeats=20,random_state=42,n_jobs=1,scoring="roc_auc")
imp=pd.DataFrame({"feature":Xte.columns,"importance":perm.importances_mean}).sort_values("importance",ascending=False)
print("\nTop properties driving antimicrobial activity (permutation importance):")
print(imp.head(10).to_string(index=False))
imp.to_csv(os.path.join(OUT,"amp_feature_importance.csv"),index=False,encoding="utf-8-sig")

# mean property in AMP vs non-AMP (interpretation direction)
Xtr2=Xtr.copy(); Xtr2["y"]=ytr
comp=Xtr2.groupby("y")[["charge","frac_cationic","gravy","aromaticity","length"]].mean().T
comp.columns=["non_AMP","AMP"]; print("\nProperty means (train):\n",comp.round(3).to_string())

fpr,tpr,_=roc_curve(yte,p)
plt.figure(figsize=(5.5,5))
plt.plot(fpr,tpr,label=f"RandomForest (AUC={obs:.3f})",lw=2)
plt.plot([0,1],[0,1],"k--",lw=1,label="chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("AMP classifier — held-out test (1,670 peptides)"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG,"amp_roc.png"),dpi=130); plt.close()
print("\nsaved amp_roc.png, amp_feature_importance.csv")

# probability-threshold table for later virome screening
print("\nScore-threshold guide (on test, for future candidate ranking):")
for th in [0.5,0.7,0.9,0.95]:
    sel=p>=th
    prec=yte[sel].mean() if sel.sum() else float("nan")
    print("  score>=%.2f : precision=%.3f, flags %d/%d test peptides"%(th,prec,sel.sum(),len(p)))
print("DONE-AMP2")
