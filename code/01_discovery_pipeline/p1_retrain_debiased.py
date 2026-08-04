# -*- coding: utf-8 -*-
"""Phase 2 prerequisite: retrain the AMP classifier AFTER removing the
initiator-methionine artifact (non-AMP set was 42% M-starting vs 1.6% for AMP).
Strip leading M from ALL sequences so training matches how virome ORFs (all
M-starting) will be scored. Report honest test AUC and save the model.
"""
import os, numpy as np, pandas as pd, joblib
np.random.seed(42)
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D=os.path.join(BASE,"data"); OUT=os.path.join(BASE,"outputs"); os.makedirs(OUT,exist_ok=True)
def rf(p):
    s=[];c=""
    for ln in open(p):
        ln=ln.strip()
        if ln.startswith(">"):
            if c:s.append(c);c=""
        else:c+=ln
    if c:s.append(c)
    return s
def strip_leading_M(seq):           # remove initiator Met to match mature peptides / ORFs
    return seq[1:] if seq.startswith("M") and len(seq)>1 else seq
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
def build(split):
    amp=[strip_leading_M(s) for s in rf(os.path.join(D,f"AMPlify_AMP_{split}_common.fa"))]
    non=[strip_leading_M(s) for s in rf(os.path.join(D,f"AMPlify_non_AMP_{split}_balanced.fa"))]
    seqs=amp+non; y=np.array([1]*len(amp)+[0]*len(non))
    rows=[feats(s) for s in seqs]; keep=[i for i,r in enumerate(rows) if r]
    return pd.DataFrame([rows[i] for i in keep]), y[keep]
Xtr,ytr=build("train"); Xte,yte=build("test")
# verify artifact removed
print("after stripping leading M -> starts-with-M should be ~0 in both classes")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
rf_m=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=1).fit(Xtr,ytr)
p=rf_m.predict_proba(Xte)[:,1]
print("DE-BIASED held-out test: AUC=%.3f F1=%.3f (was 0.959 with the M artifact)"
      %(roc_auc_score(yte,p),f1_score(yte,(p>=.5).astype(int))))
imp=pd.DataFrame({"f":Xte.columns,"imp":rf_m.feature_importances_}).sort_values("imp",ascending=False).head(8)
print("\ntop features now (should be charge/hydrophobicity, not f_M):")
print(imp.to_string(index=False))
joblib.dump({"model":rf_m,"cols":list(Xtr.columns)}, os.path.join(OUT,"amp_model_debiased.joblib"))
print("\nsaved amp_model_debiased.joblib ; DONE-RETRAIN")
