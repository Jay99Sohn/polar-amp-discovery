# -*- coding: utf-8 -*-
"""AMP RandomForest hyperparameter tuning (evidence-based grid).
Protocol: search with 5-fold CV (ROC-AUC) on the training set only -> evaluate the test set once at the end.
Directly compare against the CV of the defaults (current final model) to check 'does tuning actually improve?'.
de-bias (strip leading M) to match our pipeline exactly.
"""
import os, time, numpy as np, pandas as pd
np.random.seed(42)
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=os.path.join(BASE,"data"); OUT=os.path.join(BASE,"outputs")
KD=dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2); AAS="ACDEFGHIKLMNPQRSTVWY"
def rf(p):
    s=[];c=""
    for ln in open(p):
        ln=ln.strip()
        if ln.startswith(">"):
            if c:s.append(c);c=""
        elif ln:c+=ln
    if c:s.append(c)
    return s
def stripM(s): return s[1:] if s.startswith("M") and len(s)>1 else s
def feats(seq):
    seq=[a for a in stripM(seq) if a in KD]; L=len(seq)
    if L<5: return None
    pos=sum(seq.count(a) for a in "KR"); neg=sum(seq.count(a) for a in "DE")
    return dict(length=L,charge=pos-neg,charge_per_res=(pos-neg)/L,gravy=np.mean([KD[a] for a in seq]),
        frac_cationic=pos/L,frac_hydrophobic=sum(seq.count(a) for a in "AILMFVW")/L,
        aromaticity=sum(seq.count(a) for a in "FWY")/L,**{f"f_{a}":seq.count(a)/L for a in AAS})
def build(split):
    a=rf(os.path.join(D,f"AMPlify_AMP_{split}_common.fa")); n=rf(os.path.join(D,f"AMPlify_non_AMP_{split}_balanced.fa"))
    seqs=a+n; y=np.array([1]*len(a)+[0]*len(n)); rows=[feats(s) for s in seqs]; keep=[i for i,r in enumerate(rows) if r]
    return pd.DataFrame([rows[i] for i in keep]), y[keep]
Xtr,ytr=build("train"); Xte,yte=build("test")
print("train",Xtr.shape,"test",Xte.shape, flush=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score
cv=StratifiedKFold(5,shuffle=True,random_state=42)

# --- baseline: CV of the current default model ---
base=RandomForestClassifier(n_estimators=500,min_samples_leaf=2,random_state=42,n_jobs=1)
base_cv=cross_val_score(base,Xtr,ytr,cv=cv,scoring="roc_auc",n_jobs=-1)
print("defaults (500/leaf2/sqrt) CV-AUC = %.4f ± %.4f"%(base_cv.mean(),base_cv.std()), flush=True)

# --- evidence-based grid ---
grid={"n_estimators":[400,800],"max_depth":[None,16],"min_samples_leaf":[1,2,4],"max_features":["sqrt","log2",0.5]}
t0=time.time()
gs=GridSearchCV(RandomForestClassifier(random_state=42,n_jobs=1),grid,scoring="roc_auc",cv=cv,n_jobs=-1,verbose=0)
gs.fit(Xtr,ytr)
print("\ntuning done (%.0fs, %d combos x5-fold)"%(time.time()-t0,len(gs.cv_results_["params"])), flush=True)
print("best params:",gs.best_params_, flush=True)
print("best CV-AUC = %.4f"%gs.best_score_, flush=True)
print("improvement vs defaults = %+.4f"%(gs.best_score_-base_cv.mean()), flush=True)

# --- final test once (defaults vs tuned) ---
base.fit(Xtr,ytr); pb=base.predict_proba(Xte)[:,1]
best=gs.best_estimator_; pt=best.predict_proba(Xte)[:,1]
print("\n=== independent test set (once) ===", flush=True)
print("defaults : test AUC %.4f  F1 %.4f"%(roc_auc_score(yte,pb),f1_score(yte,(pb>=.5).astype(int))), flush=True)
print("tuned    : test AUC %.4f  F1 %.4f"%(roc_auc_score(yte,pt),f1_score(yte,(pt>=.5).astype(int))), flush=True)

# save top combos
res=pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")[["params","mean_test_score","std_test_score","rank_test_score"]].head(8)
res.to_csv(os.path.join(OUT,"amp_tuning_results.csv"),index=False,encoding="utf-8-sig")
print("\ntop combos:\n"+res.to_string(index=False), flush=True)
print("saved amp_tuning_results.csv ; DONE-TUNE", flush=True)
