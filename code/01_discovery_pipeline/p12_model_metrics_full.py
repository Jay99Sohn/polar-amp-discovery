# -*- coding: utf-8 -*-
"""
p12_model_metrics_full.py
Reproduces the 5-model comparison of CELL5 (colab_full_run.py), additionally computing the
Precision / Sensitivity / Specificity added to report Table 4-1, and saves an extended model_comparison.csv.

- Training set: AMPlify train (positive/negative), 27 de-biased physicochemical features (initiator M stripped)
- CV: cv_AUC via StratifiedKFold(5)
- Independent test set: metrics at threshold 0.5 on AMPlify test
- Specificity = TN/(TN+FP) = recall(pos_label=0)
Run: python code/01_discovery_pipeline/p12_model_metrics_full.py
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef, accuracy_score,
                             precision_score, recall_score)
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
REF  = os.path.join(HERE, "..", "data")
OUT_ROOT = os.path.join(HERE, "..", "..")   # project root

KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
    'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
AAS=sorted(KD)

def readfa(p):
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
    seq=[a for a in seq if a in KD]; L=len(seq)
    if L<5: return None
    pos=sum(seq.count(a) for a in "KR"); neg=sum(seq.count(a) for a in "DE")
    return dict(length=L,charge=pos-neg,charge_per_res=(pos-neg)/L,gravy=np.mean([KD[a] for a in seq]),
        frac_cationic=pos/L,frac_hydrophobic=sum(seq.count(a) for a in "AILMFVW")/L,
        aromaticity=sum(seq.count(a) for a in "FWY")/L,**{f"f_{a}":seq.count(a)/L for a in AAS})
def build(pa,pb,ya,yb):
    A=[stripM(s) for s in readfa(pa)]; B=[stripM(s) for s in readfa(pb)]
    seqs=A+B; y=np.array([ya]*len(A)+[yb]*len(B))
    rows=[feats(s) for s in seqs]; keep=[i for i,r in enumerate(rows) if r]
    return pd.DataFrame([rows[i] for i in keep]), y[keep]

def main():
    Xa,ya=build(REF+'/AMPlify_AMP_train_common.fa', REF+'/AMPlify_non_AMP_train_balanced.fa',1,0)
    Xte,yte=build(REF+'/AMPlify_AMP_test_common.fa', REF+'/AMPlify_non_AMP_test_balanced.fa',1,0)
    print(f"train {len(ya)} / test {len(yte)} / features {Xa.shape[1]}")
    def Pl(e): return Pipeline([("sc",StandardScaler()),("m",e)])
    models={"LogReg":Pl(LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=.5,C=.5,max_iter=8000)),
     "SVM_RBF":Pl(SVC(probability=True,C=2,random_state=42)),
     "kNN":Pl(KNeighborsClassifier(n_neighbors=11)),
     "RandomForest":RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=42,n_jobs=1),
     "XGBoost":XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.1,subsample=.9,colsample_bytree=.8,eval_metric="logloss",random_state=42,n_jobs=1)}
    cv=StratifiedKFold(5,shuffle=True,random_state=42); rows=[]
    for n,e in models.items():
        r=cross_validate(e,Xa,ya,cv=cv,scoring=["roc_auc","f1"],n_jobs=1)
        e.fit(Xa,ya); p=e.predict_proba(Xte)[:,1]; yh=(p>=.5).astype(int)
        rows.append(dict(model=n,
            cv_AUC=round(r["test_roc_auc"].mean(),3), cv_AUC_sd=round(r["test_roc_auc"].std(),3),
            test_AUC=round(roc_auc_score(yte,p),3),
            test_precision=round(precision_score(yte,yh),3),
            test_sensitivity=round(recall_score(yte,yh),3),                 # = recall (pos_label=1)
            test_specificity=round(recall_score(yte,yh,pos_label=0),3),     # TN/(TN+FP)
            test_F1=round(f1_score(yte,yh),3), test_MCC=round(matthews_corrcoef(yte,yh),3),
            test_ACC=round(accuracy_score(yte,yh),3)))
        print(f"  {n} done")
    cmp=pd.DataFrame(rows).sort_values("cv_AUC",ascending=False)
    cmp.to_csv(os.path.join(OUT_ROOT,'results','model_comparison.csv'),index=False,encoding='utf-8-sig')
    print("\n"+cmp.to_string(index=False))
    print("\nsaved: results/model_comparison.csv — added precision/sensitivity/specificity columns")

if __name__=="__main__":
    main()
