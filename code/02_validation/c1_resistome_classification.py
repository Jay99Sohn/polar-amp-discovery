# -*- coding: utf-8 -*-
"""C1 (background) CRITICAL SELF-CHECK — is the active/permafrost classification trivial or overfit?
Q1. Does the 83-gene MODEL beat a TRIVIAL baseline (total ARG load / richness)?
    If total-abundance-only does as well, the ML adds nothing -> the result is trivial.
Q2. Is depth just correlated with total load? (Spearman)
Q3. Threshold honesty: report ALL thresholds, not just the best.
Q4. Leakage sanity: shuffle labels -> AUC should collapse to ~0.5.

Leakage-free GroupKFold (groups = drilling cores). Verified on the KPDC ARG data
(KOPRI-KPDC-00002608, active layer <= 40cm): the total-load baseline (logistic, GroupKFold)
AUC ~0.88 matches/exceeds the full 83-gene model (RF) AUC ~0.79 -> the gene composition adds
nothing over total load, so the depth split is trivial (report §4.4). Raw KPDC data is not
bundled; place it under `dataset/` to run."""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.random.seed(42)
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score

REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATASET=os.path.join(REPO,'dataset')   # raw KPDC data (not included in repo)
argdir=[e.path for e in os.scandir(DATASET) if '2608' in e.name][0]
df=pd.read_csv(os.path.join(argdir,"ARG.abund.table.txt"),sep="\t")
df["genekey"]=df["geneID"].astype(str)+" | "+df["gene.description"].astype(str)
mat=df.pivot_table(index="SampleID",columns="genekey",values="normalized.value",aggfunc="sum",fill_value=0.0)
meta=df.groupby("SampleID").agg(Depth_cm=("Depth.cm","first")).loc[mat.index]
X=mat.values
groups=meta.index.str.split("_").str[0].values      # drilling core = leading token of SampleID
gkf=GroupKFold(n_splits=5)

total=X.sum(1)                      # total ARG load per sample
rich=(X>0).sum(1)                   # number of genes present
print("=== Q2. Is depth just total load? ===")
r,p=spearmanr(meta["Depth_cm"].values,total)
print("Spearman(depth, total_load) = %.3f (p=%.3g)"%(r,p))
r2,p2=spearmanr(meta["Depth_cm"].values,rich)
print("Spearman(depth, richness)   = %.3f (p=%.3g)"%(r2,p2))

def auc_cv(Xf,y):
    Xf=Xf.reshape(-1,1) if Xf.ndim==1 else Xf
    est=Pipeline([("sc",StandardScaler()),("lr",LogisticRegression(max_iter=5000))])
    pr=cross_val_predict(est,Xf,y,cv=gkf,groups=groups,method="predict_proba")[:,1]
    return roc_auc_score(y,pr)
def auc_rf(Xf,y):
    Xf=Xf.reshape(-1,1) if Xf.ndim==1 else Xf
    rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=42,n_jobs=1)
    pr=cross_val_predict(rf,Xf,y,cv=gkf,groups=groups,method="predict_proba")[:,1]
    return roc_auc_score(y,pr)

print("\n=== Q1+Q3. Trivial baselines vs full model, ALL thresholds ===")
print(f"{'thr':>4} {'n_act':>5} | {'total(LR)':>9} {'rich(LR)':>8} {'83gene(LR)':>10} {'83gene(RF)':>10}")
for t in [30,40,50,60,70]:
    y=(meta["Depth_cm"]>t).astype(int).values
    if y.sum()==0 or y.sum()==len(y): continue
    n_act=int((y==0).sum())
    print(f"{t:>4} {n_act:>5} | {auc_cv(total,y):>9.3f} {auc_cv(rich,y):>8.3f} {auc_cv(X,y):>10.3f} {auc_rf(X,y):>10.3f}")

print("\n=== Q4. Leakage sanity: shuffled labels should give ~0.5 ===")
y=(meta["Depth_cm"]>40).astype(int).values
rng=np.random.default_rng(0)
shuf=[auc_rf(X,rng.permutation(y)) for _ in range(20)]
print("shuffled-label RF AUC: mean=%.3f range=[%.3f, %.3f]  (should hover ~0.5)"%(np.mean(shuf),min(shuf),max(shuf)))
print("\nDONE-SELFCHECK")
