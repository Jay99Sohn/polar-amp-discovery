# -*- coding: utf-8 -*-
"""Item 4 figures:
 (1) AMP classifier feature importance (interpretability)
 (2) Honest novelty figure: [left] distribution of candidates' max identity to DRAMP (sequence novelty)
     [right] PCA of physicochemical space, candidates vs known AMPs — honestly showing
     'properties are AMP-like (overlap), sequences are novel'.
 Note: we do NOT force candidates into a 'cluster separate from known AMPs'. Property overlap is expected.
"""
import os, numpy as np, pandas as pd, joblib, openpyxl
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(BASE,"outputs"); FIG=os.path.join(OUT,"figures")
KD=dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2); AAS="ACDEFGHIKLMNPQRSTVWY"
def stripM(s): return s[1:] if s.startswith("M") and len(s)>1 else s
def feats(seq):
    seq=[a for a in stripM(seq) if a in KD]; L=len(seq)
    if L<5: return None
    pos=sum(seq.count(a) for a in "KR"); neg=sum(seq.count(a) for a in "DE")
    return dict(length=L,charge=pos-neg,charge_per_res=(pos-neg)/L,gravy=np.mean([KD[a] for a in seq]),
        frac_cationic=pos/L,frac_hydrophobic=sum(seq.count(a) for a in "AILMFVW")/L,
        aromaticity=sum(seq.count(a) for a in "FWY")/L,**{f"f_{a}":seq.count(a)/L for a in AAS})

# ---------- (1) feature importance ----------
M=joblib.load(os.path.join(OUT,"amp_model_debiased.joblib")); model,COLS=M["model"],M["cols"]
imp=pd.DataFrame({"f":COLS,"imp":model.feature_importances_}).sort_values("imp",ascending=True).tail(12)
lab={"charge_per_res":"net charge / residue","charge":"net charge","gravy":"hydrophobicity (GRAVY)",
     "frac_cationic":"cationic fraction","frac_hydrophobic":"hydrophobic fraction","aromaticity":"aromaticity","length":"length"}
imp["name"]=[lab.get(x,x.replace("f_","% ")) for x in imp.f]
plt.figure(figsize=(7,5))
plt.barh(imp["name"],imp["imp"],color="#3b7dd8")
plt.xlabel("feature importance (RandomForest)"); plt.title("AMP classifier: what drives antimicrobial activity (top 12)")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"feature_importance.png"),dpi=140); plt.close()
print("saved feature_importance.png")

# ---------- (2) honest novelty figure ----------
cand=pd.read_csv(os.path.join(BASE,"..","results","virome_final_candidates_FULL.csv"))
pri=cand[cand.PRIORITY].copy()
# DRAMP sequences
wb=openpyxl.load_workbook(os.path.join(BASE,"data","DRAMP_general.xlsx"),read_only=True); ws=wb.active
it=ws.iter_rows(values_only=True); next(it); AA=set(AAS); dramp=[]
for r in it:
    if r[1]:
        s="".join(c for c in str(r[1]).upper() if c in AA)
        if 5<=len(s)<=100: dramp.append(s)
rng=np.random.default_rng(42); dramp_s=list(rng.choice(dramp,size=min(3000,len(dramp)),replace=False))
Xd=pd.DataFrame([f for f in map(feats,dramp_s) if f is not None])[COLS]
Xc=pd.DataFrame([f for f in map(feats,pri.pep) if f is not None])[COLS]
sc=StandardScaler().fit(pd.concat([Xd,Xc]))
Z=PCA(2,random_state=42).fit(sc.transform(pd.concat([Xd,Xc])))
Zd=Z.transform(sc.transform(Xd)); Zc=Z.transform(sc.transform(Xc))

fig,ax=plt.subplots(1,2,figsize=(13,5.2))
nov=100*(cand.max_identity_to_DRAMP<40).mean()
ax[0].hist(cand.max_identity_to_DRAMP,bins=25,color="#e07b39",edgecolor="white")
ax[0].axvline(40,ls="--",c="k"); ax[0].text(41,ax[0].get_ylim()[1]*.85,"novelty cutoff 40%",fontsize=9)
ax[0].set_xlabel("max sequence identity to known AMPs (DRAMP), % — EXHAUSTIVE"); ax[0].set_ylabel("candidates (top-500)")
ax[0].set_title("(A) Sequence novelty (top-500): %.0f%% are < 40%% (novel)"%nov)
ax[1].scatter(Zd[:,0],Zd[:,1],s=10,c="#cfcfcf",label=f"Known AMPs (DRAMP, n={len(Xd)})",zorder=1)
ax[1].scatter(Zc[:,0],Zc[:,1],s=45,c="#d62728",edgecolor="#5a0000",linewidth=.5,label=f"Permafrost priority candidates (n={len(Xc)})",zorder=2)
_plist=list(pri.pep); _t="KTEIAKKALKFASDNSPTILTSIGVGGV"   # highlight top-scoring candidate YW1_4
if _t in _plist:
    _j=_plist.index(_t)
    ax[1].scatter(Zc[_j,0],Zc[_j,1],s=230,marker="*",c="#ffd21f",edgecolor="#5a0000",linewidth=1.1,zorder=4,label="top-ranked candidate (YW1_4)")
    ax[1].annotate("YW1_4",(Zc[_j,0],Zc[_j,1]),xytext=(Zc[_j,0]-3.2,Zc[_j,1]+2.6),fontsize=9,fontweight="bold",
                   arrowprops=dict(arrowstyle="->",color="#333",lw=1.1))
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].legend(fontsize=8.5)
ax[1].set_title("(B) Physicochemical space — candidates lie within AMP region (AMP-like)")
fig.suptitle("Permafrost candidates are sequence-novel (A) yet AMP-like in properties (B)  =  novel + active-like",fontsize=12)
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(os.path.join(FIG,"novelty_honest.png"),dpi=140); plt.close()
print("saved novelty_honest.png")

# --- final_candidates scatter (based on corrected CSV) ---
plt.figure(figsize=(6.5,5))
c=["#d62728" if x else "#bbbbbb" for x in cand.PRIORITY]
plt.scatter(cand.amp_score,cand.hemolysis_score,c=c,s=26,alpha=.75,edgecolor="none")
plt.axhline(0.5,ls="--",c="k",lw=1)
plt.xlabel("AMP score (higher = more antimicrobial-like)")
plt.ylabel("hemolysis score (lower = safer)")
plt.title("Permafrost virome AMP candidates (top-500)\n(red = %d priority: novel[exhaustive]+low-tox+score>=0.8)"%int(cand.PRIORITY.sum()))
plt.tight_layout(); plt.savefig(os.path.join(FIG,"final_candidates.png"),dpi=140); plt.close()
print("saved final_candidates.png")
print("PCA explained variance:",[round(v,2) for v in Z.explained_variance_ratio_])
print("DONE-FIG")
