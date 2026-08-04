# -*- coding: utf-8 -*-
"""Capstone: integrate all Phase-2 analyses on the representative candidates.
 - novelty vs DRAMP (max % identity)
 - hemolysis (toxicity) score
 - amphipathicity: Eisenberg hydrophobic moment (uH)
 -> rank 'priority' candidates: novel + high AMP score + low hemolysis + amphipathic
"""
import os, math, numpy as np, pandas as pd, joblib, openpyxl
from collections import defaultdict
from Bio.Align import PairwiseAligner, substitution_matrices
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(BASE,"outputs"); FIG=os.path.join(OUT,"figures")
os.makedirs(FIG,exist_ok=True)
AAS="ACDEFGHIKLMNPQRSTVWY"
KD=dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,
        K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
EIS=dict(A=.62,R=-2.53,N=-.78,D=-.90,C=.29,Q=-.85,E=-.74,G=.48,H=-.40,I=1.38,L=1.06,
         K=-1.50,M=.64,F=1.19,P=.12,S=-.18,T=-.05,W=.81,Y=.26,V=1.08)
def feats(seq):
    seq=[a for a in seq if a in KD]; L=len(seq)
    if L<5: return None
    pos=sum(seq.count(a) for a in "KR"); neg=sum(seq.count(a) for a in "DE")
    return dict(length=L,charge=pos-neg,charge_per_res=(pos-neg)/L,gravy=np.mean([KD[a] for a in seq]),
        frac_cationic=pos/L,frac_hydrophobic=sum(seq.count(a) for a in "AILMFVW")/L,
        aromaticity=sum(seq.count(a) for a in "FWY")/L,**{f"f_{a}":seq.count(a)/L for a in AAS})
def hydrophobic_moment(seq, delta=100.0):
    seq=[a for a in seq if a in EIS]
    if not seq: return 0.0
    rad=math.radians(delta)
    sx=sum(EIS[a]*math.cos(i*rad) for i,a in enumerate(seq))
    sy=sum(EIS[a]*math.sin(i*rad) for i,a in enumerate(seq))
    return math.hypot(sx,sy)/len(seq)

rep=pd.read_csv(os.path.join(OUT,"virome_candidates_rep.csv"))
top=rep[rep.amp_score>=0.7].drop_duplicates("pep").copy().reset_index(drop=True)
top=top.sort_values("amp_score",ascending=False).head(300).reset_index(drop=True)
print("finalizing top %d unique candidates (score>=0.7)"%len(top))

# --- hemolysis ---
H=joblib.load(os.path.join(OUT,"hemolysis_model.joblib")); hmodel,hcols=H["model"],H["cols"]
Xh=pd.DataFrame([feats(p) for p in top.pep])[hcols]
top["hemolysis_score"]=hmodel.predict_proba(Xh)[:,1]
# --- amphipathicity ---
top["hydrophobic_moment"]=[hydrophobic_moment(p) for p in top.pep]
# --- novelty vs DRAMP ---
wb=openpyxl.load_workbook(os.path.join(BASE,"data","DRAMP_general.xlsx"),read_only=True); ws=wb.active
it=ws.iter_rows(values_only=True); next(it); AA=set(AAS); dramp=[]
for r in it:
    if r[1]:
        s="".join(c for c in str(r[1]).upper() if c in AA)
        if 5<=len(s)<=100: dramp.append((str(r[0]),s))
aln=PairwiseAligner(); aln.substitution_matrix=substitution_matrices.load("BLOSUM62")
aln.open_gap_score=-10; aln.extend_gap_score=-.5; aln.mode="local"
def maxid(q):
    # EXHAUSTIVE: align to ALL DRAMP. (A k-mer/score prefilter misses short high-identity local
    #  alignments and thus overestimates novelty -> always do an exhaustive alignment. For large
    #  sets use the joblib parallelism in p11_novelty_fix.py.)
    best=0.0
    for _,t in dramp:
        try: a=aln.align(q,t)[0]
        except Exception: continue
        pid=100*sum(1 for x,y in zip(a[0],a[1]) if x==y and x!="-")/len(q)
        if pid>best: best=pid
    return round(best,1)
top["max_identity_to_DRAMP"]=[maxid(p) for p in top.pep]
top["novel"]=top["max_identity_to_DRAMP"]<40
top["low_toxicity"]=top["hemolysis_score"]<0.5
top["amphipathic"]=top["hydrophobic_moment"]>0.35
top["PRIORITY"]=top["novel"]&top["low_toxicity"]&top["amphipathic"]&(top["amp_score"]>=0.8)

cols=["sample","site","length","amp_score","max_identity_to_DRAMP","novel",
      "hemolysis_score","low_toxicity","hydrophobic_moment","amphipathic","PRIORITY","pep"]
top=top[cols].sort_values(["PRIORITY","amp_score"],ascending=False)
top.to_csv(os.path.join(OUT,"virome_final_candidates.csv"),index=False,encoding="utf-8-sig")
print("\n=== SUMMARY ===")
print("candidates score>=0.7:",len(top))
print("  novel (<40%% DRAMP):",int(top.novel.sum()))
print("  low toxicity (<0.5):",int(top.low_toxicity.sum()))
print("  amphipathic (uH>0.35):",int(top.amphipathic.sum()))
print("  *** PRIORITY (novel+low-tox+amphipathic+score>=0.8): %d ***"%int(top.PRIORITY.sum()))
print("\nTOP PRIORITY candidates:")
pc=top[top.PRIORITY].head(12)
print(pc[["sample","site","length","amp_score","max_identity_to_DRAMP","hemolysis_score","hydrophobic_moment","pep"]].to_string(index=False))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(6.5,5))
c=["#d62728" if x else "#bbbbbb" for x in top.PRIORITY]
plt.scatter(top.amp_score,top.hemolysis_score,c=c,s=28,alpha=.75,edgecolor="none")
plt.axhline(0.5,ls="--",c="k",lw=1); plt.xlabel("AMP score (higher=more antimicrobial-like)")
plt.ylabel("hemolysis score (lower=safer)"); plt.title("Permafrost virome AMP candidates\n(red = priority: novel+amphipathic+low-tox)")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"final_candidates.png"),dpi=130); plt.close()
print("\nsaved virome_final_candidates.csv + final_candidates.png ; DONE-FINAL")
