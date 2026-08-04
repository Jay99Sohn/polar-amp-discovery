# -*- coding: utf-8 -*-
"""Phase 2 novelty: are the top virome AMP candidates already-known AMPs, or novel?
For each candidate, find max % sequence identity to any DRAMP peptide
(k-mer prefilter -> local BLOSUM62 alignment). Flag <40% identity as novel.
"""
import os, numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from Bio.Align import PairwiseAligner, substitution_matrices
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(BASE,"outputs")

# --- load DRAMP peptides ---
wb=openpyxl.load_workbook(os.path.join(BASE,"data","DRAMP_general.xlsx"), read_only=True)
ws=wb.active; it=ws.iter_rows(values_only=True); next(it)
AA=set("ACDEFGHIKLMNPQRSTVWY"); dramp=[]
for r in it:
    did,seq=r[0],r[1]
    if not seq: continue
    seq="".join(c for c in str(seq).upper() if c in AA)
    if 5<=len(seq)<=100: dramp.append((str(did),seq))
print("DRAMP reference peptides:",len(dramp))

aln=PairwiseAligner(); aln.substitution_matrix=substitution_matrices.load("BLOSUM62")
aln.open_gap_score=-10; aln.extend_gap_score=-0.5; aln.mode="local"
def pct_identity(q,t):
    try: a=aln.align(q,t)[0]
    except Exception: return 0.0
    idt=sum(1 for x,y in zip(a[0],a[1]) if x==y and x!="-")
    return 100.0*idt/len(q)

cand=pd.read_csv(os.path.join(OUT,"virome_amp_candidates.csv"))
top=cand[cand.amp_score>=0.7].copy().reset_index(drop=True)
print("candidates (score>=0.7):",len(top))
res=[]
for _,row in top.iterrows():
    q=row["pep"]; best=0.0; bid=""
    for did,t in dramp:                       # EXHAUSTIVE: align to ALL DRAMP (a prefilter overestimates novelty)
        pid=pct_identity(q,t)
        if pid>best: best=pid; bid=did
    res.append((round(best,1),bid))
top["max_identity_to_DRAMP"]=[r[0] for r in res]
top["best_DRAMP_hit"]=[r[1] for r in res]
top["novel(<40%)"]=top["max_identity_to_DRAMP"]<40
top=top.sort_values("amp_score",ascending=False)
top.to_csv(os.path.join(OUT,"virome_candidates_novelty.csv"),index=False,encoding="utf-8-sig")

print("\nidentity-to-known-AMP distribution:")
print("  <40%% (novel): %d | 40-70%%: %d | >70%% (known-like): %d"
      %((top["max_identity_to_DRAMP"]<40).sum(),
        ((top["max_identity_to_DRAMP"]>=40)&(top["max_identity_to_DRAMP"]<=70)).sum(),
        (top["max_identity_to_DRAMP"]>70).sum()))
print("\nTOP 12 candidates with novelty:")
print(top.head(12)[["sample","length","amp_score","max_identity_to_DRAMP","best_DRAMP_hit","pep"]].to_string(index=False))
print("\nsaved virome_candidates_novelty.csv ; DONE-P3")
