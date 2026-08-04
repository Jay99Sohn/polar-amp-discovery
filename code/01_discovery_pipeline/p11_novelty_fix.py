# -*- coding: utf-8 -*-
"""Novelty correction (ground truth = exhaustive alignment). A k-mer/score prefilter misses short
high-identity local alignments and is inaccurate -> use exhaustive alignment. joblib parallelism
(Windows-safe: __main__ guard). Self-check against the existing 56 exhaustive results, then correct the top-500."""
import os, time, numpy as np, pandas as pd, openpyxl
from joblib import Parallel, delayed
from Bio.Align import PairwiseAligner, substitution_matrices

def maxid_chunk(peps, dramp):
    al=PairwiseAligner(); al.substitution_matrix=substitution_matrices.load("BLOSUM62")
    al.open_gap_score=-10; al.extend_gap_score=-.5; al.mode="local"
    out=[]
    for q in peps:
        best=0.0
        for t in dramp:
            try: a=al.align(q,t)[0]
            except Exception: continue
            pid=100.0*sum(1 for x,y in zip(a[0],a[1]) if x==y and x!="-")/len(q)
            if pid>best: best=pid
        out.append(round(best,1))
    return out

def exhaustive_all(peps, dramp, nchunks=16):
    chunks=[list(c) for c in np.array_split(np.array(peps,dtype=object),nchunks)]
    res=Parallel(n_jobs=8, backend="threading")(delayed(maxid_chunk)(c,dramp) for c in chunks)  # threading: Windows-safe
    return [x for r in res for x in r]

if __name__=="__main__":
    BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT=os.path.dirname(BASE); OUT=os.path.join(BASE,"outputs")
    wb=openpyxl.load_workbook(os.path.join(BASE,"data","DRAMP_general.xlsx"),read_only=True); ws=wb.active
    it=ws.iter_rows(values_only=True); next(it); AA=set("ACDEFGHIKLMNPQRSTVWY"); DRAMP=[]
    for r in it:
        if r[1]:
            s="".join(c for c in str(r[1]).upper() if c in AA)
            if 5<=len(s)<=100: DRAMP.append(s)
    print("DRAMP:",len(DRAMP), flush=True)

    ex=pd.read_csv(os.path.join(OUT,"novelty_exhaustive_check.csv"))
    t0=time.time(); chk=exhaustive_all(list(ex.pep),DRAMP,8)
    print("[self-check %.0fs] parallel-exhaustive vs existing-exhaustive max diff %.1f (0=identical)"%(time.time()-t0,np.abs(np.array(chk)-ex.exhaustive.values).max()),flush=True)

    fin=pd.read_csv(os.path.join(ROOT,"virome_final_candidates_FULL.csv"))
    fin["max_identity_OLD"]=fin["max_identity_to_DRAMP"]
    t0=time.time(); fin["max_identity_to_DRAMP"]=exhaustive_all(list(fin.pep),DRAMP,16)
    print("[exhaustive %.0fs] top500 done"%(time.time()-t0),flush=True)
    fin["novel"]=fin["max_identity_to_DRAMP"]<40
    fin["PRIORITY"]=fin["novel"]&fin["low_toxicity"]&(fin["amp_score"]>=0.8)
    fin=fin.sort_values(["PRIORITY","amp_score"],ascending=False)
    fin.to_csv(os.path.join(ROOT,"virome_final_candidates_FULL.csv"),index=False,encoding="utf-8-sig")
    pri=fin[fin.PRIORITY]
    print("\n"+"="*60,flush=True)
    print(f"novel(<40%): OLD {int((fin.max_identity_OLD<40).sum())}/500({100*(fin.max_identity_OLD<40).mean():.0f}%) -> corrected {int(fin.novel.sum())}/500({100*fin.novel.mean():.0f}%)",flush=True)
    print(f"PRIORITY: OLD 56 -> corrected {len(pri)}",flush=True)
    print(f"PRIORITY cysteine-containing {int(pri.pep.str.contains('C').sum())}/{len(pri)} | median identity {pri.max_identity_to_DRAMP.median():.1f}%",flush=True)
    print("PRIORITY sites:",pri.site.value_counts().to_dict(),flush=True)
    print("\n=== top-10 priority after correction (truly novel) ===",flush=True)
    print(pri.head(10)[["sample","site","length","amp_score","max_identity_to_DRAMP","hemolysis_score","pep"]].to_string(index=False),flush=True)
    print("\nOLD #1 QSGFTGF... corrected identity:",fin[fin.pep=="QSGFTGFCPLAIFLKKLGVKESH"]["max_identity_to_DRAMP"].values,flush=True)
    print("DONE-FIX2",flush=True)
