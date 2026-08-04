# -*- coding: utf-8 -*-
"""Task 3: enrichment. Is the permafrost virome MORE (or less) rich in AMP-like
short peptides than a non-permafrost comparison virome (human gut, GPD)?
Score both with the SAME de-biased AMP model, same 10-60aa window, compare
AMP-like fraction via odds ratio + chi-square. Coarse comparison (caveats noted).
Result in the report: odds ratio ~1.17 (no meaningful enrichment) — consistent with
the false-positive base-rate finding (§4.3.2).
Inputs: the de-biased AMP model (outputs/), a GPD gut-phage comparison FASTA
(data/compare/gpd_sample.faa; public Gut Phage Database, not bundled), and the
representative-run scores (outputs/virome_candidates_rep.csv)."""
import os, numpy as np, pandas as pd, joblib
from scipy.stats import chi2_contingency
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(BASE,"outputs")
LMIN,LMAX=10,60; THR=0.7
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
def strip_M(s): return s[1:] if s.startswith("M") and len(s)>1 else s
M=joblib.load(os.path.join(OUT,"amp_model_debiased.joblib")); model,cols=M["model"],M["cols"]

def score_fasta(path):
    seqs=[];c=""
    for ln in open(path):
        ln=ln.strip()
        if ln.startswith(">"):
            if c:seqs.append(c);c=""
        elif ln:c+=ln
    if c:seqs.append(c)
    rows=[]
    for s in seqs:
        p=strip_M(s)
        if LMIN<=len(p)<=LMAX:
            f=feats(p)
            if f: rows.append(f)
    if not rows: return 0,0
    X=pd.DataFrame(rows)[cols]; sc=model.predict_proba(X)[:,1]
    return int((sc>=THR).sum()), len(sc)

# comparison env: human gut phage (GPD sample)
gpd_pos,gpd_n=score_fasta(os.path.join(BASE,"data","compare","gpd_sample.faa"))
print("GPD (gut phage): %d AMP-like / %d window peptides = %.3f%%"%(gpd_pos,gpd_n,100*gpd_pos/gpd_n))

# permafrost: from representative run candidate table (all window peptides w/ scores)
rep=pd.read_csv(os.path.join(OUT,"virome_candidates_rep.csv"))
pf_pos=int((rep.amp_score>=THR).sum()); pf_n=len(rep)
print("Permafrost virome: %d AMP-like / %d window peptides = %.3f%%"%(pf_pos,pf_n,100*pf_pos/pf_n))

# odds ratio + chi-square
a,b,c,d=pf_pos,pf_n-pf_pos,gpd_pos,gpd_n-gpd_pos
OR=(a*d)/(b*c) if b*c else float("inf")
chi2,pval,_,_=chi2_contingency([[a,b],[c,d]])
print("\n=== ENRICHMENT (permafrost vs gut phage, score>=%.1f) ==="%THR)
print("odds ratio = %.2f  (>1 = permafrost enriched)  chi2 p=%.2e"%(OR,pval))
pd.DataFrame([dict(env="permafrost",amp_like=pf_pos,total=pf_n,frac=pf_pos/pf_n),
             dict(env="gut_phage_GPD",amp_like=gpd_pos,total=gpd_n,frac=gpd_pos/gpd_n)]).to_csv(
    os.path.join(OUT,"enrichment.csv"),index=False,encoding="utf-8-sig")
print("saved enrichment.csv ; DONE-ENRICH")
