# -*- coding: utf-8 -*-
"""Phase 2: mine permafrost virome for candidate AMPs.
 contigs -> pyrodigal ORFs -> short peptides (strip leading M) -> score with
 de-biased AMP model -> rank. SUBSET run first (validate before full 127k).
"""
import os, sys, numpy as np, pandas as pd, joblib, pyrodigal
np.random.seed(42)
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=os.path.join(BASE,"outputs")
VIR=os.path.join(os.path.dirname(BASE),"dataset","[KOPRI-KPDC-00002605]","virome.fasta")  # raw KPDC data (not included in repo)
N_CONTIGS=int(sys.argv[1]) if len(sys.argv)>1 else 3000
LMIN,LMAX=10,60          # mature-AMP length window

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

M=joblib.load(os.path.join(OUT,"amp_model_debiased.joblib")); model,cols=M["model"],M["cols"]
orf_finder=pyrodigal.GeneFinder(meta=True)

def contigs(path,n):
    hid=None; seq=[]; got=0
    for ln in open(path):
        if ln.startswith(">"):
            if hid is not None:
                yield hid,"".join(seq); got+=1
                if got>=n: return
            hid=ln[1:].strip().split()[0]; seq=[]
        else: seq.append(ln.strip())
    if hid is not None and got<n: yield hid,"".join(seq)

rows=[]; n_orf=0; n_contig=0
for hid,seq in contigs(VIR,N_CONTIGS):
    n_contig+=1
    if len(seq)<60: continue
    for gene in orf_finder.find_genes(seq.encode()):
        prot=gene.translate().rstrip("*")
        n_orf+=1
        pep=prot[1:] if prot.startswith("M") and len(prot)>1 else prot   # strip initiator M
        if not (LMIN<=len(pep)<=LMAX): continue
        f=feats(pep)
        if f is None: continue
        sample=hid.split(".")[0]; site="".join(c for c in sample if not c.isdigit()).rstrip("_")
        rows.append(dict(contig=hid,sample=sample,site=site,pep=pep,length=len(pep),feats=f))

print("contigs scanned=%d, total ORFs=%d, ORFs in %d-%d aa window=%d"%(n_contig,n_orf,LMIN,LMAX,len(rows)))
if not rows:
    print("no candidate-length ORFs; DONE-P2"); sys.exit()
X=pd.DataFrame([r["feats"] for r in rows])[cols]
score=model.predict_proba(X)[:,1]
df=pd.DataFrame([{k:r[k] for k in ("contig","sample","site","pep","length")} for r in rows])
df["amp_score"]=score
df=df.sort_values("amp_score",ascending=False).reset_index(drop=True)
print("\nscore distribution: >=0.9: %d | >=0.7: %d | >=0.5: %d  (of %d)"
      %((score>=.9).sum(),(score>=.7).sum(),(score>=.5).sum(),len(score)))
print("\nsite distribution of high-conf (score>=0.9) candidates:")
print(df[df.amp_score>=.9]["site"].value_counts().to_string())
print("\nTOP 12 candidates:")
print(df.head(12)[["sample","site","length","amp_score","pep"]].to_string(index=False))
df.to_csv(os.path.join(OUT,"virome_amp_candidates.csv"),index=False,encoding="utf-8-sig")
print("\nsaved virome_amp_candidates.csv (%d rows) ; DONE-P2"%len(df))
