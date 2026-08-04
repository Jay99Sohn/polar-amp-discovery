# -*- coding: utf-8 -*-
"""Representative run: stride-sample contigs across the WHOLE virome.fasta
(covers all 5 sites) -> pyrodigal ORFs -> score with de-biased AMP model."""
import os, sys, numpy as np, pandas as pd, joblib, pyrodigal
np.random.seed(42)
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(BASE,"outputs")
VIR=os.path.join(os.path.dirname(BASE),"dataset","[KOPRI-KPDC-00002605]","virome.fasta")  # raw KPDC data (not included in repo)
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 10
LMIN,LMAX=10,60
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
gf=pyrodigal.GeneFinder(meta=True)
def stride_contigs(path,stride):
    hid=None; seq=[]; k=-1
    for ln in open(path):
        if ln.startswith(">"):
            if hid is not None:
                k+=1
                if k%stride==0: yield hid,"".join(seq)
            hid=ln[1:].strip().split()[0]; seq=[]
        else: seq.append(ln.strip())
    if hid is not None:
        k+=1
        if k%stride==0: yield hid,"".join(seq)
rows=[]; nc=0; norf=0
for hid,seq in stride_contigs(VIR,STRIDE):
    nc+=1
    if len(seq)<60: continue
    for g in gf.find_genes(seq.encode()):
        prot=g.translate().rstrip("*"); norf+=1
        pep=prot[1:] if prot.startswith("M") and len(prot)>1 else prot
        if not (LMIN<=len(pep)<=LMAX): continue
        f=feats(pep)
        if f is None: continue
        sample=hid.split(".")[0]; site="".join(c for c in sample if not c.isdigit()).rstrip("_")
        rows.append(dict(contig=hid,sample=sample,site=site,pep=pep,length=len(pep),feats=f))
    if nc%2000==0: print("...%d contigs, %d ORFs, %d candidates"%(nc,norf,len(rows)),flush=True)
print("TOTAL contigs=%d ORFs=%d length-window candidates=%d"%(nc,norf,len(rows)),flush=True)
X=pd.DataFrame([r["feats"] for r in rows])[cols]
sc=model.predict_proba(X)[:,1]
df=pd.DataFrame([{k:r[k] for k in ("contig","sample","site","pep","length")} for r in rows]); df["amp_score"]=sc
df=df.sort_values("amp_score",ascending=False).reset_index(drop=True)
df.to_csv(os.path.join(OUT,"virome_candidates_rep.csv"),index=False,encoding="utf-8-sig")
print("score>=0.9:%d >=0.7:%d  sites covered:%s"%((sc>=.9).sum(),(sc>=.7).sum(),sorted(df.site.unique())),flush=True)
print("site distribution of score>=0.7:\n"+df[df.amp_score>=.7].site.value_counts().to_string(),flush=True)
print("DONE-REP",flush=True)
