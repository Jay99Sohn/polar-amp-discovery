# -*- coding: utf-8 -*-
"""Independent (blind) reproduction. Does NOT reference any of our result files. Recomputes,
using only the raw data + freshly downloaded reference data, with the exact same method.
Reproducible (lightweight): contig count, ARG drug-mapping, classifier performance, FPR, HemoPI.
Not reproducible here (needs Colab): full funnel (pyrodigal 127k), cysteine, novelty, functional annotation -> noted separately."""
import os, urllib.request, ssl, numpy as np, pandas as pd, re
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATASET=os.path.join(REPO,'dataset')   # raw KPDC data (not included in repo)
VR=os.path.join(os.path.dirname(os.path.abspath(__file__)),'verify_ref'); os.makedirs(VR,exist_ok=True)
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
print("="*60,"\nIndependent reproduction (no result files referenced)\n"+"="*60)

# ---------- 1. virome contig count (from raw data) ----------
vdir=[e.path for e in os.scandir(DATASET) if '2605' in e.name][0]
vfa=os.path.join(vdir,'virome.fasta')
n=0
with open(vfa,encoding='utf-8',errors='ignore') as f:
    for ln in f:
        if ln[:1]=='>': n+=1
print(f"\n[1] virome contig count = {n:,}   (report 127,005)")

# ---------- 2. ARG drug-mapping (from raw data) ----------
adir=[e.path for e in os.scandir(DATASET) if '2608' in e.name][0]
a=pd.read_csv(os.path.join(adir,'ARG.abund.table.txt'),sep="\t"); a['present']=a['counts']>0
ns=a.SampleID.nunique()
def sp(mask):
    s=a[mask & a.present].SampleID.nunique(); return round(100*s/ns)
print(f"\n[2] ARG: samples {ns} · unique genes {a.geneID.nunique()} · classes {a.Class.nunique()}   (report 69·71·15)")
print(f"    beta-lactam {sp(a.Class.eq('BETA-LACTAM'))}% | vancomycin {sp(a.Class.eq('GLYCOPEPTIDE'))}% | rifampin {sp(a.geneID.eq('arr'))}%   (report 99·78·100)")
print(f"    blaOXA {sp(a.geneID.eq('blaOXA'))}% | vanA {sp(a.geneID.eq('vanA'))}% | aminoglycoside {sp(a.Class.eq('AMINOGLYCOSIDE'))}%   (report 93·7·90)")
print(f"    mcr {len([g for g in a.geneID.unique() if 'mcr' in str(g).lower()])} genes | qnr {len([g for g in a.geneID.unique() if 'qnr' in str(g).lower()])} genes | Hg/As {sp(a.Class.eq('MERCURY'))}/{sp(a.Class.eq('ARSENIC'))}%   (report 0·0·100/100)")

# ---------- download reference data afresh ----------
def dl(url,path):
    if os.path.exists(path) and os.path.getsize(path)>0: return
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    open(path,'wb').write(urllib.request.urlopen(req,timeout=90,context=ctx).read())
amp='https://raw.githubusercontent.com/bcgsc/AMPlify/master/data/'
for f in ['AMPlify_AMP_train_common.fa','AMPlify_non_AMP_train_balanced.fa','AMPlify_AMP_test_common.fa','AMPlify_non_AMP_test_balanced.fa']:
    dl(amp+f,os.path.join(VR,f))
os.makedirs(os.path.join(VR,'hemopi'),exist_ok=True)
for d in ['HemoPI-2','HemoPI-3']:
    hp='https://raw.githubusercontent.com/plissonf/ML-guided-discovery-and-design-of-non-hemolytic-peptides/master/Data/Hemolytik_datasets/%s/'%d
    for f in [d+'_model_class0.fasta',d+'_model_class1.fasta',d+'_validation_class0.fasta',d+'_validation_class1.fasta']:
        dl(hp+f,os.path.join(VR,'hemopi',f))

# ---------- 3. classifier (independent re-implementation) ----------
KD=dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
AAS="ACDEFGHIKLMNPQRSTVWY"
def rf_(p):
    s=[];c=""
    for ln in open(p,encoding='utf-8',errors='ignore'):
        ln=ln.strip()
        if ln.startswith('>'):
            if c:s.append(c);c=""
        elif ln:c+=ln
    if c:s.append(c)
    return s
def sM(s): return s[1:] if s.startswith('M') and len(s)>1 else s
def feat(seq):
    seq=[x for x in seq if x in KD]; L=len(seq)
    if L<5: return None
    pos=sum(seq.count(x) for x in 'KR'); neg=sum(seq.count(x) for x in 'DE')
    return dict(length=L,charge=pos-neg,cpr=(pos-neg)/L,gravy=float(np.mean([KD[x] for x in seq])),
        fc=pos/L,fh=sum(seq.count(x) for x in 'AILMFVW')/L,ar=sum(seq.count(x) for x in 'FWY')/L,
        **{f'f_{x}':seq.count(x)/L for x in AAS})
def build(pa,pb):
    A=[sM(s) for s in rf_(pa)]; B=[sM(s) for s in rf_(pb)]; y=np.array([1]*len(A)+[0]*len(B))
    rows=[feat(s) for s in A+B]; keep=[i for i,r in enumerate(rows) if r]
    return pd.DataFrame([rows[i] for i in keep]), y[keep]
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
Xtr,ytr=build(os.path.join(VR,'AMPlify_AMP_train_common.fa'),os.path.join(VR,'AMPlify_non_AMP_train_balanced.fa'))
Xte,yte=build(os.path.join(VR,'AMPlify_AMP_test_common.fa'),os.path.join(VR,'AMPlify_non_AMP_test_balanced.fa'))
base=Pipeline([('s',StandardScaler()),('l',LogisticRegression(max_iter=5000))]).fit(Xtr[['length','charge']],ytr)
pb=base.predict_proba(Xte[['length','charge']])[:,1]
rf=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=-1).fit(Xtr,ytr)
pr=rf.predict_proba(Xte)[:,1]
print(f"\n[3] classifier: baseline(length+charge) AUC={roc_auc_score(yte,pb):.3f} (report 0.705) | de-bias RF AUC={roc_auc_score(yte,pr):.3f} (report 0.953~0.956)")

# ---------- 4. HemoPI ----------
COLS=list(Xtr.columns)
for d in ['HemoPI-2','HemoPI-3']:
    Xm,ym=build(os.path.join(VR,'hemopi',d+'_model_class1.fasta'),os.path.join(VR,'hemopi',d+'_model_class0.fasta'))
    Xv,yv=build(os.path.join(VR,'hemopi',d+'_validation_class1.fasta'),os.path.join(VR,'hemopi',d+'_validation_class0.fasta'))
    m=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=-1).fit(Xm[COLS],ym)
    print(f"[4] {d} validation AUC={roc_auc_score(yv,m.predict_proba(Xv[COLS])[:,1]):.3f} (report 0.868/0.869)")

# ---------- 5. FPR ----------
neg=Xte[yte==0]; pn=rf.predict_proba(neg)[:,1]
print(f"\n[5] FPR (negative pass rate): >=0.7 {100*(pn>=0.7).mean():.2f}% >=0.8 {100*(pn>=0.8).mean():.2f}% >=0.9 {100*(pn>=0.9).mean():.2f}% (report 2.88/0.84/0.24)")

print("\n"+"="*60)
print("* Not reproducible here (needs Colab): full funnel (pyrodigal 127k contigs)·cysteine 67%·novelty 48%·functional annotation 0/61")
print("  -> pyrodigal not installed locally / heavy. These require a separate Colab rerun for fully blind reproduction.")
