# =====================================================================
#  Polar virome antimicrobial-peptide (AMP) pipeline — for Google Colab (revised)
#  - Handles GitHub 429 (rate-limit) retries
#  - Unifies input/output into a single Drive folder 'polar_amp'
#  - Includes both model comparison (#1) and the full virome run (#2)
#  Copy each '# %% [CELL n]' block into its own Colab cell and run top -> bottom.
# =====================================================================

# %% [CELL 1] Mount Drive + set working folder (put virome.fasta in this folder)
import os
from google.colab import drive
drive.mount('/content/drive')
WORKDIR = '/content/drive/MyDrive/polar_amp'      # all inputs/outputs are collected here
os.makedirs(WORKDIR, exist_ok=True)
REF = '/content/refdata'                          # reference data (temporary, not on Drive)
os.makedirs(REF+'/hemopi', exist_ok=True)
print('working folder:', WORKDIR)
print('Put virome.fasta in this folder. Results are saved here too.')
print('current folder contents:', os.listdir(WORKDIR))

# %% [CELL 2] install packages
!pip -q install pyrodigal biopython xgboost openpyxl joblib

# %% [CELL 3] download reference data (with 429 retries)
import time, urllib.request
def dl(url, path, tries=6):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print('already present:', os.path.basename(path)); return
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=90) as r, open(path,'wb') as f:
                f.write(r.read())
            if os.path.getsize(path) > 0:
                print('downloaded:', os.path.basename(path)); return
        except Exception as e:
            wait = 4*(t+1); print('retry %d (%s) waiting %ds...'%(t+1, str(e)[:40], wait)); time.sleep(wait)
    raise RuntimeError('download failed: '+url)
amp='https://raw.githubusercontent.com/bcgsc/AMPlify/master/data/'
for f in ['AMPlify_AMP_train_common.fa','AMPlify_non_AMP_train_balanced.fa',
          'AMPlify_AMP_test_common.fa','AMPlify_non_AMP_test_balanced.fa']:
    dl(amp+f, REF+'/'+f); time.sleep(2)
for ds in ['HemoPI-2','HemoPI-3']:   # negatives = experimentally validated non-hemolytic peptides
    hp='https://raw.githubusercontent.com/plissonf/ML-guided-discovery-and-design-of-non-hemolytic-peptides/master/Data/Hemolytik_datasets/%s/'%ds
    for f in [ds+'_model_class0.fasta',ds+'_model_class1.fasta',ds+'_validation_class0.fasta',ds+'_validation_class1.fasta']:
        dl(hp+f, REF+'/hemopi/'+f); time.sleep(2)
dl('http://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/general_amps.xlsx', REF+'/DRAMP_general.xlsx')
print('reference data ready')

# %% [CELL 4] feature function + train AMP/hemolysis classifiers (strip leading M = de-biased)
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
KD=dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,
        K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2); AAS="ACDEFGHIKLMNPQRSTVWY"
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
Xa,ya=build(REF+'/AMPlify_AMP_train_common.fa', REF+'/AMPlify_non_AMP_train_balanced.fa',1,0)
COLS=list(Xa.columns)
amp_model=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=-1).fit(Xa,ya)
Xh2,yh2=build(REF+'/hemopi/HemoPI-2_model_class1.fasta', REF+'/hemopi/HemoPI-2_model_class0.fasta',1,0)
hemo_model=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=-1).fit(Xh2[COLS],yh2)   # HemoPI-2
Xh3,yh3=build(REF+'/hemopi/HemoPI-3_model_class1.fasta', REF+'/hemopi/HemoPI-3_model_class0.fasta',1,0)
hemo_model3=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=-1).fit(Xh3[COLS],yh3)  # HemoPI-3 (for consensus)
print('models trained')

# %% [CELL 5] #1 multi-model comparison (CV selection -> test once) -> saved to polar_amp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score,f1_score,matthews_corrcoef,accuracy_score
from xgboost import XGBClassifier
Xte,yte=build(REF+'/AMPlify_AMP_test_common.fa', REF+'/AMPlify_non_AMP_test_balanced.fa',1,0)
def P(e): return Pipeline([("sc",StandardScaler()),("m",e)])
models={"LogReg":P(LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=.5,C=.5,max_iter=8000)),
 "SVM_RBF":P(SVC(probability=True,C=2,random_state=42)),
 "kNN":P(KNeighborsClassifier(n_neighbors=11)),
 "RandomForest":RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=42,n_jobs=-1),
 "XGBoost":XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.1,subsample=.9,colsample_bytree=.8,eval_metric="logloss",random_state=42,n_jobs=-1)}
cv=StratifiedKFold(5,shuffle=True,random_state=42); rows=[]
for n,e in models.items():
    r=cross_validate(e,Xa,ya,cv=cv,scoring=["roc_auc","f1"],n_jobs=-1)
    e.fit(Xa,ya); p=e.predict_proba(Xte)[:,1]; yh2=(p>=.5).astype(int)
    rows.append(dict(model=n,cv_AUC=round(r["test_roc_auc"].mean(),3),cv_AUC_sd=round(r["test_roc_auc"].std(),3),
        test_AUC=round(roc_auc_score(yte,p),3),test_F1=round(f1_score(yte,yh2),3),
        test_MCC=round(matthews_corrcoef(yte,yh2),3),test_ACC=round(accuracy_score(yte,yh2),3)))
cmp=pd.DataFrame(rows).sort_values("cv_AUC",ascending=False)
cmp.to_csv(WORKDIR+'/model_comparison.csv',index=False,encoding='utf-8-sig')
print(cmp.to_string(index=False)); print('\nsaved:', WORKDIR+'/model_comparison.csv')

# %% [CELL 6] full virome run (reads polar_amp/virome.fasta)
import pyrodigal
VIROME = WORKDIR+'/virome.fasta'     # put virome.fasta in the polar_amp folder
assert os.path.exists(VIROME), 'virome.fasta not found in the polar_amp folder: '+VIROME
LMIN,LMAX=10,60; gf=pyrodigal.GeneFinder(meta=True)
def contigs(path):
    hid=None; seq=[]
    for ln in open(path):
        if ln.startswith(">"):
            if hid is not None: yield hid,"".join(seq)
            hid=ln[1:].strip().split()[0]; seq=[]
        else: seq.append(ln.strip())
    if hid is not None: yield hid,"".join(seq)
rows=[]; nc=0
for hid,seq in contigs(VIROME):
    nc+=1
    if len(seq)<60: continue
    for g in gf.find_genes(seq.encode()):
        pep=stripM(g.translate().rstrip("*"))
        if not (LMIN<=len(pep)<=LMAX): continue
        f=feats(pep)
        if f is None: continue
        sample=hid.split(".")[0]; site="".join(c for c in sample if not c.isdigit()).rstrip("_")
        rows.append(dict(contig=hid,sample=sample,site=site,pep=pep,length=len(pep),feats=f))
    if nc%10000==0: print('...%d contigs, %d candidates'%(nc,len(rows)))
X=pd.DataFrame([r['feats'] for r in rows])[COLS]
df=pd.DataFrame([{k:r[k] for k in ('contig','sample','site','pep','length')} for r in rows])
df['amp_score']=amp_model.predict_proba(X)[:,1]
df=df.sort_values('amp_score',ascending=False).reset_index(drop=True)
df.to_csv(WORKDIR+'/virome_candidates_ALL.csv',index=False,encoding='utf-8-sig')
print('full run done: contigs %d, candidates %d, score>=0.7: %d'%(nc,len(df),(df.amp_score>=0.7).sum()))

# %% [CELL 7] novelty + hemolysis + amphipathicity -> final table (saved to polar_amp)
import math, openpyxl
from collections import defaultdict
from Bio.Align import PairwiseAligner, substitution_matrices
top=df[df.amp_score>=0.7].drop_duplicates('pep').sort_values('amp_score',ascending=False).head(500).reset_index(drop=True)
_Xtop=pd.DataFrame([feats(p) for p in top.pep])[COLS]
top['hemolysis_score']=hemo_model.predict_proba(_Xtop)[:,1]                 # HemoPI-2
top['hemolysis_score_hemopi3']=hemo_model3.predict_proba(_Xtop)[:,1]        # HemoPI-3
EIS=dict(A=.62,R=-2.53,N=-.78,D=-.90,C=.29,Q=-.85,E=-.74,G=.48,H=-.40,I=1.38,L=1.06,K=-1.5,M=.64,F=1.19,P=.12,S=-.18,T=-.05,W=.81,Y=.26,V=1.08)
def uH(seq,delta=100.):
    seq=[a for a in seq if a in EIS]
    if not seq: return 0.
    r=math.radians(delta); sx=sum(EIS[a]*math.cos(i*r) for i,a in enumerate(seq)); sy=sum(EIS[a]*math.sin(i*r) for i,a in enumerate(seq))
    return round(math.hypot(sx,sy)/len(seq),3)
top['hydrophobic_moment']=[uH(p) for p in top.pep]
wb=openpyxl.load_workbook(REF+'/DRAMP_general.xlsx',read_only=True); ws=wb.active; it=ws.iter_rows(values_only=True); next(it)
AA=set(AAS); dramp=[]
for r in it:
    if r[1]:
        s="".join(c for c in str(r[1]).upper() if c in AA)
        if 5<=len(s)<=100: dramp.append(s)
al=PairwiseAligner(); al.substitution_matrix=substitution_matrices.load("BLOSUM62"); al.open_gap_score=-10; al.extend_gap_score=-.5; al.mode="local"
def pct_id(q,t):
    try: a=al.align(q,t)[0]
    except Exception: return 0.
    return 100*sum(1 for x,y in zip(a[0],a[1]) if x==y and x!="-")/len(q)
def maxid(q):   # EXHAUSTIVE: align to all DRAMP. (a k-mer/score prefilter misses short high-identity local alignments and overestimates novelty)
    return round(max((pct_id(q,t) for t in dramp), default=0.),1)
import joblib
top['max_identity_to_DRAMP']=joblib.Parallel(n_jobs=-1, prefer='threads')(joblib.delayed(maxid)(p) for p in top.pep)
top['novel']=top['max_identity_to_DRAMP']<40
top['low_toxicity']=(top['hemolysis_score']<0.5)&(top['hemolysis_score_hemopi3']<0.5)   # consensus of both models (conservative)
top['PRIORITY']=top['novel']&top['low_toxicity']&(top['amp_score']>=0.8)
top=top.sort_values(['PRIORITY','amp_score'],ascending=False)
top.to_csv(WORKDIR+'/virome_final_candidates_FULL.csv',index=False,encoding='utf-8-sig')
print('saved ->', WORKDIR+'/virome_final_candidates_FULL.csv')
print('novel:%d  low-tox:%d  priority:%d'%(int(top.novel.sum()),int(top.low_toxicity.sum()),int(top.PRIORITY.sum())))
print('\nfinal polar_amp folder contents:', os.listdir(WORKDIR))
