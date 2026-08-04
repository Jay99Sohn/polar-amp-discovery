# =====================================================================
#  ESM-2 (protein language model) independent comparison — for Google Colab (T4 GPU)
#  Fair comparison of our 27-feature physicochemical RandomForest (test AUC 0.956)
#  against a modern deep-learning embedding (§4.1).
#
#  [before running] Runtime -> Change runtime type -> Hardware accelerator: T4 GPU
#  Paste this whole block into one Colab cell and run (self-contained, no other files).
#  Expected time: ~3-6 min on a T4 (embedding + training).
# =====================================================================
!pip -q install fair-esm scikit-learn

import torch, esm, os, time, urllib.request, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, torch.cuda.get_device_name(0) if device=='cuda' else '(no GPU — set runtime type to T4)')

# ---------- 1) download the AMP benchmark data (with 429 retries) ----------
os.makedirs('esm_data', exist_ok=True)
def dl(url, path, tries=6):
    if os.path.exists(path) and os.path.getsize(path)>0: return
    for t in range(tries):
        try:
            req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            open(path,'wb').write(urllib.request.urlopen(req,timeout=90).read()); return
        except Exception as e:
            print('  retry', t+1, str(e)[:40]); time.sleep(4*(t+1))
    raise RuntimeError('download failed: '+url)
BASE='https://raw.githubusercontent.com/bcgsc/AMPlify/master/data/'
for f in ['AMPlify_AMP_train_common.fa','AMPlify_non_AMP_train_balanced.fa',
          'AMPlify_AMP_test_common.fa','AMPlify_non_AMP_test_balanced.fa']:
    dl(BASE+f, 'esm_data/'+f); time.sleep(2)
print('data ready')

def readfa(p):
    s=[]; c=''
    for ln in open(p):
        ln=ln.strip()
        if ln.startswith('>'):
            if c: s.append(c); c=''
        elif ln: c+=ln
    if c: s.append(c)
    return s
def stripM(x): return x[1:] if x.startswith('M') and len(x)>1 else x   # strip leading M (same as our pipeline)
def load(split):
    a=[stripM(x) for x in readfa(f'esm_data/AMPlify_AMP_{split}_common.fa')]
    n=[stripM(x) for x in readfa(f'esm_data/AMPlify_non_AMP_{split}_balanced.fa')]
    return a+n, np.array([1]*len(a)+[0]*len(n))
Xtr_seq,ytr = load('train'); Xte_seq,yte = load('test')
print(f'train {len(Xtr_seq)}, test {len(Xte_seq)}')

# ---------- 2) load ESM-2 (t30_150M: fits a T4) ----------
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model = model.to(device).eval()
bc = alphabet.get_batch_converter(); LAYER = 30

@torch.no_grad()
def embed(seqs, bs=64):
    out=[]
    for i in range(0, len(seqs), bs):
        chunk=seqs[i:i+bs]
        batch=[(str(j), s[:60]) for j,s in enumerate(chunk)]   # truncate to 60aa for stability
        _,_,toks = bc(batch); toks=toks.to(device)
        rep = model(toks, repr_layers=[LAYER])['representations'][LAYER]
        for k,s in enumerate(chunk):
            L=min(len(s),60); out.append(rep[k,1:L+1].mean(0).float().cpu().numpy())  # mean over residues
        if (i//bs)%5==0: print(f'  embedding {min(i+bs,len(seqs))}/{len(seqs)}')
    return np.array(out)

print('embedding train...'); Etr=embed(Xtr_seq)
print('embedding test...');  Ete=embed(Xte_seq)
print('embedding dim:', Etr.shape)

# ---------- 3) train a classifier on the embeddings -> compare to our physicochemical model ----------
print('\n================ result (independent held-out test) ================')
for name, clf in [('ESM-2 + LogReg', LogisticRegression(max_iter=3000, C=1.0)),
                  ('ESM-2 + RandomForest', RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))]:
    clf.fit(Etr, ytr); p=clf.predict_proba(Ete)[:,1]
    print(f'{name:24s}: test AUC={roc_auc_score(yte,p):.3f}  F1={f1_score(yte,(p>=.5).astype(int)):.3f}')
print(f'{"physchem-27 + RF (our main)":24s}: test AUC=0.956  (reference)')
print('\nInterpretation: if ESM-2 is similar to or higher than our physicochemical model, "the modern method also validates"; if similar, it supports "a simple, interpretable physicochemical model is enough".')
