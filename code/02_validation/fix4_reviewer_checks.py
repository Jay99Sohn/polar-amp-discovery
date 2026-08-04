# -*- coding: utf-8 -*-
"""Immediate checks for reviewer comments: C (FPR base rate) · A2 (low-hemolysis recall / magainin) · D (order info = shuffle control)."""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from pipeline_common import *
import pandas as pd, numpy as np, random
random.seed(42); np.random.seed(42)
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); REF=os.path.join(HERE,'ref')
M=train_models(REF)

# ================= C. false-positive base rate (FPR) =================
print("="*66)
print("C. FPR base rate — pass rate per threshold on the AMPlify test set")
neg=[stripM(s) for s in readfa(REF+'/AMPlify_non_AMP_test_balanced.fa')]   # actual non-AMP 835
pos=[stripM(s) for s in readfa(REF+'/AMPlify_AMP_test_common.fa')]         # actual AMP 835
sneg=score_peptides(neg,M); spos=score_peptides(pos,M)
print(f"  (negatives n={len(sneg)}, positives n={len(spos)})")
for t in [0.7,0.8,0.9]:
    fpr=100*(sneg.amp_score>=t).mean(); tpr=100*(spos.amp_score>=t).mean()
    print(f"  threshold >={t}: FPR(false-positive rate)={fpr:5.2f}%   TPR(sensitivity)={tpr:5.1f}%")
print(f"  > virome funnel >=0.7 pass rate = 3514/168938 = 2.08%")
fpr07=100*(sneg.amp_score>=0.7).mean()
print(f"  > comparison: AMPlify negatives FPR@0.7 = {fpr07:.2f}%  -> basis to judge whether virome 2.08% is explained by noise")

# ================= A2. low-hemolysis AMP recall =================
print("\n"+"="*66)
print("A2. low-toxicity consensus filter (both<0.5) recall — on experimentally verified 'non-hemolytic' peptides")
for ds in ['HemoPI-2','HemoPI-3']:
    non=[stripM(s) for s in readfa(REF+f'/hemopi/{ds}_validation_class0.fasta')]  # validation-set non-hemolytic
    s=score_peptides(non,M)
    passed=((s.hemo2<0.5)&(s.hemo3<0.5)).mean()*100
    print(f"  {ds} validation non-hemolytic n={len(s)}: consensus-filter pass (recall)={passed:.0f}%  (rejected={100-passed:.0f}%)")
# individual well-known low-hemolysis AMPs
print("  individual low-hemolysis controls:")
for nm,seq in [("magainin-2","GIGKFLHSAKKFGKAFVGEIMNS"),("cecropin-A","KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"),
               ("buforin-II","TRSSRAGLQFPVGRVHRLLRK"),("LL-37","LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES")]:
    s=score_peptides([seq],M).iloc[0]
    ok="pass" if (s.hemo2<0.5 and s.hemo3<0.5) else "REJECT"
    print(f"    {nm:12s} HemoPI-2={s.hemo2:.2f} HemoPI-3={s.hemo3:.2f} amp={s.amp_score:.2f} -> consensus filter {ok}")

# ================= D. order information = 0 (shuffle control) =================
print("\n"+"="*66)
print("D. presence of order information — rescore after shuffling residues of priority candidates")
full=pd.read_csv(os.path.join(ROOT,'results','virome_final_candidates_FULL.csv'))
pri=full[full.PRIORITY].pep.tolist()
def shuf(s):
    L=list(s); random.shuffle(L); return "".join(L)
orig=score_peptides(pri,M).set_index('pep')
sh=[shuf(p) for p in pri]
shs=score_peptides(sh,M)
# matching: shuffling changes pep, so compare in order
o=score_peptides(pri,M).amp_score.values
z=score_peptides(sh,M).amp_score.values
print(f"  {len(pri)} priority: original vs shuffled amp_score max diff = {np.abs(o-z).max():.6f}")
print(f"    (0 means the model never looks at sequence order = learns composition only -> uH/amphipathic helices have no selection pressure to begin with)")
print("  3 examples:")
for p,a,b in list(zip(pri,o,z))[:3]:
    print(f"    {p[:28]}... original {a:.4f} / shuffled {b:.4f}")
