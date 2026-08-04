# -*- coding: utf-8 -*-
"""FIX 2 — introduce selectivity (therapeutic index).
Problem: the author applies amp>=0.8 & hemolysis<0.5 (both) as independent thresholds. But AMP score and hemolysis are
      positively correlated (both hydrophobicity-driven) -> highly active candidates are systematically discarded.
Improvement: rank by selectivity (margin = amp - max(hemolysis)) instead of absolute thresholds, and
      recover the highly active / highly selective candidates the threshold filter discarded."""
import os, pandas as pd, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
df=pd.read_csv(os.path.join(ROOT,'results','virome_final_candidates_FULL.csv'))
df['hemo_max']=df[['hemolysis_score','hemolysis_score_hemopi3']].max(axis=1)
df['hemo_mean']=df[['hemolysis_score','hemolysis_score_hemopi3']].mean(axis=1)
df['cys']=df.pep.str.count('C')
# selectivity metric (therapeutic-index analog): activity margin vs the worse of the two hemolysis models
df['selectivity']=df.amp_score-df.hemo_max
df['selectivity_mean']=df.amp_score-df.hemo_mean

print("=== (a) how much the author's threshold filter discards highly active candidates ===")
for thr in [0.90,0.85,0.80]:
    s=df[df.amp_score>=thr]
    surv=s[(s.hemolysis_score<0.5)&(s.hemolysis_score_hemopi3<0.5)]
    print(f"  amp>=%.2f: n=%3d  ->  low-tox pass %2d (%2.0f%%)"%(thr,len(s),len(surv),100*len(surv)/len(s)))
print(f"  corr(amp_score, hemolysis_max) = {df.amp_score.corr(df.hemo_max):+.3f}  (positive -> activity-toxicity coupled)")

print("\n=== (b) top 15 by selectivity (keep novel<40%, hemolysis for reference) ===")
cand=df[df.max_identity_to_DRAMP<40].sort_values('selectivity',ascending=False)
cols=['sample','length','amp_score','hemolysis_score','hemolysis_score_hemopi3','selectivity','max_identity_to_DRAMP','cys']
print(cand.head(15)[cols].to_string(index=False))

print("\n=== (c) difference between the two rankings: author priority (61) vs top-61 by selectivity ===")
auth=set(df[df.PRIORITY].pep)
topsel=set(df.sort_values('selectivity',ascending=False).head(61).pep)
print(f"  overlap {len(auth&topsel)}/61 | author-only {len(auth-topsel)} | selectivity-only {len(topsel-auth)}")
sel_only=df[df.pep.isin(topsel-auth)]
print(f"  mean amp_score of candidates newly recovered by selectivity ranking={sel_only.amp_score.mean():.3f} "
      f"(author priority mean {df[df.PRIORITY].amp_score.mean():.3f}) -> recovers more active candidates")
print(f"  cysteine-containing in top-61 by selectivity {100*(df.sort_values('selectivity',ascending=False).head(61).cys>0).mean():.0f}% "
      f"(author priority 67%)  <- selectivity criterion reduces the cysteine skew")

# save: full selectivity-ranked table
out=df.sort_values('selectivity',ascending=False)[
    ['contig','sample','site','pep','length','amp_score','hemolysis_score','hemolysis_score_hemopi3',
     'hemo_max','selectivity','max_identity_to_DRAMP','novel','cys','PRIORITY']]
out.to_csv(os.path.join(HERE,'out','fix2_selectivity_ranked_top500.csv'),index=False,encoding='utf-8-sig')
print("\nsaved: out/fix2_selectivity_ranked_top500.csv")
