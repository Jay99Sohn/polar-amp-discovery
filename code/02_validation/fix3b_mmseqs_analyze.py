# -*- coding: utf-8 -*-
"""FIX 3b — re-judge novelty with MMseqs2 (standard tool). Compared against DRAMP + AMPSphere (863,498).
best hit = max bits. 'non-novel' = best hit has fident>=0.40 AND qcov>=0.50 (BLAST convention)."""
import os, pandas as pd, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
COLS=["query","target","fident","alnlen","qcov","tcov","evalue","bits","qlen","tlen"]

def besthits(path):
    df=pd.read_csv(path,sep="\t",names=COLS)
    df=df.sort_values("bits",ascending=False).drop_duplicates("query",keep="first")
    return df.set_index("query")

dramp=besthits(os.path.join(HERE,"out","mm_top500_vs_dramp.m8"))
amps =besthits(os.path.join(HERE,"out","mm_top500_vs_ampsphere.m8"))

full=pd.read_csv(os.path.join(ROOT,'results','virome_final_candidates_FULL.csv')).reset_index(drop=True)
full['qid']=["cand%d_%s"%(i,r['sample']) for i,r in full.iterrows()]

def annotate(df):
    d=df.copy()
    for tag,bh in [("dramp",dramp),("amps",amps)]:
        d[f"{tag}_fident"]=d.qid.map(bh.fident).fillna(0.0)
        d[f"{tag}_qcov"]  =d.qid.map(bh.qcov).fillna(0.0)
        # non-novel = fident>=0.4 & qcov>=0.5
        d[f"nonnovel_{tag}"]=(d[f"{tag}_fident"]>=0.40)&(d[f"{tag}_qcov"]>=0.50)
        # near-duplicate (essentially identical) = fident>=0.9 & qcov>=0.9
        d[f"dup_{tag}"]=(d[f"{tag}_fident"]>=0.90)&(d[f"{tag}_qcov"]>=0.90)
    d["nonnovel_any"]=d.nonnovel_dramp|d.nonnovel_amps
    d["novel_strict"]=~d.nonnovel_any
    return d

def summarize(d,name):
    n=len(d)
    print(f"\n=== {name} (n={n}) ===")
    print(f"  author call (novel<40%, denom=query length): novel {int((d.max_identity_to_DRAMP<40).sum())}/{n}")
    print(f"  MMseqs vs DRAMP     : non-novel(id>=40&cov>=50) {int(d.nonnovel_dramp.sum())}  -> novel {int((~d.nonnovel_dramp).sum())}/{n}")
    print(f"  MMseqs vs AMPSphere : non-novel(id>=40&cov>=50) {int(d.nonnovel_amps.sum())}  -> novel {int((~d.nonnovel_amps).sum())}/{n}")
    print(f"  > combined (DRAMP + AMPSphere) novel: {int(d.novel_strict.sum())}/{n} ({100*d.novel_strict.mean():.0f}%)")
    print(f"     of which newly caught as non-novel by AMPSphere: {int((d.nonnovel_amps & ~d.nonnovel_dramp).sum())}")
    print(f"     near-duplicates (>=90%id & >=90%cov) vs AMPSphere: {int(d.dup_amps.sum())}  vs DRAMP: {int(d.dup_dramp.sum())}")
    print(f"     best-hit AMPSphere fident median {d.amps_fident.median():.2f}, qcov median {d.amps_qcov.median():.2f}")

pri=annotate(full[full.PRIORITY])
top=annotate(full)
summarize(pri,"priority 61")
summarize(top,"top500")

# save
keep=['sample','length','pep','amp_score','max_identity_to_DRAMP',
      'dramp_fident','dramp_qcov','nonnovel_dramp','amps_fident','amps_qcov','nonnovel_amps','dup_amps','novel_strict','PRIORITY']
top[keep].to_csv(os.path.join(HERE,'out','fix3b_mmseqs_novelty_top500.csv'),index=False,encoding='utf-8-sig')
pri[keep].to_csv(os.path.join(HERE,'out','fix3b_mmseqs_novelty_priority61.csv'),index=False,encoding='utf-8-sig')
print("\nsaved: out/fix3b_mmseqs_novelty_{top500,priority61}.csv")

# candidates among priority whose novelty is shaken by AMPSphere proximity/similarity
susp=pri[pri.nonnovel_amps].sort_values('amps_fident',ascending=False)
if len(susp):
    print("\n[priority 61: analogues exist in AMPSphere (reclassified non-novel)]")
    print(susp[['sample','length','amp_score','amps_fident','amps_qcov','max_identity_to_DRAMP']].head(20).to_string(index=False))
