# -*- coding: utf-8 -*-
"""FIX 3 — standardize the novelty metric (BLAST-style identity+coverage).
Problem with the author metric: identity = matches/len(query) (local alignment, divided by full length even if only part of the query is covered).
  - dividing by len(query) -> overestimates novelty.
  - dividing by alignment length only (pure local) degenerates to 100% because of short exact-match fragments -> also inappropriate.
Standard fix: take each candidate's 'top-scoring hit' (BLAST top hit) and look at that hit's
  identity (= matches/alignment length) together with query coverage (= fraction of the query covered by the alignment).
  'non-novel' is defined as identity>=40% AND coverage>=50% (BLAST convention)."""
import os, sys, time, numpy as np, pandas as pd, openpyxl
from joblib import Parallel, delayed
from Bio.Align import PairwiseAligner, substitution_matrices
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); REF=os.path.join(HERE,'ref')

wb=openpyxl.load_workbook(os.path.join(REF,'DRAMP_general.xlsx'),read_only=True); ws=wb.active
it=ws.iter_rows(values_only=True); next(it); AA=set("ACDEFGHIKLMNPQRSTVWY"); DRAMP=[]
for r in it:
    if r[1]:
        s="".join(c for c in str(r[1]).upper() if c in AA)
        if 5<=len(s)<=100: DRAMP.append(s)
print("DRAMP:",len(DRAMP),flush=True)

def analyze(peps):
    al=PairwiseAligner(); al.substitution_matrix=substitution_matrices.load("BLOSUM62")
    al.open_gap_score=-10; al.extend_gap_score=-.5; al.mode="local"
    res=[]
    for q in peps:
        Lq=len(q); best_auth=0.0
        top_score=-1e9; top_id=0.0; top_cov=0.0
        for t in DRAMP:
            try: a=al.align(q,t)[0]
            except Exception: continue
            s0,s1=a[0],a[1]
            m=sum(1 for x,y in zip(s0,s1) if x==y and x!="-")
            auth=100.0*m/Lq
            if auth>best_auth: best_auth=auth
            sc=a.score
            if sc>top_score:                        # BLAST top hit = highest-scoring alignment
                top_score=sc
                aln_len=len(s0)
                top_id=100.0*m/aln_len              # standard identity (alignment-length denominator)
                covered=sum(e-b for b,e in a.aligned[0])  # number of query residues covered by the alignment
                top_cov=100.0*covered/Lq
        res.append((round(best_auth,1),round(top_id,1),round(top_cov,1)))
    return res

def run(peps,nchunks=8):
    chunks=[list(c) for c in np.array_split(np.array(peps,dtype=object),nchunks)]
    out=Parallel(n_jobs=8,backend="threading")(delayed(analyze)(c) for c in chunks)
    return [x for r in out for x in r]

if __name__=="__main__":
    full=pd.read_csv(os.path.join(ROOT,'results','virome_final_candidates_FULL.csv'))
    pri=full[full.PRIORITY].copy().reset_index(drop=True)
    print("re-aligning %d priority candidates (identity+coverage)..."%len(pri),flush=True)
    t0=time.time(); r=run(list(pri.pep),8)
    pri['id_author']=[x[0] for x in r]
    pri['tophit_identity']=[x[1] for x in r]
    pri['tophit_coverage']=[x[2] for x in r]
    print("elapsed %.0fs"%(time.time()-t0),flush=True)
    diff=(pri.id_author-pri.max_identity_to_DRAMP).abs()
    print("author-metric reproduction max error %.1f%%p"%diff.max(),flush=True)

    pri['novel_author']=pri.id_author<40
    # BLAST-style non-novel = top hit has identity>=40% AND coverage>=50%
    pri['novel_blast']=~((pri.tophit_identity>=40)&(pri.tophit_coverage>=50))
    print("\n=== novelty re-judgment of the 61 priority candidates ===",flush=True)
    print("  author metric (<40, denominator=query length) novel: %d/61"%pri.novel_author.sum(),flush=True)
    print("  BLAST-style (id>=40 & cov>=50 => non-novel) novel: %d/61"%pri.novel_blast.sum(),flush=True)
    print("  top-hit coverage median %.0f%% (100 if full-length aligned)"%pri.tophit_coverage.median(),flush=True)
    print("  top-hit identity median %.0f%%"%pri.tophit_identity.median(),flush=True)
    nn=pri[~pri.novel_blast]
    print("\ncandidates reclassified as non-novel %d (top hit id>=40 & cov>=50):"%len(nn),flush=True)
    if len(nn):
        print(nn.sort_values('tophit_identity',ascending=False)[['sample','length','amp_score','id_author','tophit_identity','tophit_coverage']].to_string(index=False),flush=True)
    print("\n  * Note: the large divergence between the author metric and BLAST-style itself shows that a single"
          "\n     local-alignment identity value is unstable as a novelty metric -> MMseqs2/BLASTp (e-value+coverage) recommended.",flush=True)
    pri[['sample','length','pep','amp_score','max_identity_to_DRAMP','id_author','tophit_identity','tophit_coverage','novel_author','novel_blast']].to_csv(
        os.path.join(HERE,'out','fix3_novelty_blast_priority61.csv'),index=False,encoding='utf-8-sig')
    print("\nsaved: out/fix3_novelty_blast_priority61.csv",flush=True)
