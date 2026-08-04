# -*- coding: utf-8 -*-
import os, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATASET = os.path.join(REPO, "dataset")   # raw KPDC data (not included in repo)
argdir = None
for e in os.scandir(DATASET):
    if '2608' in e.name: argdir = e.path
print("ARG dir:", argdir)
f = os.path.join(argdir, "ARG.abund.table.txt")
df = pd.read_csv(f, sep="\t")
print("shape:", df.shape)
print("columns:", list(df.columns))
print("\n=== first 3 rows (first 6 cols) ===")
print(df.iloc[:3, :6].to_string())
# find Class column
classcol = [c for c in df.columns if c.lower()=='class']
genecol = [c for c in df.columns if 'gene' in c.lower() and 'id' in c.lower().replace('.','')]
print("\nClass col:", classcol, "| gene col:", genecol)
if classcol:
    cc = classcol[0]
    print("\n=== unique genes per antibiotic class (Class) ===")
    gc = genecol[0] if genecol else df.columns[2]
    per = df.groupby(cc)[gc].nunique().sort_values(ascending=False)
    print(per.to_string())
    print("\ntotal unique classes:", df[cc].nunique(), "| total unique genes:", df[gc].nunique())
    print("\n=== check for high-risk clinical genes ===")
    genes = df[gc].astype(str)
    import re
    for label,pat in [("beta-lactam (bla/OXA)", r"bla|oxa|ctx|tem|shv|kpc|ndm|imp|vim"),
                      ("vancomycin (van)", r"van"),
                      ("colistin (mcr)", r"mcr"),
                      ("aminoglycoside (aac/aph/ant)", r"aac|aph|ant|aad|str"),
                      ("macrolide (erm/mph)", r"erm|mph|mef"),
                      ("tetracycline (tet)", r"tet"),
                      ("quinolone (qnr/gyr)", r"qnr|gyr|par"),
                      ("rifampin (arr/rpo)", r"arr|rpo"),
                      ("sulfonamide/trimethoprim (sul/dfr)", r"sul|dfr")]:
        hits = sorted(set(g for g in genes if re.search(pat, g, re.I)))
        print(f"  {label:36s}: {len(hits)} genes {hits[:12]}")
