# AI · Pretrained-Model · Tool Usage Disclosure (Competition Rule 3)

## 1. Generative-AI use
This study used generative AI (Anthropic Claude) as an **assistive tool** for code writing, data analysis, and the design and documentation of self-validation (critical validation).
- **Scope of use:** writing Python analysis code, reproducing/debugging the pipeline, running statistical/ML computations, generating figures, drafting the report.
- **Author's role:** setting the research question, deciding the direction (reframing from discovery → critical validation), the pharmacological interpretation (antibiotic classes, resistance mechanisms, drug susceptibility, stewardship), and the review and final judgment of the numbers and results were all performed and finalized by the author.

## 2. Machine-learning / pretrained models used, and licenses
| Type | Model / tool | License | Scope of use |
|---|---|---|---|
| Machine learning | scikit-learn (RandomForest · SVM · kNN · LogReg) | BSD-3-Clause | AMP classifier · hemolysis prediction (trained in-house on public benchmarks) |
| Machine learning | XGBoost | Apache-2.0 | Model comparison |
| **Pretrained model** | **ESM-2** (esm2_t30_150M, Meta AI) | **MIT** | Used only for **comparative validation** of classifier performance (§4.1) |
| Sequence analysis | MMseqs2 (Steinegger & Söding 2017) | GPL-3.0 | Functional annotation · standard novelty validation |
| Gene prediction | pyrodigal | GPL-3.0 | virome ORF prediction |

## 3. Reference-data sources (Rule 2)
AMPlify (BC Cancer GSC), HemoPI (Raghava lab), DRAMP 3.0, AMPSphere (Santos-Júnior 2024, Cell), UniProt/Swiss-Prot — all public data; detailed sources are given in the report's §6 references. **The KPDC raw data (KOPRI-KPDC-00002605 · 00002608) was used solely for the purpose of competition entry and is NOT redistributed in this repository.**

## 4. Reproducibility
All analytical results are reproducible with the provided scripts; `code/02_validation/independent_reproduction.py` recomputes the key numbers from the KPDC raw data alone (obtained separately), without referencing any of our result files, and confirms agreement with the report. Encoding UTF-8, Windows 11 / Python 3.13, seed 42 (details in report §7.2 and `code/requirements.txt`).
