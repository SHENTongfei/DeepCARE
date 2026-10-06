# DeepCARE

**DeepCARE** is a deep learning framework for screening-first assessment of
clinical allergen cross-reaction: given a pair of proteins, it predicts whether
shared clinical IgE sensitization is documented, using eleven complementary
protein views (nine pretrained language-model towers, a family prior, and an
InterPro domain spectrum) fused by a gated ensemble under calibrated ranking.

This repository releases the analysis/figure code, a **real stratified data
slice**, the supplementary figures, and partial result tables so that the
screening baseline and the holdout evaluation can be reproduced end to end.
Per journal policy, the full manuscript text and the main figures are not
hosted here.

## What is in the box

```
code/
  figures/            plotting scripts for the six main figures (Fig 1-6)
                      and the two supplementary figures (Fig S1, Fig S2)
  care_model.py       shared metrics (AUPRC, AUROC, ECE10, Brier, EF@10)
  care_judge.py       screening baseline scorer (35%-identity cgi35 rule)
data/
  holdout_gold_pairs_sample.csv   313 never-trained holdout rows (73 gold,
                                  240 context negatives) with CGI identity,
                                  DeepCARE ensemble score, and gold label
  internal_gold_pairs_sample.csv  the 284 internal clinical gold pairs
supplement/
  FigS1_extended_comparisons.pdf  extended comparisons and disclosure (10 panels)
  FigS2_protocol.pdf              protocol, corpus, and audit (11 panels)
tables/
  table2_main_comparison.csv      internal + holdout benchmark (partial)
  table3_components.csv           adopted-component economics
  table4_case_study.csv           four-pair clinical case study
```

## Environment

- Python 3.10+
- numpy, scikit-learn (only these two are required for the evaluation script)

```bash
pip install numpy scikit-learn
```

## One-command reproduction

Recompute the holdout screening baseline and the ensemble AUPRC from the
released slice, including the per-label breakdown:

```bash
python code/evaluate_holdout.py --data data/holdout_gold_pairs_sample.csv
```

Expected output (values match Table 2 of the paper):

```
rows: 313  gold: 73
cgi35 screening baseline AUPRC: 0.362
DeepCARE ensemble AUPRC:        0.652
gain: +80.1%
```

## Regenerating the figures

The figure scripts read the frozen JSON result tables; place them under
`data/` (see `data/README.md` for the expected file names) and run:

```bash
python code/figures/fig1_internal_v5.py     # Fig 1  (internal benchmark)
python code/figures/fig2_ext1_v4.py         # Fig 2  (never-trained holdout)
python code/figures/fig3_ext3_v8.py         # Fig 3  (clinical case study)
python code/figures/fig4_ablation_v5.py     # Fig 4  (component economics)
python code/figures/fig5_bio_v5.py          # Fig 5  (family structure, chords)
python code/figures/fig6_xai_v4.py          # Fig 6  (gate routing, attribution)
python code/figures/figS1_extended_v3.py    # Fig S1 (extended comparisons)
python code/figures/figS2_protocol_v4.py    # Fig S2 (protocol and audit)
```

Each script writes a 600-dpi PNG plus a vector PDF into
`figures/output/assembled_figures/`.

## Protocol safeguards

- The 73-pair holdout is family-stratified and was scored exactly once per
  seed after the protocol was locked; no holdout statistic participates in
  model or snapshot selection.
- Hard negatives in the training corpus carry downgraded priors (structure,
  not asserted negatives); the released slice contains the holdout scoring
  table and the internal gold list.
- The identity-oracle control (tree ensembles that reach 0.94 internal AUPRC
  and collapse on the holdout) is part of the released evaluation, so internal
  scores always carry transfer context.
- One segment record in the pair-mining data was found out-of-bounds for its
  protein and is skipped by the plotting code; it does not affect any reported
  number.

## Data provenance

Clinical gold pairs follow the curation criteria of the allergen resources
cited in the paper (AllergenOnline and the WHO/IUIS nomenclature). The slice
released here is a subset of the full corpus, sufficient to run the holdout
evaluation; the full corpus follows the source databases' distribution terms.

## Citation

If you use DeepCARE or the released slice, please cite the paper:

> DeepCARE: a deep learning framework that transfers clinical allergen
> cross-reaction assessment to never-trained pairs.

## License

MIT.
