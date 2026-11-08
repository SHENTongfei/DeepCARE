# DeepCARE — Supplementary Material

Source files for the Supplementary Material accompanying the manuscript
*DeepCARE: a deep learning framework that transfers clinical allergen
cross-reaction assessment to never-trained pairs*.

## File map

| File | Role | Content |
|---|---|---|
| `supplement.pdf` | **Upload this as the Supplementary File** | Compiled 3-page Supplementary Material (S1–S3) |
| `supplement.tex` | LaTeX source | S1–S3 source, synced from the Overleaf project |
| `FigS1_extended_comparisons.pdf` | Figure S1 (vector) | Extended comparisons / full per-metric record, 10 panels |
| `FigS1_extended.png` | Figure S1 (source image) | 600-dpi raster included by `supplement.tex` |
| `FigS2_protocol.pdf` | Figure S2 (vector) | Protocol, corpus and provenance, 11 panels |
| `FigS2_protocol.png` | Figure S2 (source image) | 600-dpi raster included by `supplement.tex` |

## Sections

- **S1 — Extended comparisons and the full per-metric record** (Figure S1):
  per-seed AUPRC deltas, identity-oracle array, nearest-kin error, carried
  calibration cells, knockout strips, case-block distribution, snapshot pool,
  recorded screens, per-seed AUROC and ECE10 deltas.
- **S2 — Protocol, corpus, and provenance** (Figure S2): screen ledger,
  seed-by-fold matrix, family-stratified split, validation temperature,
  protocol-fix timeline, data funnel, holdout coverage, model inventory,
  class balance, holdout AUPRC, case-corpus provenance.
- **S3 — Corpus quality checks and provenance** (text): holdout discipline,
  the out-of-bounds segment record, and the auditable screen ledger.

## MDPI *Life* "Supplementary Materials" mapping

For the back-matter paragraph, cite the elements as:

```
Figure S1: Extended comparisons and the full per-metric record;
Figure S2: Protocol, corpus, and provenance;
Section S3: Corpus quality checks and provenance.
```

## Rebuild

`supplement.tex` expects `FigS1_extended.png` and `FigS2_protocol.png` in the
same directory and the `elsarticle` class. Compile with `pdflatex
supplement.tex` (or latexmk). The two PNGs and the synced `.tex` were added
so that the released source is compilable end to end.

## Notes

- `supplement.tex` still carries the previous target journal's `\journal{}`
  header. Update it (and regenerate `supplement.pdf`) when the manuscript is
  reformatted for the new target journal.
- Code, the released data slice, and the frozen result tables live in `code/`,
  `data/` and `tables/`; see the repository root README for reproduction.
