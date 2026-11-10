# DeepCARE — Supplementary Material

Source files for the Supplementary Material accompanying the manuscript
*DeepCARE: Transferable assessment of clinically relevant allergen cross-reactivity for unseen pairs using a protein foundation language model-based deep learning framework*, submitted to MDPI *Life*.

**v9.8 (2026-11-10):** the Supplementary Material is now built on the **same
MDPI LaTeX template as the main article** — `mdpi.cls` with the official
`supfile` document type (`\documentclass[life,supfile,submit,pdftex,moreauthors]{mdpi}`).
`supfile` automatically typesets the "Supplementary Materials: …" cover with the
full 16-author line, renumbers figures/tables as S1/S2 and prints the S-page
header. The earlier `elsarticle` shell has been retired. The file now also
carries the complete author roster with initials, all three affiliations, the
equal-contribution and corresponding-author footnotes, and the full manuscript
declarations (CRediT, funding, ethics, data availability, AI statement,
conflicts of interest, supplementary-materials note).

**v9.9 (2026-11-10, layout fix):** `mdpi.cls`'s `supfile` branch prints only
`\maketitle` (title + authors) and silently **skips `\maketitlen`**, which is
the command that typesets `\address`, `\corres` and `\firstnote`. The source now
calls `\maketitlen` explicitly, so the three numbered affiliations, the
corresponding-author addresses and the equal-contribution footnote appear in
the cover area exactly as in `main.tex`. `\abstractkeywords` starts with a
hard-coded `\vspace{-8pt}` that normally cancels `\maketitlen`'s trailing
space; a compensating `\vspace{14pt}` keeps a clear gap above the Abstract.
The duplicated affiliation blocks were removed from the body, leaving the
author-abbreviation legend and the joint-first-author note.

## File map

| File | Role | Content |
|---|---|---|
| `supplement.pdf` | **Upload this as the Supplementary File** | Compiled 4-page MDPI-template Supplementary Material (S1–S3) |
| `supplement.tex` | LaTeX source | MDPI `supfile` source, synced from the Overleaf project |
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

## Result wording

The `supplement.tex` abstract (and therefore the compiled `supplement.pdf`)
reports the holdout figures as: on the family-stratified holdout of 73
never-trained clinical pairs, the ensemble attained an AUPRC of **0.647**
against **0.242** for the homology baseline, and an AUROC of **0.799** against
**0.552**. These are the values reproduced in per-seed form by Figures S1–S2;
they reproduce from the frozen scoring tables and the analysis code released
with the manuscript. No further numeric claims are made in this file map.

## MDPI *Life* "Supplementary Materials" mapping

For the back-matter paragraph, cite the elements as:

```
Figure S1: Extended comparisons and the full per-metric record;
Figure S2: Protocol, corpus, and provenance;
Section S3: Corpus quality checks and provenance.
```

## Rebuild

`supplement.tex` shares the MDPI article template with `main.tex`. It needs, in
the same directory tree:

```
mdpi.cls
Definitions/          (journalnames.tex, logo-*.eps/pdf, *.bst)
figures/FigS1_extended.png
figures/FigS2_protocol.png
```

Compile twice with `pdflatex supplement.tex` (second pass stabilises
`LastPage` and the `Figure S1/S2` cross-references). No `bibtex` pass is
required — the supplementary cites no bibliography entries. The two PNGs are
included in the submission ZIP under `figures/`, so the released source is
compilable end to end.

## Notes

- The `supfile` class option is the MDPI-provided supplementary document type;
  `\supplementary{}` in `main.tex` is a separate back-matter declaration macro
  and is not a class option.
- Figure/table S-numbering comes from `mdpi.cls` itself; do not redefine
  `\thefigure` / `\thetable` locally.
- Because `supfile` skips `\maketitlen` and `\abstractkeywords`, this source
  calls both explicitly after `\begin{document}`: `\maketitlen` restores the
  affiliations / correspondence / footnotes, and `\abstractkeywords` emits the
  abstract and keywords. The `\vspace{14pt}` between them offsets the
  `\vspace{-8pt}` baked into `\abstractkeywords`; without it the Abstract is
  pulled flush against the author line.
- Code, the released data slice, and the frozen result tables live in `code/`,
  `data/` and `tables/`; see the repository root README for reproduction.
