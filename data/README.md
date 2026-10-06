# Data slice (real, released subset)

- holdout_gold_pairs_sample.csv: all 313 never-trained holdout rows (73 clinical
  gold pairs + 240 context negatives) with the 35%-identity screening feature,
  the DeepCARE ensemble score, and the gold label. Enough to reproduce the
  holdout evaluation (Table 2) with one command.
- internal_gold_pairs_sample.csv: the 284 internal clinical gold pairs used for
  training. The 9,065 hard negatives and 3,000 anchor negatives follow the
  source databases' distribution terms and are documented in the paper.

Values are copied from the frozen scoring tables; no synthetic rows.
