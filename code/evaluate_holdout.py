"""Holdout evaluation from the released slice (real data, no external calls)."""
import argparse, csv
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/holdout_gold_pairs_sample.csv')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.data, encoding='utf-8')))
    y = np.array([int(r['label_gold']) for r in rows])
    ens = np.array([float(r['deepcare_ens_score']) for r in rows])
    cgi = np.array([float(r['cgi35_identity']) for r in rows])

    ap_ens = average_precision_score(y, ens)
    ap_cgi = average_precision_score(y, cgi)
    auc_ens = roc_auc_score(y, ens)
    auc_cgi = roc_auc_score(y, cgi)

    print(f'rows: {len(y)}  gold: {int(y.sum())}')
    print(f'cgi35 screening baseline AUPRC: {ap_cgi:.3f}')
    print(f'DeepCARE ensemble AUPRC:        {ap_ens:.3f}')
    print(f'gain: +{(ap_ens / ap_cgi - 1) * 100:.1f}%')
    print(f'AUROC  baseline: {auc_cgi:.3f}   ensemble: {auc_ens:.3f}')


if __name__ == '__main__':
    main()
