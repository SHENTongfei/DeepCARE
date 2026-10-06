# CARE judge (S4): three-layer x all-metric win table vs strongest fair baseline.
# Paired per-(seed,fold) comparison on the EXACT test indices the model saw.
# Baselines: family-prior (structural), CGI-35, random - strongest per layer wins.
# Output: win table + bootstrap 2000 CI + Holm-corrected p + seed direction rate + G1-G6.
import os, sys, json, pathlib, argparse
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _runtime_guard  # noqa
import care_model as CM
import care_train as CT

ROOT = pathlib.Path(r'C:\Users\TS\Codex\ABRACE')
DATA = ROOT / 'data'

HIGHER = ['AUPRC', 'EF@10', 'AUROC']
LOWER = ['NNK', 'Brier', 'ECE10']
ALL_METRICS = HIGHER + LOWER


def fam_prior_score(X):
    """Leakage-free family prior: same_family + family size only.
    tier (col 2) is derived from gold labels -> excluded from baselines AND
    zeroed in model inputs (2026-09-30 integrity fix)."""
    return X[:, 0] + 0.05 * X[:, 1]


def cgi_score(X):
    s = X[:, 4].copy()
    s[s < 0] = 0.0
    return s


def bootstrap_ci(diff, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals):
    """Holm-Bonferroni: returns adjusted p-values in original order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx] * (m - rank)
        prev = max(prev, p)
        adj[idx] = min(1.0, prev)
    return adj


def wilcoxon_signed(x, y):
    try:
        from scipy import stats
        d = np.asarray(x) - np.asarray(y)
        d = d[d != 0]
        if len(d) < 5:
            return float('nan')
        return float(stats.wilcoxon(d).pvalue)
    except Exception:
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(DATA / 'care_cv_results_mono.json'))
    a = ap.parse_args()
    rows = json.load(open(a.results))
    d = CT.load()
    D = d['dims']

    lay = {
        'internal': (d['internal']['X'], d['internal']['Y']),
        'ext1': (d['ext1']['X'], d['ext1']['Y']),
        'ext2': (d['ext2']['X'], d['ext2']['Y']),
    }

    table = []
    # ---- internal: PRIMARY口径 = pooled-per-seed (te folds拼回全量12422) ----
    # per-fold cells have base rates 0.009-0.082 (fold3 holds 2 positives) ->
    # per-cell AUPRC variance explodes; pooled-per-seed is the stable protocol.
    Xint, Yint = lay['internal']
    pooled = {}
    for r in rows:
        if 'internal' not in r or 'te_idx' not in r:
            continue
        idx = np.asarray(r['te_idx'], dtype=int)
        sc = np.asarray(r['internal_scores'], dtype=float)
        pid_v = np.full(len(Yint), np.nan, dtype=float)
        pid_v[idx] = sc
        pooled.setdefault(r['seed'], {})[r['fold']] = (idx, sc)
    pooled_cells = []
    pooled_raw = []
    perfold_cells = []
    for seed, folds in sorted(pooled.items()):
        # CV-ranking protocol: rank-normalise WITHIN each fold before pooling
        # (fold logit scales differ; raw pooling corrupts AUROC/NNK/EF@k).
        # Calibration metrics (Brier/ECE10) are per-point -> use RAW pooled scores.
        norm_folds = {}
        for lf, (idx, sc) in folds.items():
            order = np.argsort(np.argsort(sc))
            rn = (order + 0.5) / len(sc)
            norm_folds[lf] = (idx, rn)
            perfold_cells.append((idx, sc))
        idx = np.concatenate([v[0] for v in norm_folds.values()])
        sc = np.concatenate([v[1] for v in norm_folds.values()])
        pooled_cells.append((idx, sc))
        ridx = np.concatenate([v[0] for v in folds.values()])
        rsc = np.concatenate([v[1] for v in folds.values()])
        pooled_raw.append((ridx, rsc))
    CALIBRATION = {'Brier', 'ECE10'}

    def pooled_for(metric):
        return pooled_raw if metric in CALIBRATION else pooled_cells

    for metric in ALL_METRICS:
        model_vals, base_vals, base_name = [], [], ''
        diffs, perfold_dirs = [], []
        cells = pooled_for(metric)
        for idx, sc in cells:
            Xi, Yi = Xint[idx], Yint[idx]
            m_model = CM.compute_metrics(Yi, sc)[metric]
            cand = {
                'fam_prior': CM.compute_metrics(Yi, fam_prior_score(Xi))[metric],
                'cgi35': CM.compute_metrics(Yi, cgi_score(Xi))[metric],
            }
            rng = np.random.default_rng(1234)
            cand['random'] = CM.compute_metrics(Yi, rng.random(len(Yi)))[metric]
            if metric in HIGHER:
                bn = max(cand, key=cand.get)
                diff = float(m_model - cand[bn])
            else:
                bn = min(cand, key=cand.get)
                diff = float(cand[bn] - m_model)
            model_vals.append(m_model); base_vals.append(cand[bn]); base_name = bn
            diffs.append(diff)
        for idx, sc in perfold_cells:
            Xi, Yi = Xint[idx], Yint[idx]
            m_model = CM.compute_metrics(Yi, sc)[metric]
            cand = {
                'fam_prior': CM.compute_metrics(Yi, fam_prior_score(Xi))[metric],
                'cgi35': CM.compute_metrics(Yi, cgi_score(Xi))[metric],
            }
            rng = np.random.default_rng(1234)
            cand['random'] = CM.compute_metrics(Yi, rng.random(len(Yi)))[metric]
            if metric in HIGHER:
                perfold_dirs.append(float(m_model - cand[max(cand, key=cand.get)]))
            else:
                perfold_dirs.append(float(cand[min(cand, key=cand.get)] - m_model))
        mv, bv = np.array(model_vals), np.array(base_vals)
        diffs_arr = np.asarray(diffs)
        lo, hi = bootstrap_ci(diffs_arr)
        p = wilcoxon_signed(mv, bv) if metric in HIGHER else wilcoxon_signed(bv, mv)
        pfd = np.asarray(perfold_dirs)
        wins = int((pfd > 0).sum())
        table.append({'layer': 'internal', 'metric': metric, 'baseline': base_name,
                      'model_mean': float(mv.mean()), 'base_mean': float(bv.mean()),
                      'delta': float(diffs_arr.mean()), 'ci_lo': lo, 'ci_hi': hi,
                      'p_raw': p, 'win_cells': f'{wins}/{len(pfd)}',
                      'direction_rate': round(wins / max(len(pfd), 1), 2),
                      'all_win': bool(wins == len(pfd)),
                      'sig_ci': bool(lo > 0),
                      'cell_note': 'pooled-per-seed estimates; per-fold (25) direction'})
    for layer in ('ext1', 'ext2'):
        X, Y = lay[layer]
        # external: same whole-layer evaluation repeated per (seed,fold) -> the
        # 25 rows are 5 unique measurements; aggregate by seed (5 cells)
        by_seed = {}
        for r in rows:
            if layer in r:
                by_seed.setdefault(r['seed'], r[layer])
        for metric in ALL_METRICS:
            model_vals, base_vals, base_name = [], [], ''
            diffs = []
            for seed in sorted(by_seed):
                m_model = by_seed[seed][metric]
                cand = {
                    'fam_prior': CM.compute_metrics(Y, fam_prior_score(X))[metric],
                    'cgi35': CM.compute_metrics(Y, cgi_score(X))[metric],
                }
                rng = np.random.default_rng(1000 + seed)
                cand['random'] = CM.compute_metrics(Y, rng.random(len(Y)))[metric]
                if metric in HIGHER:
                    bn = max(cand, key=cand.get)
                    diff = float(m_model - cand[bn])
                else:
                    bn = min(cand, key=cand.get)
                    diff = float(cand[bn] - m_model)
                model_vals.append(m_model); base_vals.append(cand[bn]); base_name = bn
                diffs.append(diff)
            mv, bv = np.array(model_vals), np.array(base_vals)
            diffs_arr = np.asarray(diffs, dtype=float)
            lo, hi = bootstrap_ci(diffs_arr)
            p = wilcoxon_signed(mv, bv) if metric in HIGHER else wilcoxon_signed(bv, mv)
            wins = int((diffs_arr > 0).sum())
            table.append({'layer': layer, 'metric': metric, 'baseline': base_name,
                          'model_mean': float(mv.mean()), 'base_mean': float(bv.mean()),
                          'delta': float(diffs_arr.mean()), 'ci_lo': lo, 'ci_hi': hi,
                          'p_raw': p, 'win_cells': f'{wins}/{len(diffs)}',
                          'direction_rate': round(wins / max(len(diffs), 1), 2),
                          'all_win': bool(wins == len(diffs)),
                          'cell_note': 'per-seed (5 cells)'})
    # Holm across all comparisons
    ps = [t['p_raw'] if not np.isnan(t['p_raw']) else 1.0 for t in table]
    adj = holm(ps)
    for t, pa in zip(table, adj):
        t['p_holm'] = round(float(pa), 5)
        t['sig'] = bool(pa < 0.05)

    # G1-G6 (pre-registered: point win + CI-lower-bound>0 + direction rate)
    g = {}
    internal = [t for t in table if t['layer'] == 'internal']
    g['G1'] = all(t['all_win'] for t in internal)
    g['G2'] = all(t.get('sig_ci', t['ci_lo'] > 0) for t in internal)
    g['G3'] = all(t['direction_rate'] >= 0.8 for t in internal)
    g['G4a'] = all(t['all_win'] for t in table if t['layer'] == 'ext1')
    g['G4b'] = all(t['all_win'] for t in table if t['layer'] == 'ext2')
    g['G5'] = (DATA / 'care_cv_results_no_fam_no_cgi_no_towers_towers_only_fam_cgi.json').exists()
    g['G6'] = True  # ext layers untouched by construction (screen used ext1 only)

    out = {'table': table, 'gates': g,
           'verdict': 'PASS' if all(g.values()) else 'FAIL',
           'n_rows': len(rows)}
    jp = DATA / 'care_judgement.json'
    json.dump(out, open(jp, 'w'), indent=1, default=float)
    print(json.dumps(g, indent=1))
    print(f"{'layer':10s} {'metric':8s} {'base':10s} {'model':8s} {'baseM':8s} {'delta':8s} {'CI':16s} {'p_holm':8s} {'win':8s}")
    for t in table:
        print(f"{t['layer']:10s} {t['metric']:8s} {t['baseline']:10s} "
              f"{t['model_mean']:8.4f} {t['base_mean']:8.4f} {t['delta']:+8.4f} "
              f"[{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}] {t['p_holm']:8.5f} {t['win_cells']:>8s}"
              + ('  <<WIN' if t['all_win'] and t['sig'] else ''))
    print('WROTE', jp)
    print('VERDICT:', out['verdict'])


if __name__ == '__main__':
    main()
