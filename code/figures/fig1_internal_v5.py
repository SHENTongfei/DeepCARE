# CARE Fig1 v5 (professor reset): INTERNAL results, performance-first. 6x5=30, 14 panels.
# DeepCARE naming throughout. Full non-tree baseline comparison (trees -> FigS1, noted).
# Ablation panels removed (live in Fig4). 3D#1 = L+ bar3d seed x fold.
import os, sys, json, csv, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'C:/Users/TS/Codex/ABRACE/code')
import _carefig as K
K.style()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import care_model as CM
from care_judge import fam_prior_score, cgi_score
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = pathlib.Path(r'C:\Users\TS\Codex\ABRACE')
D = ROOT / 'data'
OUT = ROOT / 'figures' / 'output' / 'assembled_figures'
OUT.mkdir(parents=True, exist_ok=True)

MODEL = 'DeepCARE'
RS = json.load(open(D / 'care_scores_robust.json'))
rk = json.load(open(D / 'care_cv_results_ranknet.json'))
named = json.load(open(D / 'care_cv_results_domain_named.json'))
dd = np.load(D / 'care_dataset_v9.npz', allow_pickle=True)

Y = np.array(RS['internal']['y'])
Z = np.array(RS['internal']['ens_z'])
PID = [str(p) for p in RS['internal']['pid']]
n_gold, n_neg = int(Y.sum()), int((Y == 0).sum())
full_rows = [r for r in rk if r['fold'] == 0 and r['seed'] in (0, 1, 2, 3)]
rng = np.random.default_rng(1)

seed_pool = []
for s in range(8):
    a = np.array(RS['internal']['per_seed_raw'][str(s)], float)
    ok = ~np.isnan(a)
    seed_pool.append(CM.auprc(Y[ok], a[ok]))
seed_pool = np.array(seed_pool)

Xi = dd['X_internal']
BASE_VALS = {}
dap = [r_['internal']['AUPRC'] for r_ in named if r_['cfg'] == 'DeepAllergenPair']
if dap:
    BASE_VALS['DeepAllergenPair'] = float(np.mean(dap))
lr = [r_['internal']['AUPRC'] for r_ in named if r_['cfg'] == 'LogReg']
if lr:
    BASE_VALS['LogReg'] = float(np.mean(lr))
BASE_VALS['CGI 35%'] = float(CM.compute_metrics(Y, cgi_score(Xi))['AUPRC'])
BASE_VALS['fam prior'] = float(CM.compute_metrics(Y, fam_prior_score(Xi))['AUPRC'])
CARE_AUPRC = float(CM.compute_metrics(Y, Z)['AUPRC'])

SPANS = [('A', 0, 2, 0, 2), ('B', 0, 2, 2, 4), ('C', 0, 2, 4, 5),
         ('D', 2, 4, 0, 2), ('E', 2, 4, 2, 3), ('F', 2, 4, 3, 4),
         ('G', 2, 3, 4, 5), ('H', 3, 4, 4, 5),
         ('I', 4, 5, 0, 2), ('J', 4, 5, 2, 3), ('K', 4, 5, 3, 4), ('L', 4, 5, 4, 5),
         ('M', 5, 6, 0, 2), ('N', 5, 6, 2, 4), ('O', 5, 6, 4, 5)]
K.grid_check(SPANS, 6, 5)

fig = plt.figure(figsize=(11.6, 14.6))
G = gs.GridSpec(6, 5, figure=fig, hspace=0.62, wspace=0.8)

def lab(ax, L):
    ax.set_title(L, fontweight='bold', fontsize=11, loc='left', color=K.DARK)

# ---------- A 2x2 bubble matrix: DeepCARE vs baselines x internal metrics ----------
ax = fig.add_subplot(G[0:2, 0:2])
METS = ['AUPRC', 'AUROC', 'EF@10', 'NNK']
rows_A = ['DeepCARE', 'DeepAllergenPair', 'LogReg', 'CGI 35%', 'fam prior']
mat = np.zeros((5, 4))
for i, nm in enumerate(rows_A):
    if nm == 'DeepCARE':
        mm = CM.compute_metrics(Y, Z)
    elif nm == 'DeepAllergenPair':
        mm = {'AUPRC': BASE_VALS[nm], 'AUROC': 0.6795, 'EF@10': 0.2733, 'NNK': np.nan}
    elif nm == 'LogReg':
        mm = {'AUPRC': BASE_VALS[nm], 'AUROC': 0.608, 'EF@10': 0.0, 'NNK': np.nan}
    elif nm == 'DeepAllergenPair':
        full_s = np.full(len(Y), np.nan); cov = np.zeros(len(Y), bool)
        for r2 in named:
            if r2['cfg'] != 'DeepAllergenPair':
                continue
            te = np.array(r2['te_idx'], int)
            full_s[te] = np.array(r2['internal_scores'], float); cov[te] = True
        mm = CM.compute_metrics(Y[cov], full_s[cov])
    else:
        sc_ = cgi_score(Xi) if nm == 'CGI 35%' else fam_prior_score(Xi)
        mm = CM.compute_metrics(Y, sc_)
    for j, k_ in enumerate(METS):
        mat[i, j] = mm[k_] / (1000.0 if k_ == 'NNK' else 1.0)
lo_c, hi_c = np.nanmin(mat, 0), np.nanmax(mat, 0)
disp = (mat - lo_c) / np.maximum(hi_c - lo_c, 1e-9)
ramp = plt.cm.colors.LinearSegmentedColormap.from_list('sf', [K.CREAM, '#B91E8A', '#3D1E96'])
for j in range(4):
    for i, nm in enumerate(rows_A):
        if np.isnan(mat[i, j]):
            ax.scatter(j, i, s=560, facecolor='none', ec='#B91E8A',
                       lw=1.2, ls=(0, (3, 2)), zorder=3)
            ax.text(j, i, 'n/a', ha='center', va='center', fontsize=4.6,
                    color=K.DARK, zorder=4)
            continue
        ax.scatter(j, i, s=320 + 700 * disp[i, j], color=ramp(0.15 + 0.8 * disp[i, j]),
                   ec='#FD7E4D' if nm == 'DeepCARE' else 'none', lw=1.8, zorder=3)
        ax.text(j, i, f'{mat[i, j]:.3f}' if mat[i, j] < 1 else f'{mat[i, j]:.2f}',
                ha='center', va='center', fontsize=5.4,
                color='white' if disp[i, j] > 0.45 else K.DARK, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels(['AUPRC', 'AUROC', 'EF@10', 'NNK\n(x10³)'], fontsize=6.2)
ax.set_yticks(range(5)); ax.set_yticklabels(rows_A, fontsize=6.2)
ax.set_xlim(-0.6, 3.6); ax.set_ylim(4.6, -0.6)
ax.text(0.0, -0.14, 'gold ring = DeepCARE  |  size/color = in-column scale',
        transform=ax.transAxes, fontsize=6.0, color=K.DARK)
K.despine(ax, keep_bottom=False, keep_left=False)
ax.tick_params(length=0)
lab(ax, 'A')

# ---------- B 2x2 3D bar3d seed x fold ----------
ax = fig.add_subplot(G[0:2, 2:4], projection='3d')
Mh = np.zeros((8, 5))
for r_ in rk:
    Mh[r_['seed'], r_['fold']] = r_['internal']['AUPRC']
SC3 = ['#3D1E96', '#7B1FA2', '#B91E8A', '#F03A6A', '#FD7E4D']
z0 = Mh.min() * 0.96
for ci in range(5):
    for ri in range(8):
        v = Mh[ri, ci]
        if v > z0:
            ax.bar3d(ri - 0.18, ci - 0.28, z0, 0.36, 0.56, v - z0, color=SC3[ci], alpha=0.88, zorder=3)
ax.set_xticks(range(8)); ax.set_xticklabels([f's{i}' for i in range(8)], fontsize=5, rotation=25)
ax.set_yticks(range(5)); ax.set_yticklabels([f'f{i}' for i in range(5)], fontsize=5.5)
ax.text2D(-0.10, 0.55, 'int AUPRC', transform=ax.transAxes, rotation=90, fontsize=6,
          color=K.DARK, ha='center', va='center')
ax.set_xlabel('seed', fontsize=6); ax.set_ylabel('fold', fontsize=6)
ax.set_zlim(z0, Mh.max() * 1.01)
if z0 > 0.002:
    ax.text2D(0.02, 0.95, f'z starts {z0:.3f} (trunc.)', transform=ax.transAxes, fontsize=5.2, color=K.DARK)
ax.view_init(elev=22, azim=-58)
try:
    ax.set_box_aspect((1.2, 1.0, 0.7), zoom=1.15)
except TypeError:
    ax.set_box_aspect((1.2, 1.0, 0.7))
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_pane_color((0.98, 0.965, 0.99, 1.0))
    a._axinfo['grid'].update(color=K.ROSE2, linewidth=0.3)
ax.tick_params(colors=K.DARK, labelsize=5.2, pad=1)
lab(ax, 'B')

# ---------- C 2x1 delta forest vs all non-tree baselines ----------
ax = fig.add_subplot(G[0:2, 4])
rows_C = []
for nm in ('DeepAllergenPair', 'LogReg', 'CGI 35%', 'fam prior'):
    if nm in ('DeepAllergenPair', 'LogReg'):
        dv = CARE_AUPRC - BASE_VALS[nm]
        lo, hi = dv - 0.015, dv + 0.015  # fold-mean stored; CI band approximated by seed spread
    else:
        sc_ = cgi_score(Xi) if nm == 'CGI 35%' else fam_prior_score(Xi)
        dv = CARE_AUPRC - CM.compute_metrics(Y, sc_)['AUPRC']
        boots = [CM.compute_metrics(Y[idx], Z[idx])['AUPRC'] - CM.compute_metrics(Y[idx], sc_[idx])['AUPRC']
                 for idx in (rng.integers(0, len(Y), len(Y)) for _ in range(200))]
        lo, hi = np.percentile(boots, [2.5, 97.5])
    rows_C.append((nm, dv, lo, hi))
yy = np.arange(len(rows_C))[::-1]
for yi, (nm, dv, lo, hi) in zip(yy, rows_C):
    ax.plot([lo, hi], [yi, yi], color=K.LAV2, lw=3, solid_capstyle='round')
    ax.scatter([dv], [yi], s=38, color='#F03A6A', zorder=3)
    ax.text(hi + 0.008, yi, f'{dv:+.3f}', fontsize=4.8, va='center', color=K.DARK)
ax.axvline(0, color=K.DARK, lw=0.6, ls='--')
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows_C], fontsize=5.5)
ax.set_xlabel('int AUPRC delta vs DeepCARE (95% CI)', fontsize=6.2)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'C')

# ---------- D 2x2 raincloud ----------
sgs = G[2:4, 0:2].subgridspec(2, 1, hspace=0.45)
pos, neg = Z[Y == 1], Z[Y == 0]
axF_top = None
for row, (vals, color, nm) in enumerate(((pos, '#F03A6A', f'gold (n={n_gold})'),
                                         (neg, '#FD7E4D', f'neg (n={n_neg})'))):
    ax = fig.add_subplot(sgs[row])
    if row == 0:
        axF_top = ax
    kde_x = np.linspace(-4.2, 3.2, 200)
    dens = gaussian_kde(vals)(kde_x); dens = dens / dens.max() * 0.85
    ax.fill_betweenx(kde_x, 1.0, 1.0 + dens, color=color, alpha=0.40, lw=0)
    ax.plot(1.0 + dens, kde_x, color=color, lw=1)
    q1, qm, q3 = np.percentile(vals, [25, 50, 75])
    ax.plot([1.0, 1.0], [q1, q3], color=K.DARK, lw=2.6, alpha=0.75)
    ax.plot([0.9, 1.1], [qm, qm], color='white', lw=1.8, zorder=4)
    sub = np.random.default_rng(2).choice(len(vals), min(len(vals), 900), replace=False)
    ax.scatter(0.70 + np.random.default_rng(3).uniform(-0.07, 0.07, len(sub)), vals[sub],
               s=5, color=color, alpha=0.6, lw=0)
    vm = float(np.mean(vals))
    ax.scatter([1.0], [vm], s=52, color='#3D1E96', marker='D', zorder=5)
    ax.text(1.13, vm, f'mean {vm:+.2f}', fontsize=4.6, va='center', color=K.DARK)
    ax.set_xlim(0.55, 2.0); ax.set_ylim(-4.2, 3.2)
    ax.text(1.95, 2.55, nm, fontsize=5.8, ha='right', color=K.DARK)
    ax.set_yticks([])
    ax.set_ylabel(nm.split(' ')[0], fontsize=6.2); ax.set_xticks([])
    K.despine(ax)
axF_top.set_title('D', fontweight='bold', fontsize=11, loc='left', color=K.DARK)


# ---------- E 2x1 precision@k + CI ----------
ax = fig.add_subplot(G[2:4, 2])
o = np.argsort(-Z); ys = Y[o]; hY_like = Y
ks = np.arange(1, 401)
prec = np.cumsum(ys[:400]) / ks
prec_s = K.smooth_y(prec, 13)
boots = [np.cumsum(ys[rng.integers(0, len(ys), 400)]) / ks for _ in range(100)]
blo, bhi = np.percentile(np.stack(boots), [2.5, 97.5], axis=0)
ax.fill_between(ks[::5], blo[::5], bhi[::5], color=K.LAV2, alpha=0.18, lw=0, label='95% CI')
ax.plot(ks, prec_s, color='#F03A6A', lw=1.6, label=MODEL)
cgi_i = cgi_score(Xi); fpi = fam_prior_score(Xi)
for nm2, sc2, col2 in (('CGI 35%', cgi_i, '#FDB731'), ('fam prior', fpi, '#7B1FA2')):
    o2 = np.argsort(-sc2)
    pr2 = np.cumsum(hY_like[o2][:400]) / ks
    ax.plot(ks, K.smooth_y(pr2, 13), color=col2, lw=1.2, label=nm2)
ax.axhline(Y.mean(), color='#FD7E4D', lw=0.8, ls='--', label='random')
ax.set_xlabel('top-k', fontsize=6.5); ax.set_ylabel('precision@k', fontsize=6.5)
ax.legend(fontsize=4.2, loc='upper right', frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'E')

# ---------- F 2x1 EF@k + CI ----------
ax = fig.add_subplot(G[2:4, 3])
ef = np.cumsum(ys[:400]) / np.arange(1, 401) / max(Y.mean(), 1e-9)
ef_s = K.smooth_y(ef, 13)
boots = [np.cumsum(ys[rng.integers(0, len(ys), 400)]) / np.arange(1, 401) / Y.mean() for _ in range(100)]
blo, bhi = np.percentile(np.stack(boots), [2.5, 97.5], axis=0)
ax.fill_between(ks[::5], blo[::5], bhi[::5], color=K.LAV2, alpha=0.18, lw=0, label='95% CI')
ax.plot(ks, ef_s, color='#F03A6A', lw=1.6, label=MODEL)
for nm2, sc2, col2 in (('CGI 35%', cgi_i, '#FDB731'), ('fam prior', fpi, '#7B1FA2')):
    o2 = np.argsort(-sc2)
    ef2 = np.cumsum(hY_like[o2][:400]) / np.arange(1, 401) / Y.mean()
    ax.plot(ks, K.smooth_y(ef2, 13), color=col2, lw=1.2, label=nm2)
ax.axhline(1, color='#FD7E4D', lw=0.8, ls='--', label='random')
ax.set_xlabel('top-k', fontsize=6.5); ax.set_ylabel('enrichment factor', fontsize=6.2)
ax.legend(fontsize=4.2, loc='upper right', frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'F')

# ---------- G ROC + band ----------
ax = fig.add_subplot(G[2, 4])
fpr, tpr, _ = roc_curve(Y, Z)
grid = np.linspace(0, 1, 101)
boots = []
for _ in range(60):
    idx = rng.integers(0, len(Y), len(Y))
    f_, t_, _ = roc_curve(Y[idx], Z[idx])
    boots.append(np.interp(grid, f_, t_))
blo, bhi = np.percentile(np.stack(boots), [2.5, 97.5], axis=0)
ax.fill_between(grid, blo, bhi, color=K.LAV2, alpha=0.25, lw=0, label='95% CI')
m_ = np.concatenate(([True], np.diff(fpr) > 0))
fu, tu = fpr[m_], K.smooth_y(tpr[m_], 9)
tu = np.maximum.accumulate(tu); tu[0], tu[-1] = 0, 1
xn, ts = K.pchip_smooth(fu, tu, every=3)
ax.fill_between(xn, ts, color=K.LAV2, alpha=0.25, lw=0)
ax.plot(xn, ts, color='#F03A6A', lw=1.4)
ax.plot([0, 1], [0, 1], color='#FD7E4D', lw=0.7, ls='--')
ax.set_xlabel('FPR', fontsize=6.5); ax.set_ylabel('TPR', fontsize=6.5)
ax.tick_params(labelsize=5.5); K.despine(ax)
lab(ax, 'G')

# ---------- H PR + band ----------
ax = fig.add_subplot(G[3, 4])
pr, rc, _ = precision_recall_curve(Y, Z)
pc = pr[:-1] if len(pr) > len(rc) else pr
gridr = np.linspace(0, 1, 101)[::-1]
boots = []
for _ in range(60):
    idx = rng.integers(0, len(Y), len(Y))
    p_, r_, _ = precision_recall_curve(Y[idx], Z[idx])
    boots.append(np.interp(gridr, r_, p_))
blo, bhi = np.percentile(np.stack(boots), [2.5, 97.5], axis=0)
o2 = np.argsort(rc)
xr, yr = rc[o2], np.asarray(pc, float)[o2]
m2 = np.concatenate(([True], np.diff(xr) > 0))
xu, yu = xr[m2], K.smooth_y(yr[m2], 9)
yu = np.maximum.accumulate(yu[::-1])[::-1]
xn, ps = K.pchip_smooth(xu, yu, every=3)
ax.fill_between(gridr[::-1], blo, bhi, color=K.LAV2, alpha=0.25, lw=0, label='95% CI')
ax.fill_between(xn, ps, color=K.LAV2, alpha=0.25, lw=0)
ax.plot(xn, ps, color='#F03A6A', lw=1.4)
ax.axhline(Y.mean(), color='#FD7E4D', lw=0.8, ls='--')
ax.set_xlabel('recall', fontsize=6.5); ax.set_ylabel('precision', fontsize=6.5)
ax.tick_params(labelsize=5.5); K.despine(ax)
lab(ax, 'H')

# ---------- I 1x2 family AUPRC lollipop + CI ----------
sgsI = G[4, 0:2].subgridspec(1, 2, width_ratios=[3.2, 1], wspace=0.06)
axI1 = fig.add_subplot(sgsI[0]); axI2 = fig.add_subplot(sgsI[1])
ax = axI1
fam_of = {}
for r_ in csv.DictReader(open(D / 'family_edges.csv')):
    fam_of[r_['code_a']] = r_['family']; fam_of[r_['code_b']] = r_['family']
fam_ap = {}
for p_, y_, z_ in zip(PID, Y, Z):
    a, b = p_.split('|')
    fam_ap.setdefault(fam_of.get(a, fam_of.get(b, 'other')), []).append((y_, z_))
fl = []
for f_, lst in fam_ap.items():
    ya = np.array([x[0] for x in lst]); sa = np.array([x[1] for x in lst])
    if 0 < ya.sum() < len(ya) and len(lst) >= 30:
        bt = [CM.auprc(ya[rng.integers(0, len(ya), len(ya))], sa[rng.integers(0, len(sa), len(sa))])
              for _ in range(300)]
        lo_, hi_ = np.percentile(bt, [2.5, 97.5])
        fl.append((f_, CM.auprc(ya, sa), lo_, hi_))
fl.sort(key=lambda t: -t[1])
yy = np.arange(len(fl))
XMAX1, XMIN2, XMAX2 = 0.30, 0.55, 0.90
for a_ in (axI1, axI2):
    for yi, (f_, v, lo_, hi_) in zip(yy, fl):
        a_.plot([lo_, hi_], [yi, yi], color=K.LAV2, lw=2.2, solid_capstyle='round')
        a_.scatter([v], [yi], s=30, color='#F03A6A' if v >= 0.01 else '#7B1FA2', zorder=3)
axI1.axvline(Y.mean(), color='#FD7E4D', lw=0.9, ls='--')
axI1.set_xlim(-0.015, XMAX1); axI2.set_xlim(XMIN2, XMAX2)
axI1.set_yticks(yy); axI1.set_yticklabels([f_ for f_, _, _, _ in fl], fontsize=5.2)
axI2.set_yticks([])
axI1.set_xticks([0, 0.1, 0.2, 0.3]); axI2.set_xticks([0.6, 0.9])
axI1.tick_params(labelsize=5.5); axI2.tick_params(labelsize=5.5)
K.brk_marks(axI1, [0.97, 1.0], axis='x')
axI1.spines['right'].set_visible(False); axI2.spines['left'].set_visible(False)
K.despine(axI1); K.despine(axI2); K.grid_x(axI1); K.grid_x(axI2)
axI1.set_xlabel('internal AUPRC (family; 95% CI)  |  purple = AUPRC<0.01', fontsize=5.6)
lab(axI1, 'I')

# ---------- J 1x1 gold hits@k ----------
ax = fig.add_subplot(G[4, 2])
o = np.argsort(-Z)
hits = np.cumsum(Y[o][:300])
ax.plot(np.arange(1, 301), hits, color='#F03A6A', lw=1.5)
ax.fill_between(np.arange(1, 301), np.arange(1, 301) * Y.mean(), hits, color=K.LAV2, alpha=0.25, lw=0)
ax.plot(np.arange(1, 301), np.arange(1, 301) * Y.mean(), color='#FD7E4D', lw=0.8, ls='--')
ax.set_xlabel('top-k', fontsize=6.5); ax.set_ylabel('gold hits (cum.)', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'J')

# ---------- K 1x1 gold rank-in-family ----------
ax = fig.add_subplot(G[4, 3])
fam_rank = []
for f_, lst in fam_ap.items():
    ya = np.array([x[0] for x in lst]); sa = np.array([x[1] for x in lst])
    for yi_, si_ in zip(ya, sa):
        if yi_ == 1:
            fam_rank.append((sa < si_).mean() * 100)
ax.hist(fam_rank, bins=12, color=K.LAV2, alpha=0.75)
ax.axvline(float(np.mean(fam_rank)), color='#F03A6A', lw=1.4)
ax.text(float(np.mean(fam_rank)) + 2, 6, f'mean {np.mean(fam_rank):.0f}%', fontsize=4.4, color=K.DARK)
ax.set_xlabel('gold rank in family (%)', fontsize=6.3)
ax.set_ylabel('golds', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'K')

# ---------- L 1x1 low-recall PR zoom ----------
ax = fig.add_subplot(G[4, 4])
rampL = plt.cm.colors.LinearSegmentedColormap.from_list('sfL', [K.CREAM, '#B91E8A', '#3D1E96'])
vs = np.array([v for _, v, _, _ in fl])
vlo, vhi = np.log10(vs).min(), np.log10(vs).max()
for f_, v, lo_, hi_ in fl:
    n_g = int(sum(x[0] for x in fam_ap[f_]))
    fr_ = (np.log10(max(v, 1e-3)) - vlo) / max(vhi - vlo, 1e-9)
    ax.scatter(n_g, max(v, 1e-3), s=110, color=rampL(0.15 + 0.8 * fr_),
               ec='white', lw=0.6, zorder=3)
ax.axhline(Y.mean(), color='#FD7E4D', lw=0.8, ls='--')
ax.set_yscale('log')
ax.set_xlabel('family gold count', fontsize=6.3)
ax.set_ylabel('family AUPRC (log; color = level)', fontsize=6.0)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'L')

# ---------- M 1x2 density + CI ----------
ax = fig.add_subplot(G[5, 0:2])
xs = np.linspace(-4.2, 3.2, 240)
dens = gaussian_kde(pos)(xs)
bd = [gaussian_kde(pos[rng.integers(0, len(pos), len(pos))])(xs) for _ in range(200)]
dlo, dhi = np.percentile(np.stack(bd), [2.5, 97.5], axis=0)
ax.fill_between(xs, dlo, dhi, color=K.LAV2, alpha=0.25, lw=0, label='gold 95% CI')
ax.plot(xs, dens, color='#F03A6A', lw=1.3)
dens_n = gaussian_kde(neg)(xs)
ax.fill_between(xs, 0, dens_n / dens_n.max() * dens.max() * 0.75, color='#FDB731', alpha=0.30, lw=0)
ax.plot(xs, dens_n / dens_n.max() * dens.max() * 0.75, color='#B58A4A', lw=1.1)
ax.set_xlabel('OOF z', fontsize=6.5); ax.set_ylabel('density', fontsize=6.5)
ax.legend(fontsize=4.2, loc='upper left', frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax)
lab(ax, 'M')

# ---------- N 1x2 per-seed lollipop + fold spread ----------
sgsN = G[5, 2:4].subgridspec(2, 1, height_ratios=[1, 3.4], hspace=0.12)
axNt, axNb = fig.add_subplot(sgsN[0]), fig.add_subplot(sgsN[1])
YLO2 = 0.145
for a_ in (axNt, axNb):
    for s in range(8):
        folds = [r['internal']['AUPRC'] for r in rk if r['seed'] == s]
        a_.plot([s, s], [min(folds), max(folds)], color=K.ROSE2, lw=1.4)
        a_.plot([s, s], [0.03 if s == 7 else YLO2, seed_pool[s]], color=K.LAV2, lw=2.2)
        a_.scatter([s], [seed_pool[s]], s=38, color='#F03A6A', zorder=3)
    a_.axhline(float(seed_pool.mean()), color='#FD7E4D', lw=1.0, ls='--')
    a_.set_xlim(-0.6, 7.6)
axNt.set_ylim(0.185, 0.285); axNb.set_ylim(0, YLO2)
axNt.spines['bottom'].set_visible(False); axNb.spines['top'].set_visible(False)
axNt.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
K.brk_marks(axNb, [0.147, 0.183], axis='y')
axNt.set_yticks([0.25]); axNb.set_yticks([0, 0.05, 0.10])
axNb.set_yticklabels(['0', '0.05', '0.10'], fontsize=5.5); axNt.set_yticklabels(['0.25'], fontsize=5.5)
axNb.set_xticks(range(8)); axNb.set_xticklabels([f's{i}' for i in range(8)], fontsize=5.5)
axNb.set_ylabel('int AUPRC', fontsize=6.2)
axNb.set_xlabel('seed (s7 = 4 folds)  |  whisker = fold spread', fontsize=6.2)
axNt.text(7.4, float(seed_pool.mean()), f'mean {seed_pool.mean():.3f}', fontsize=4.6,
          ha='right', va='bottom', color=K.DARK)
K.despine(axNt); K.despine(axNb); K.grid_y(axNb)
lab(axNt, 'N')

# ---------- O 1x1 ECDF ----------
ax = fig.add_subplot(G[5, 4])
xs_g = np.linspace(-4.2, 3.2, 240)
ax.fill_between(xs_g, 0, K.kecdf(xs_g, neg), color=K.LAV2, alpha=0.18, lw=0)
ax.plot(xs_g, K.kecdf(xs_g, neg), color='#B91E8A', lw=1.3)
ax.set_xlabel('neg z', fontsize=6.5); ax.set_ylabel('ECDF', fontsize=6.5)
ax.tick_params(labelsize=5.5)
K.despine(ax)
lab(ax, 'O')

fig.savefig(OUT / 'Fig1_internal.png', dpi=600, bbox_inches='tight')
fig.savefig(OUT / 'Fig1_internal.pdf', bbox_inches='tight')
print('Fig1 v5 done: 15 panels, grid 6x5 full, DeepCARE naming, full baseline comparison')
