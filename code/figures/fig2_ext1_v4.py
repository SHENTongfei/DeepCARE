# CARE Fig2 v4 (reset): EXTERNAL DATASET 1 (holdout) ONLY. 6x5=30 slots, 16 panels.
# 3D#2 = M+ curve wall (8-seed precision@k). CI bands everywhere possible.
import os, sys, json, csv, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'C:/Users/TS/Codex/ABRACE/code')
import care_model as CM
import _carefig as K
K.style()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

ROOT = pathlib.Path(r'C:\Users\TS\Codex\ABRACE')
D = ROOT / 'data'
OUT = ROOT / 'figures' / 'output' / 'assembled_figures'
OUT.mkdir(parents=True, exist_ok=True)

RS = json.load(open(D / 'care_scores_robust.json'))
named = json.load(open(D / 'care_cv_results_domain_named.json'))
dd = np.load(D / 'care_dataset_v9.npz', allow_pickle=True)

hX, hY = dd['X_external_holdout'], dd['Y_external_holdout']
hpid = [str(p) for p in RS['ext1']['pid']]
hsc = np.array(RS['ext1']['ens_z'])
pos_s, neg_s = hsc[hY == 1], hsc[hY == 0]
cgi_col = hX[:, 4].copy(); cgi_col[cgi_col < 0] = np.nan
rng = np.random.default_rng(21)
fam_of = {}
for r_ in csv.DictReader(open(D / 'family_edges.csv')):
    fam_of[r_['code_a']] = r_['family']; fam_of[r_['code_b']] = r_['family']

SPANS = [('A', 0, 2, 0, 2), ('B', 0, 2, 2, 4), ('C', 0, 2, 4, 5), ('D', 2, 4, 0, 2),
         ('E', 2, 3, 2, 3), ('F', 2, 3, 3, 4), ('G', 2, 3, 4, 5), ('H', 3, 4, 2, 3),
         ('I', 3, 4, 3, 4), ('J', 3, 4, 4, 5), ('K', 4, 5, 0, 2), ('L', 4, 5, 2, 3),
         ('M', 4, 5, 3, 5), ('N', 5, 6, 0, 2), ('O', 5, 6, 2, 3), ('P', 5, 6, 3, 5)]
K.grid_check(SPANS, 6, 5)

fig = plt.figure(figsize=(11.6, 14.6))
G = gs.GridSpec(6, 5, figure=fig, hspace=0.62, wspace=0.8)

def lab(ax, L):
    ax.set_title(L, fontweight='bold', fontsize=11, loc='left', color=K.DARK)

# A 2x2 ranked top-40 family stems
ax = fig.add_subplot(G[0:2, 0:2])
order = np.argsort(-hsc)[:40]
top_fams = [fam_of.get(hpid[i].split('|')[0], fam_of.get(hpid[i].split('|')[1], 'other')) for i in order]
fc = {}
for f_ in top_fams:
    fc[f_] = fc.get(f_, 0) + 1
big = [f_ for f_, _ in sorted(fc.items(), key=lambda t: -t[1])[:4]]
FAMCOL = dict(zip(big, ['#3D1E96', '#B91E8A', '#FD7E4D', '#FDB731']))
for k_, i_ in enumerate(order):
    yi = 39 - k_
    col = FAMCOL.get(top_fams[k_], '#3D1E96')
    ax.plot([min(hsc[order]) - 0.3, hsc[i_]], [yi, yi], color=col, lw=1.0, zorder=1)
    ax.scatter([hsc[i_]], [yi], s=30 if hY[i_] == 1 else 14, color=col, zorder=3,
               linewidth=1.2, edgecolor='white' if hY[i_] == 1 else 'none',
               marker='D' if hY[i_] == 1 else 'o')
ax.set_yticks([39, 29, 19, 9, 0])
ax.set_yticklabels(['#1', '#10', '#20', '#30', '#40'], fontsize=5.0)
ax.set_ylabel('rank', fontsize=6.2)
ax.set_xlabel('holdout z, top 40 (white edge = clinical)', fontsize=6.5)
ax.legend(handles=[Line2D([], [], marker='o', ls='', color=FAMCOL[f_], ms=4, label=f_) for f_ in big] +
                  [Line2D([], [], marker='D', ls='', mfc='#B91E8A', mec='white', mew=1.2, ms=4, label='clinical')],
          fontsize=4.6, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False,
          columnspacing=1.6, handletextpad=0.5)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'A')

# B 2x2 precision@k 3 series + CI
ax = fig.add_subplot(G[0:2, 2:4])
fam_prior = hX[:, 0] + 0.05 * hX[:, 1]
cgi_s = np.nan_to_num(cgi_col, nan=0.0)
for name, s_, color in (('DeepCARE', hsc, '#F03A6A'), ('CGI 35%', cgi_s, '#FDB731'),
                        ('fam prior', fam_prior, '#7B1FA2')):
    o = np.argsort(-s_)
    ys_ = hY[o]
    ks = np.arange(1, 121)
    prec = np.cumsum(ys_[:120]) / ks
    prec_s = K.smooth_y(prec, 15)
    if name == 'CARE':
        lo_b, hi_b = [], []
        for k_ in ks[::6]:
            boots = [np.cumsum(ys_[rng.permutation(len(ys_))][:120])[k_ - 1] / k_ for _ in range(100)]
            lo_b.append(np.percentile(boots, 2.5)); hi_b.append(np.percentile(boots, 97.5))
        ax.fill_between(ks[::6], lo_b, hi_b, color=color, alpha=0.15, lw=0, label='DeepCARE 95% CI')
    ax.plot(ks, prec_s, color=color, lw=1.6, label=name)
ax.axhline(hY.mean(), color=K.LAV2, lw=0.7, ls='--', label='prevalence')
ax.set_xlabel('top-k screened', fontsize=7); ax.set_ylabel('precision@k', fontsize=7)
ax.legend(fontsize=5.0, loc='upper right')
K.despine(ax); K.grid_y(ax)
lab(ax, 'B')

# C 2x1 ECDF tall
ax = fig.add_subplot(G[0:2, 4])
for vals, color, lb in ((pos_s, '#F03A6A', 'clinical'), (neg_s, '#FDB731', 'non-clinical')):
    v = np.sort(vals)
    ax.plot(np.arange(1, len(v) + 1) / len(v), v, lw=1.2, color=color, label=lb)
ax.set_xlabel('ECDF', fontsize=6.5); ax.set_ylabel('z', fontsize=6.5)
ax.legend(fontsize=5.0, loc='upper left')
ax.tick_params(labelsize=6); K.despine(ax); K.grid_x(ax)
lab(ax, 'C')

# D 2x2 M+ curve wall: per-seed precision@k (3D#2)
sys.path.insert(0, r'C:/Users/TS/.agents/skills/academic-paper-figure-sop/references/code')
from figure_kit import wall3d
seed_scores = [np.array(z, float) for z in RS['ext1']['per_seed_raw_mean'] if z]
Mwall = np.zeros((len(seed_scores), 120))
for si, z in enumerate(seed_scores):
    o = np.argsort(-z)
    Mwall[si] = np.cumsum(hY[o][:120]) / np.arange(1, 121)
Mw = Mwall[::-1]                                   # far seeds first (correct 3D paint order)
Plab = [f's{i}' for i in range(7, -1, -1)]
PAL8 = ['#3D1E96', '#7B1FA2', '#B91E8A', '#F03A6A', '#FD7E4D', '#FDB731', '#FDE725', '#FDE9B8']
Pal8r = PAL8_Later if False else ['#FDE9B8', '#FDE725', '#FDB731', '#FD7E4D', '#F03A6A', '#B91E8A', '#7B1FA2', '#3D1E96']
ax = fig.add_subplot(G[2:4, 0:2], projection='3d')
wall3d(ax, Mw, row_lab=Plab, colors=Pal8r,
       value_labels=False, legend=False, xlab='k (top-k)', ylab='seed', zlab='',
       elev=22, azim=-55)
ax.text2D(-0.04, 0.55, 'precision@k', transform=ax.transAxes, rotation=90,
          fontsize=6, color=K.DARK, ha='center', va='center')
_leg = ax.legend(handles=[Line2D([], [], color=PAL8[i], label=f's{i}') for i in range(8)],
                 fontsize=4.4, loc='upper left', frameon=False, bbox_to_anchor=(0.0, 1.04))
_leg.set_zorder(1000)
ax.add_artist(_leg)
# depth-correct draw order: walls+curves of far seeds behind near seeds
# (wall3d draws g=0..7 in order but mpl's mixed 2D/3D zorder puts all lines over all walls)
ax.computed_zorder = False
n_coll = len(ax.collections)
for gi, coll in enumerate(list(ax.collections)[:8]):
    coll.set_zorder(20 + gi * 2)          # wall of seed gi
for gi, ln in enumerate(list(ax.lines)[:8]):
    ln.set_zorder(20 + gi * 2 + 1)        # curve of seed gi sits just above its own wall
ax.set_zorder(1)
lab(ax, 'D')

# E density + CI
ax = fig.add_subplot(G[2, 2])
xs = np.linspace(min(pos_s) - 1.2, max(pos_s) + 1.2, 240)
dens = gaussian_kde(pos_s)(xs)
bd = [gaussian_kde(pos_s[rng.integers(0, len(pos_s), len(pos_s))])(xs) for _ in range(200)]
dlo, dhi = np.percentile(np.stack(bd), [2.5, 97.5], axis=0)
ax.fill_between(xs, dlo, dhi, color=K.LAV2, alpha=0.25, lw=0, label='95% CI')
ax.plot(xs, dens, color='#F03A6A', lw=1.3)
dens_n = gaussian_kde(neg_s)(xs)
ax.plot(xs, dens_n / dens_n.max() * dens.max() * 0.75, color='#B58A4A', lw=1.1)
ax.fill_between(xs, 0, dens_n / dens_n.max() * dens.max() * 0.75, color='#FDB731', alpha=0.30, lw=0)
ax.set_xlabel('z', fontsize=6.5); ax.set_ylabel('density', fontsize=6.5)
ax.legend(fontsize=4.2, loc='upper left', frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax)
lab(ax, 'E')

# F decile bars + binomial CI
ax = fig.add_subplot(G[2, 3])
o = np.argsort(-hsc)
dec = np.array_split(o, 10)
dp = [hY[c_].mean() for c_ in dec]
yy = np.arange(10)[::-1]
bcol, balpha = [], []
for i, v in enumerate(dp):
    if i < 4:
        bcol.append('#F03A6A'); balpha.append(0.95)
    else:
        bcol.append('#D9BFEA'); balpha.append(0.45)
for bi, (b_y, b_v) in enumerate(zip(yy, dp)):
    ax.barh(b_y, b_v, color=bcol[bi], height=0.6, alpha=balpha[bi])
ax.axvline(hY.mean(), color='#FD7E4D', lw=1.0, ls='--')
ax.text(hY.mean() + 0.01, 9.4, 'prevalence', fontsize=4.0, color=K.DARK)
for yi, (c_, v) in zip(yy, zip(dec, dp)):
    n_ = len(c_); k_ = int(hY[c_].sum())
    se = np.sqrt(max(v * (1 - v), 1e-9) / n_)
    ax.plot([max(v - 1.96 * se, 0), min(v + 1.96 * se, 1)], [yi, yi], color=K.DARK, lw=1.0)
ax.set_yticks(yy); ax.set_yticklabels([f'D{i+1}' for i in range(10)], fontsize=4.8)
ax.set_xlabel('precision by decile', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'F')

# G ROC + band
ax = fig.add_subplot(G[2, 4])
fpr, tpr, _ = roc_curve(hY, hsc)
grid = np.linspace(0, 1, 101)
boots = []
for _ in range(80):
    idx = rng.integers(0, len(hY), len(hY))
    f_, t_, _ = roc_curve(hY[idx], hsc[idx])
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
ax.legend(fontsize=4.2, loc='lower right', frameon=False)
ax.tick_params(labelsize=5.5); K.despine(ax)
lab(ax, 'G')

# H PR + band
ax = fig.add_subplot(G[3, 2])
pr, rc_, _ = precision_recall_curve(hY, hsc)
pr_c = np.asarray(pr[:-1] if len(pr) > len(rc_) else pr, dtype=float)
grid = np.linspace(0, 1, 101)[::-1]
boots = []
for _ in range(80):
    idx = rng.integers(0, len(hY), len(hY))
    p_, r_, _ = precision_recall_curve(hY[idx], hsc[idx])
    boots.append(np.interp(grid, r_, p_))
blo, bhi = np.percentile(np.stack(boots), [2.5, 97.5], axis=0)
o2 = np.argsort(rc_)
xr, yr = rc_[o2], pr_c[o2]
m2 = np.concatenate(([True], np.diff(xr) > 0))
xu, yu = xr[m2], K.smooth_y(yr[m2], 9)
yu = np.maximum.accumulate(yu[::-1])[::-1]
xn, ps = K.pchip_smooth(xu, yu, every=3)
ax.fill_between(grid[::-1], blo, bhi, color=K.LAV2, alpha=0.25, lw=0, label='95% CI')
ax.fill_between(xn, ps, color=K.LAV2, alpha=0.25, lw=0)
ax.plot(xn, ps, color='#F03A6A', lw=1.4)
ax.axhline(hY.mean(), color='#FD7E4D', lw=0.8, ls='--', label='prev')
ax.legend(fontsize=4.2, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
ax.set_xlabel('recall', fontsize=6.5); ax.set_ylabel('precision', fontsize=6.5)
ax.tick_params(labelsize=5.5); K.despine(ax)
lab(ax, 'H')

# I EF@k + CI
ax = fig.add_subplot(G[3, 3])
o = np.argsort(-hsc)
ys_ = hY[o]
ks = np.arange(1, 121)
ef = np.cumsum(ys_[:120]) / ks / max(hY.mean(), 1e-9)
ef_s = K.smooth_y(ef, 11)
boots = [np.cumsum(ys_[rng.permutation(len(ys_))][:120]) / ks / hY.mean() for _ in range(100)]
blo, bhi = np.percentile(np.stack(boots), [2.5, 97.5], axis=0)
ax.fill_between(ks[::4], blo[::4], bhi[::4], color=K.LAV2, alpha=0.18, lw=0, label='95% CI')
ax.plot(ks, ef_s, color='#F03A6A', lw=1.5, label='DeepCARE')
ax.axhline(1, color='#FD7E4D', lw=0.8, ls='--', label='random')
ax.set_xlabel('top-k', fontsize=6.5); ax.set_ylabel('enrichment factor', fontsize=6.3)
ax.legend(fontsize=4.0, loc='upper right', frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'I')

# J CGI percentile scatter
ax = fig.add_subplot(G[3, 4])
pct_rank = np.argsort(np.argsort(hsc)) / (len(hsc) - 1) * 100
has_cgi = cgi_col >= 0
blind = int((~has_cgi).sum())
ax.scatter(cgi_col[has_cgi & (hY == 1)], pct_rank[has_cgi & (hY == 1)], s=18,
           color='#F03A6A', alpha=0.85, label='clinical')
ax.scatter(cgi_col[has_cgi & (hY == 0)], pct_rank[has_cgi & (hY == 0)], s=9,
           color='#FDB731', alpha=0.6, label='non-clin.')
ax.axvline(35, color=K.LAV2, lw=0.7, ls=':')
ax.text(37, 3, '35% thr', fontsize=4.2, color=K.DARK)
ax.set_xlabel(f'CGI %id ({blind} no-hit excl.)', fontsize=6.3)
ax.set_ylabel('z percentile (%)', fontsize=6.3)
ax.legend(fontsize=4.2, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
ax.tick_params(labelsize=6); K.despine(ax); K.grid_y(ax)
lab(ax, 'J')

# K 1x2 family lollipop + CI
ax = fig.add_subplot(G[4, 0:2])
fam_ap = {}
for i in range(len(hX)):
    fam_ap.setdefault(fam_of.get(hpid[i].split('|')[0], fam_of.get(hpid[i].split('|')[1], 'other')),
                      []).append((hY[i], hsc[i]))
fam_list = []
for f_, pairs in fam_ap.items():
    yy_ = np.array([p[0] for p in pairs]); ss_ = np.array([p[1] for p in pairs])
    if yy_.sum() == 0 or yy_.sum() == len(yy_) or len(pairs) < 5:
        continue
    bt = [CM.auprc(yy_[rng.integers(0, len(yy_), len(yy_))], ss_[rng.integers(0, len(ss_), len(ss_))])
          for _ in range(300)]
    lo_, hi_ = np.percentile(bt, [2.5, 97.5])
    fam_list.append((f_, CM.auprc(yy_, ss_), lo_, hi_))
fam_list.sort(key=lambda t: -t[1])
yy = np.arange(len(fam_list))
for yi, (f_, v, lo_, hi_) in zip(yy, fam_list):
    ax.plot([lo_, hi_], [yi, yi], color=K.LAV2, lw=2.2, solid_capstyle='round')
    ax.scatter([v], [yi], s=30, color='#F03A6A' if v >= 0.01 else '#7B1FA2', zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([f_ for f_, _, _, _ in fam_list], fontsize=5.2)
ax.set_xlabel('holdout AUPRC (family; 95% CI)  |  purple = AUPRC<0.01', fontsize=5.8)
ax.set_title('K', fontweight='bold', fontsize=11, loc='left', color=K.DARK)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'K')

# L per-seed lollipop + fold spread
ax = fig.add_subplot(G[4, 2])
sp = [average_precision_score(hY, np.array(z)) for z in RS['ext1']['per_seed_raw_mean'] if z]
rk_ext = {}
for r_ in json.load(open(D / 'care_cv_results_ranknet.json')):
    rk_ext.setdefault(r_['seed'], []).append(r_['ext1']['AUPRC'])
for s in range(8):
    ax.plot([s, s], [min(rk_ext[s]), max(rk_ext[s])], color=K.ROSE2, lw=1.4)
    ax.plot([s, s], [0, sp[s]], color=K.LAV2, lw=2.2)
    ax.scatter([s], [sp[s]], s=36, color='#F03A6A', zorder=3)
ax.axhline(float(np.mean(sp)), color='#FD7E4D', lw=1.0, ls='--')
ax.text(0.98, float(np.mean(sp)) + 0.10, f'mean {np.mean(sp):.3f}', transform=ax.get_yaxis_transform(), fontsize=4.6, ha='right', va='bottom', color=K.DARK)
ax.set_xticks(range(8)); ax.set_xticklabels([f's{i}' for i in range(8)], fontsize=4.8)
ax.set_ylabel('holdout AUPRC (seed)', fontsize=6.2)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'L')

# M 1x2 baseline delta forest
ax = fig.add_subplot(G[4, 3:5])
care_ap = average_precision_score(hY, hsc)
au = {}
for r_ in named:
    au.setdefault(r_['cfg'], []).append(r_['ext1']['AUPRC'])
au = {k_: float(np.mean(v)) for k_, v in au.items()}
names = sorted(au)
deltas = [au[n_] - care_ap for n_ in names]
ys = np.arange(len(names))[::-1]
mcols = ['#7B1FA2' if d_ < 0 else '#D9BFEA' for d_ in deltas]
mhgt = [0.55 if d_ < 0 else 0.30 for d_ in deltas]
for y_, d_, c_, h_ in zip(ys, deltas, mcols, mhgt):
    ax.barh(y_, d_, color=c_, height=h_)
ax.axvline(0, color=K.DARK, lw=0.6)
ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=6)
ax.set_xlabel('holdout AUPRC delta vs DeepCARE (trees = identity oracle)', fontsize=6.0)
lo_, hi_ = ax.get_xlim(); ax.set_xlim(lo_, hi_ + 0.30 * (hi_ - lo_))
ax.legend(handles=[Patch(fc='#7B1FA2', label='below DeepCARE'), Patch(fc='#D9BFEA', label='above DeepCARE')],
          fontsize=4.8, loc='upper right', frameon=False, handlelength=1.2)
K.despine(ax); K.grid_x(ax)
lab(ax, 'M')

# N 1x2 class strip + IQR
ax = fig.add_subplot(G[5, 0:2])
for yi, (vals, color, nm) in enumerate(((pos_s, '#F03A6A', 'clinical'), (neg_s, '#FDB731', 'non-clin.'))):
    ax.scatter(rng.uniform(-0.18, 0.18, len(vals)) + yi, vals, s=9, color=color, alpha=0.55, lw=0)
    q1, qm, q3 = np.percentile(vals, [25, 50, 75])
    ax.plot([yi - 0.3, yi + 0.3], [qm, qm], color=K.DARK, lw=2.4)
    ax.plot([yi - 0.3, yi - 0.3], [q1, q3], color=K.DARK, lw=1.6)
    ax.plot([yi + 0.3, yi + 0.3], [q1, q3], color=K.DARK, lw=1.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(['clinical', 'non-clin.'], fontsize=6)
ax.set_ylabel('z', fontsize=6.5)
ax.set_xlim(-0.6, 1.6)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'N')

# O hits@k
ax = fig.add_subplot(G[5, 2])
o = np.argsort(-hsc)
hits = np.cumsum(hY[o][:120])
ax.plot(np.arange(1, 121), hits, color='#F03A6A', lw=1.5)
ax.fill_between(np.arange(1, 121), np.arange(1, 121) * hY.mean(), hits, color=K.LAV2, alpha=0.25, lw=0)
ax.plot(np.arange(1, 121), np.arange(1, 121) * hY.mean(), color='#FD7E4D', lw=0.8, ls='--')
ax.set_xlabel('top-k', fontsize=6.5); ax.set_ylabel('clinical hits (cum.)', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'O')

# P 1x2 top-10
ax = fig.add_subplot(G[5, 3:5])
o = np.argsort(-hsc)[:10]
yy = np.arange(10)[::-1]
vmin_, vmax_ = hsc[o].min(), hsc[o].max()
for yi, i_ in zip(yy, o):
    is_clin = hY[i_] == 1
    ax.plot([vmin_ - 0.1, hsc[i_]], [yi, yi], color=K.ROSE2 if is_clin else K.LAV2,
            lw=1.8, solid_capstyle='round', zorder=1)
    ax.scatter([hsc[i_]], [yi], s=40 if is_clin else 24,
               color='#F03A6A' if is_clin else '#FDB731', zorder=3,
               ec='white', lw=0.6)
ax.set_yticks(yy)
ax.set_yticklabels([hpid[i_] for i_ in o], fontsize=4.4)
ax.set_xlim(vmin_ - 0.15, vmax_ + 0.05)
ax.set_ylim(-0.7, 9.9)
ax.set_xlabel('holdout z (top-10)', fontsize=6.2)
ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#F03A6A', ms=4, label='clinical'),
                   Line2D([], [], marker='o', ls='', color='#FDB731', ms=4, label='neg')],
          fontsize=4.2, loc='lower right', frameon=False)
ax.tick_params(labelsize=5.0)
K.despine(ax); K.grid_x(ax)
ax.legend(fontsize=4.4, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
lab(ax, 'P')

fig.savefig(OUT / 'Fig2_ext1.png', dpi=600, bbox_inches='tight')
fig.savefig(OUT / 'Fig2_ext1.pdf', bbox_inches='tight')
print('Fig2 v4 done: 16 panels, grid 6x5 full, 3D#2 M+ wall ok')
