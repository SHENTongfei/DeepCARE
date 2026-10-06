# CARE Fig4 v5: ABLATION. 5x5=25, 12 panels A-L. Prof round: B/D bubbles iron-040
# (number IN bubble, size-insensitive, colour-sensitive gradient), C number moved,
# H -> table heatmap, I -> Cleveland dot+whisker, J/K/L gradient bars+seed whiskers.
import os, sys, json, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'C:/Users/TS/Codex/ABRACE/code')
import _carefig as K
K.style()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score
import care_model as CM
from care_judge import cgi_score

ROOT = pathlib.Path(r'C:\Users\TS\Codex\ABRACE')
D = ROOT / 'data'
OUT = ROOT / 'figures' / 'output' / 'assembled_figures'
OUT.mkdir(parents=True, exist_ok=True)

RS = json.load(open(D / 'care_scores_robust.json'))
JR = json.load(open(D / 'care_judgement_robust.json'))
led = json.load(open(D / 'inno_ledger.json'))
rk = json.load(open(D / 'care_cv_results_ranknet.json'))
plain = json.load(open(D / 'care_cv_results_plain.json'))
abl = json.load(open(D / 'care_cv_results_no_fam_no_cgi_no_towers_towers_only_fam_cgi.json'))
raw1 = np.load(D / 'raw_snap_ext1.npy')
y1 = np.array(RS['ext1']['y'])
rng = np.random.default_rng(11)

CFGS = ['full', 'no_fam', 'no_cgi', 'no_towers', 'towers_only', 'fam_cgi']
SHORT = {'full': 'full', 'no_fam': '-fam', 'no_cgi': '-cgi', 'no_towers': '-towers',
         'towers_only': 'towers only', 'fam_cgi': 'fam+cgi'}
full_rows = [r for r in rk if r['fold'] == 0 and r['seed'] in (0, 1, 2, 3)]

def cell(cfg):
    return full_rows if cfg == 'full' else [r for r in abl if r['cfg'] == cfg]

def m(cfg, layer, key):
    return float(np.mean([r[layer][key] for r in cell(cfg)]))

def m_seed(cfg, layer, key):
    return np.array([r[layer][key] for r in cell(cfg)], float)

SPANS = [('A', 0, 2, 0, 2), ('B', 0, 2, 2, 4), ('C', 0, 1, 4, 5), ('D', 1, 2, 4, 5),
         ('E', 2, 4, 0, 2), ('F', 2, 3, 2, 4), ('G', 2, 3, 4, 5), ('H', 3, 4, 2, 4),
         ('I', 3, 4, 4, 5), ('J', 4, 5, 0, 2), ('K', 4, 5, 2, 4), ('L', 4, 5, 4, 5)]
K.grid_check(SPANS, 5, 5)

fig = plt.figure(figsize=(11.2, 13.4))
G = gs.GridSpec(5, 5, figure=fig, hspace=0.58, wspace=0.75)

def lab(ax, L):
    ax.set_title(L, fontweight='bold', fontsize=11, loc='left', color=K.DARK)

# ---- shared: delta colour mapper (iron-040 colour-sensitive, sunsetflare family) ----
NEG_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list('negR', ['#F5B786', '#F03A6A', '#B91E8A'])
POS_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list('posA', ['#FDE725', '#FD9E4D', '#FD7E4D'])

def dcol(d, dn, dp):
    if abs(d) < 0.005:
        return '#E1DFF6'
    if d < 0:
        return NEG_CMAP(min(abs(d) / max(dn, 1e-9), 1.0))
    return POS_CMAP(min(d / max(dp, 1e-9), 1.0))

def txtcol(c):
    if isinstance(c, str):
        from matplotlib.colors import to_rgb
        c = to_rgb(c)
    return 'white' if (c[0] + c[1] + c[2]) < 1.75 else K.DARK

# A 2x2 3D bar3d knockout x seed (int AUPRC), L+ spec
ax = fig.add_subplot(G[0:2, 0:2], projection='3d')
zc = np.array([[r['internal']['AUPRC'] for r in cell(c)] for c in CFGS])
SC3 = ['#3D1E96', '#7B1FA2', '#B91E8A', '#F03A6A']
z0 = zc.min() * 0.96
for ci in range(4):
    for ri in range(6):
        v = zc[ri, ci]
        if v > z0:
            ax.bar3d(ri - 0.18, ci - 0.3, z0, 0.36, 0.6, v - z0, color=SC3[ci], alpha=0.88, zorder=3)
ax.set_xticks(range(6)); ax.set_xticklabels([SHORT[c] for c in CFGS], fontsize=5.2, rotation=25)
ax.set_yticks(range(4)); ax.set_yticklabels(['s0', 's1', 's2', 's3'], fontsize=5.5)
ax.set_zlabel('int AUPRC', fontsize=6)
ax.set_zlim(z0, zc.max() * 1.01)
ax.text2D(0.02, 0.95, f'z starts {z0:.3f} (trunc.)', transform=ax.transAxes, fontsize=5.4, color=K.DARK)
ax.view_init(elev=22, azim=-58)
try:
    ax.set_box_aspect((1.15, 1.0, 0.7), zoom=1.15)
except TypeError:
    ax.set_box_aspect((1.15, 1.0, 0.7))
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_pane_color((0.98, 0.965, 0.99, 1.0))
    a._axinfo['grid'].update(color=K.ROSE2, linewidth=0.3)
ax.tick_params(colors=K.DARK, labelsize=5.2, pad=1)
lab(ax, 'A')

# B 2x2 knockout x metric BUBBLES: number inside, uniform size, colour = delta depth
ax = fig.add_subplot(G[0:2, 2:4])
METS = [('internal', 'AUPRC', 'int\nAUPRC'), ('ext1', 'AUPRC', 'HO\nAUPRC'),
        ('internal', 'EF@10', 'int\nEF@10'), ('ext1', 'EF@10', 'HO\nEF@10')]
DM = np.array([[m(c, ly, key) - m('full', ly, key) for (ly, key, _) in METS] for c in CFGS[1:]])
dn, dp = float(np.abs(DM[DM < 0]).max()), float(DM[DM >= 0].max())
for j in range(len(METS)):
    for i, c in enumerate(CFGS[1:]):
        d_ = DM[i, j]
        col = dcol(d_, dn, dp)
        ax.scatter(j, i, s=590, color=col, lw=0, zorder=3)
        ax.text(j, i, f'{d_:+.2f}', fontsize=5.2, ha='center', va='center',
                color=txtcol(col), zorder=5, fontweight='bold')
        sd_ = m_seed(c, METS[j][0], METS[j][1]) - m_seed('full', METS[j][0], METS[j][1])
        ax.scatter(np.full(len(sd_), j + 0.27), i + np.linspace(-0.09, 0.09, len(sd_)), s=7,
                   color=K.DARK, alpha=0.5, lw=0, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels([mm[2] for mm in METS], fontsize=6)
ax.set_yticks(range(5)); ax.set_yticklabels([SHORT[c] for c in CFGS[1:]], fontsize=6)
ax.set_xlim(-0.55, 3.65); ax.set_ylim(4.7, -0.7)
ax.text(1.5, 5.35, 'colour depth = |delta| (rose down / amber up); side dots = seeds',
        fontsize=4.8, ha='center', color=K.DARK)
K.despine(ax, keep_bottom=False, keep_left=False)
ax.tick_params(length=0)
lab(ax, 'B')

# C towers vs full: two rows with INDEPENDENT linear scales (int ~0.03 vs HO ~0.55, 15x gap)
sgsC = G[0, 4].subgridspec(2, 1, hspace=0.62)
ROWC = [('HO', m('towers_only', 'ext1', 'AUPRC'), m('full', 'ext1', 'AUPRC'),
         m_seed('towers_only', 'ext1', 'AUPRC'), m_seed('full', 'ext1', 'AUPRC')),
        ('int', m('towers_only', 'internal', 'AUPRC'), m('full', 'internal', 'AUPRC'),
         m_seed('towers_only', 'internal', 'AUPRC'), m_seed('full', 'internal', 'AUPRC'))]
for ri, (nm, v, ref, sdt, sdf) in enumerate(ROWC):
    axC = fig.add_subplot(sgsC[ri])
    allv = np.concatenate([sdt, sdf, [v, ref]])
    lo_, hi_ = float(allv.min()), float(allv.max())
    pad_ = (hi_ - lo_) * 0.22 + 1e-6
    axC.plot([v, ref], [0, 0], color=K.ROSE2, lw=2.4)
    axC.scatter([v], [0], s=38, color='#B91E8A', zorder=3, label='towers only' if ri == 0 else '')
    axC.scatter([ref], [0], s=38, color='#F03A6A', zorder=3, label='full' if ri == 0 else '')
    axC.scatter(sdt, np.full(4, 0.17), s=8, color='#B91E8A', alpha=0.6, lw=0, zorder=4)
    axC.scatter(sdf, np.full(4, -0.17), s=8, color='#F03A6A', alpha=0.6, lw=0, zorder=4)
    rel = (v - ref) / ref * 100
    axC.text(hi_ + pad_ * 0.7, 0, f'{rel:+.0f}%', fontsize=5.6, ha='left', va='center',
             color=K.DARK, fontweight='bold')
    axC.set_xlim(lo_ - pad_, hi_ + pad_ * 3.4)
    axC.set_xticks([lo_, hi_])
    axC.set_xticklabels([f'{lo_:.3f}', f'{hi_:.3f}'], fontsize=4.3)
    axC.set_yticks([0]); axC.set_yticklabels([nm], fontsize=6)
    axC.set_ylim(-0.55, 0.55)
    axC.tick_params(labelsize=4.6, length=2)
    K.despine(axC)
    if ri == 0:
        axC.legend(fontsize=4.6, loc='lower center', bbox_to_anchor=(0.5, 1.04), ncol=2,
                   frameon=False)
axC.set_xlabel('AUPRC (row scales independent)', fontsize=5.2)

# D INNO ledger BUBBLES: number inside, uniform size, colour = count depth
ax = fig.add_subplot(G[1, 4])
cats = ['feature-view', 'loss/training', 'ensemble']
cat_of = {}
for r_ in led['rejected']:
    cat_of.setdefault(r_['cat'], []).append(r_)
cnt = []
for c in cats:
    n_rej = len(cat_of.get(c, []))
    n_ad = sum(1 for a_ in led['adopted'] if a_['type'] == c or (c == 'loss/training' and a_['type'] == 'loss'))
    cnt.append((n_rej, n_ad))
nmax = max(max(r for r, _ in cnt), max(a for _, a in cnt))
for yi, ((n_rej, n_ad), c) in zip(np.arange(len(cats))[::-1], zip(cnt, cats)):
    col_r = NEG_CMAP(0.25 + 0.75 * n_rej / max(nmax, 1))
    col_a = POS_CMAP(0.25 + 0.75 * n_ad / max(nmax, 1))
    ax.scatter([0], [yi], s=470, color=col_r, lw=0, zorder=3)
    ax.text(0, yi, str(n_rej), fontsize=6.4, ha='center', va='center',
            color=txtcol(col_r), zorder=5, fontweight='bold')
    ax.scatter([1], [yi], s=470, color=col_a, lw=0, zorder=3)
    ax.text(1, yi, str(n_ad), fontsize=6.4, ha='center', va='center',
            color=txtcol(col_a), zorder=5, fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(['rej.', 'adopt.'], fontsize=6)
ax.set_yticks(np.arange(len(cats))[::-1]); ax.set_yticklabels(['feature', 'loss', 'ens'], fontsize=5.5)
ax.set_xlim(-0.55, 1.55); ax.set_ylim(-0.55, len(cats) - 0.45)
K.despine(ax, keep_bottom=False, keep_left=False)
ax.tick_params(length=0)
lab(ax, 'D')

# E 2x2 40-snapshot strip BROKEN x (snap39 outlier) + per-seed strip
sgs = G[2:4, 0:2].subgridspec(2, 1, hspace=0.5)
aps40 = [average_precision_score(y1, raw1[i]) for i in range(40)]
sg2 = sgs[0].subgridspec(1, 2, width_ratios=[1, 4.0], wspace=0.08)
axL, axR = fig.add_subplot(sg2[0]), fig.add_subplot(sg2[1])
axL.scatter([39], [aps40[39]], s=26, color='#B91E8A', ec=K.DARK, lw=0.6, zorder=5)
axL.set_xlim(37.5, 40.5); axL.set_ylim(0.30, 0.34)
axL.set_xticks([39]); axL.set_yticks([0.317])
axL.set_yticklabels(['.317'])
axR.scatter(np.arange(39) + rng.uniform(-0.35, 0.35, 39), aps40[:39], s=13,
            color='#B91E8A', alpha=0.85, lw=0)
fap = average_precision_score(y1, np.array(RS['ext1']['ens_z']))
axR.scatter([20], [fap], s=44, color='#F03A6A', marker='D', zorder=5)
axR.annotate(f'fused {fap:.3f}', (20, fap), xytext=(25, fap - 0.10), fontsize=4.6,
             color=K.DARK, arrowprops=dict(arrowstyle='-', color=K.DARK, lw=0.5))
axR.set_xlim(-2.5, 40.5); axR.set_ylim(0.38, 0.85)
axR.set_xticks([0, 10, 20, 30, 39])
axR.minorticks_off()
axL.spines['right'].set_visible(False); axR.spines['left'].set_visible(False)
K.brk_marks(axR, [0.995, 1.0], axis='x')
axL.tick_params(labelsize=5.2); axR.tick_params(labelsize=5.2)
axL.set_ylabel('HO AUPRC', fontsize=6)
axR.set_xlabel('snapshot id (left: snap39 diverged, excl.)', fontsize=5.8)
K.despine(axR); K.despine(axL)
lab(axL, 'E')
axs = fig.add_subplot(sgs[1])
sp = [average_precision_score(y1, np.array(z)) for z in RS['ext1']['per_seed_raw_mean'] if z]
rk_ext = {}
for r_ in rk:
    rk_ext.setdefault(r_['seed'], []).append(r_['ext1']['AUPRC'])
for s in range(8):
    axs.plot([s, s], [min(rk_ext[s]), max(rk_ext[s])], color=K.ROSE2, lw=1.4)
    axs.plot([s, s], [0, sp[s]], color=K.LAV2, lw=2.2)
    axs.scatter([s], [sp[s]], s=34, color='#F03A6A', zorder=3)
axs.axhline(float(np.mean(sp)), color='#FD7E4D', lw=1.0, ls='--')
axs.set_title('E (cont.)', fontweight='bold', fontsize=9, loc='left', color=K.DARK)
axs.set_xticks(range(8)); axs.set_xticklabels([f's{i}' for i in range(8)], fontsize=4.8)
axs.set_ylabel('HO AUPRC (seed)', fontsize=6)
axs.set_xlabel('seed (whisker = fold spread)', fontsize=6)
axs.tick_params(labelsize=5.2)
K.despine(axs); axs.grid(False); K.grid_y(axs)

# F claims CI forest (real CIs incl. ens)
dd9 = np.load(D / 'care_dataset_v9.npz', allow_pickle=True)
X1 = dd9['X_external_holdout']
mb1 = CM.compute_metrics(y1, cgi_score(X1))
raw1z = np.stack([(raw1[i] - raw1[i].mean()) / (raw1[i].std() + 1e-9) for i in range(39)])
ov = {}
for met in ('AUPRC', 'EF@10', 'AUROC'):
    bt = [CM.compute_metrics(y1, raw1z[rng.integers(0, 39, 39)].mean(axis=0))[met] - mb1[met]
          for _ in range(200)]
    ov[('ext1_ens', met)] = np.percentile(bt, [2.5, 97.5])
ax = fig.add_subplot(G[2, 2:4])
claims = []
for t in JR['table']:
    if t['layer'] in ('ext1_ens', 'ext1') and t['metric'] in ('AUPRC', 'EF@10', 'AUROC'):
        lo, hi = ov.get((t['layer'], t['metric']), (t['ci_lo'], t['ci_hi']))
        claims.append((f"{t['layer'].replace('ext1', 'HO').replace('_', ' ')} {t['metric']}",
                       t['delta'], lo, hi))
claims = claims[:6]
yy = np.arange(len(claims))[::-1]
for yi, (nm, dv, lo, hi) in zip(yy, claims):
    ax.plot([lo, hi], [yi, yi], color=K.LAV2, lw=2.6, solid_capstyle='round')
    ax.scatter([dv], [yi], s=28, color='#F03A6A', zorder=3)
ax.axvline(0, color=K.DARK, lw=0.6, ls='--')
ax.set_yticks(yy); ax.set_yticklabels([c[0] for c in claims], fontsize=5)
ax.set_xlabel('delta vs strongest baseline (95% CI)', fontsize=6.2)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'F')

# G RankNet vs plain slopes
ax = fig.add_subplot(G[2, 4])
pk = [float(np.mean([r['ext1']['AUPRC'] for r in plain if r['seed'] == s])) for s in range(8)]
rkp = [float(np.mean([r['ext1']['AUPRC'] for r in rk if r['seed'] == s])) for s in range(8)]
for s in range(8):
    ax.plot([0, 1], [pk[s], rkp[s]], color=K.ROSE2, lw=1.3, marker='o', ms=3.0,
            mfc='#B91E8A', mec='none')
ax.set_xticks([0, 1]); ax.set_xticklabels(['plain', 'RankNet'], fontsize=5.5)
ax.set_ylabel('HO AUPRC', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'G')

# H leave-one-snapshot-out ensemble stability (39 healthy; supports 40-snap adoption)
ax = fig.add_subplot(G[3, 2:4])
loo = np.array([average_precision_score(y1, raw1z[np.arange(39) != i].mean(axis=0))
                for i in range(39)])
xs = np.arange(39)
ax.fill_between([-1.5, 39.5], loo.min(), loo.max(), color=K.LAV2, alpha=0.22, lw=0, zorder=1)
ax.plot(xs, loo, color='#B91E8A', lw=1.0, marker='o', ms=2.6, zorder=3)
ax.axhline(fap, color='#F03A6A', lw=1.2, ls='--', zorder=2)
ax.text(38.6, fap + 0.0008, f'fused {fap:.3f}', fontsize=4.8, ha='right',
        va='bottom', color='#F03A6A')
ax.set_ylim(loo.min() - 0.004, loo.max() + 0.010)
ax.text(-1.0, loo.max() + 0.0075, f'LOO range {loo.min():.3f} - {loo.max():.3f}',
        fontsize=4.9, ha='left', va='center', color=K.DARK)
ax.set_xlim(-1.5, 39.5)
ax.set_xlabel('left-out healthy snapshot id', fontsize=6.2)
ax.set_ylabel('HO AUPRC (38-snap ensemble)', fontsize=6.2)
ax.tick_params(labelsize=5.2)
K.despine(ax); K.grid_y(ax)
lab(ax, 'H')

# I int EF@10 -> Cleveland dot + seed whisker (full pinned on top)
ax = fig.add_subplot(G[3, 4])
vv = [(SHORT[c], m(c, 'internal', 'EF@10'), m_seed(c, 'internal', 'EF@10')) for c in CFGS]
rest = sorted([v for v in vv if v[0] != 'full'], key=lambda t: -t[1])
vv_o = [v for v in vv if v[0] == 'full'] + rest
n_ = len(vv_o)
for yi, (nm, v, sd_) in enumerate(vv_o):
    yj = n_ - 1 - yi  # full pinned on top
    ax.plot([sd_.min(), sd_.max()], [yj, yj], color=K.LAV2, lw=2.4, zorder=1)
    col = '#F03A6A' if nm == 'full' else '#B91E8A'
    ax.scatter([v], [yj], s=42, color=col, zorder=4, ec='white', lw=0.7)
    ax.text(v, yj + 0.34, f'{v:.1f}', fontsize=4.8, ha='center', color=K.DARK)
ax.set_yticks(range(n_))
ax.set_yticklabels([t[0] for t in vv_o[::-1]], fontsize=4.6)
ax.set_xlabel('internal EF@10 (whisker = seeds)', fontsize=5.6)
ax.tick_params(labelsize=5.0)
K.despine(ax); K.grid_x(ax)
lab(ax, 'I')

# J int AUROC deltas: gradient bars + seed whiskers
ax = fig.add_subplot(G[4, 0:2])
dl = [(c, m(c, 'internal', 'AUROC') - m('full', 'internal', 'AUROC')) for c in CFGS[1:]]
dnJ = max(abs(d) for _, d in dl if d < 0) if any(d < 0 for _, d in dl) else 1e-9
dpJ = max(d for _, d in dl) if any(d >= 0 for _, d in dl) else 1e-9
ys_ = np.arange(len(dl))[::-1]
for yi, (c, d) in zip(ys_, dl):
    sd_ = m_seed(c, 'internal', 'AUROC') - m_seed('full', 'internal', 'AUROC')
    ax.plot([min(0, d), max(0, d)], [yi, yi], color=dcol(d, dnJ, dpJ), lw=6.5,
            solid_capstyle='round', alpha=0.92, zorder=2)
    ax.plot([sd_.min(), sd_.max()], [yi, yi], color=K.DARK, lw=1.0, alpha=0.5, zorder=3)
    ax.scatter([d], [yi], s=52, marker='D', color=dcol(d, dnJ, dpJ), ec='white',
               lw=0.6, zorder=4)
ax.axvline(0, color=K.DARK, lw=0.6)
ax.set_yticks(ys_); ax.set_yticklabels([SHORT[c] for c, _ in dl], fontsize=5.5)
ax.set_xlabel('int AUROC delta (diamond = mean, line = seed range)', fontsize=6.2)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'J')

# K HO AUPRC deltas: gradient bars + seed whiskers
ax = fig.add_subplot(G[4, 2:4])
dl = [(c, m(c, 'ext1', 'AUPRC') - m('full', 'ext1', 'AUPRC')) for c in CFGS[1:]]
dnK = max(abs(d) for _, d in dl if d < 0) if any(d < 0 for _, d in dl) else 1e-9
dpK = max(d for _, d in dl) if any(d >= 0 for _, d in dl) else 1e-9
ys_ = np.arange(len(dl))[::-1]
for yi, (c, d) in zip(ys_, dl):
    sd_ = m_seed(c, 'ext1', 'AUPRC') - m_seed('full', 'ext1', 'AUPRC')
    ax.plot([min(0, d), max(0, d)], [yi, yi], color=dcol(d, dnK, dpK), lw=6.5,
            solid_capstyle='round', alpha=0.92, zorder=2)
    ax.plot([sd_.min(), sd_.max()], [yi, yi], color=K.DARK, lw=1.0, alpha=0.5, zorder=3)
    ax.scatter([d], [yi], s=52, marker='D', color=dcol(d, dnK, dpK), ec='white',
               lw=0.6, zorder=4)
ax.axvline(0, color=K.DARK, lw=0.6)
ax.set_yticks(ys_); ax.set_yticklabels([SHORT[c] for c, _ in dl], fontsize=5.5)
ax.set_xlabel('HO AUPRC delta (diamond = mean, line = seed range)', fontsize=6.2)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'K')

# L int AUPRC deltas: gradient bars + seed whiskers
ax = fig.add_subplot(G[4, 4])
dl = [(c, m(c, 'internal', 'AUPRC') - m('full', 'internal', 'AUPRC')) for c in CFGS[1:]]
dnL = max(abs(d) for _, d in dl if d < 0) if any(d < 0 for _, d in dl) else 1e-9
dpL = max(d for _, d in dl) if any(d >= 0 for _, d in dl) else 1e-9
ys_ = np.arange(len(dl))[::-1]
for yi, (c, d) in zip(ys_, dl):
    sd_ = m_seed(c, 'internal', 'AUPRC') - m_seed('full', 'internal', 'AUPRC')
    ax.plot([min(0, d), max(0, d)], [yi, yi], color=dcol(d, dnL, dpL), lw=6.5,
            solid_capstyle='round', alpha=0.92, zorder=2)
    ax.plot([sd_.min(), sd_.max()], [yi, yi], color=K.DARK, lw=1.0, alpha=0.5, zorder=3)
    ax.scatter([d], [yi], s=46, marker='D', color=dcol(d, dnL, dpL), ec='white',
               lw=0.6, zorder=4)
ax.axvline(0, color=K.DARK, lw=0.6)
ax.set_yticks(ys_); ax.set_yticklabels([SHORT[c] for c, _ in dl], fontsize=4.6)
ax.set_xlabel('int AUPRC delta', fontsize=6.2)
ax.tick_params(labelsize=5.0)
K.despine(ax); K.grid_x(ax)
lab(ax, 'L')

fig.savefig(OUT / 'Fig4_ablation.png', dpi=600, bbox_inches='tight')
fig.savefig(OUT / 'Fig4_ablation.pdf', bbox_inches='tight')
print('Fig4 v5: B/D iron-040 bubbles, C delta above, H table heatmap, I Cleveland, J/K/L gradient')
