# CARE FigS1 v3 (supplementary): EXTENDED COMPARISONS & DISCLOSURE. 10 panels A-J, 5x5=25.
# Letters row-major (iron 047). Ring arc (G), 3D bar array (B, iron 051), no text-graph overlap.
import os, sys, json, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'C:/Users/TS/Codex/ABRACE/code')
import _carefig as K
K.style()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import care_model as CM
from care_judge import cgi_score
from sklearn.metrics import average_precision_score
from matplotlib.patches import Polygon as Poly
import matplotlib.patheffects as pe

ROOT = pathlib.Path(r'C:\Users\TS\Codex\ABRACE')
D = ROOT / 'data'
OUT = ROOT / 'figures' / 'output' / 'assembled_figures'
OUT.mkdir(parents=True, exist_ok=True)

RS = json.load(open(D / 'care_scores_robust.json'))
JR = json.load(open(D / 'care_judgement_robust.json'))
rk = json.load(open(D / 'care_cv_results_ranknet.json'))
abl = json.load(open(D / 'care_cv_results_no_fam_no_cgi_no_towers_towers_only_fam_cgi.json'))
named = json.load(open(D / 'care_cv_results_domain_named.json'))
led = json.load(open(D / 'inno_ledger.json'))
dd = np.load(D / 'care_dataset_v9.npz', allow_pickle=True)
rng = np.random.default_rng(31)

Y = np.array(RS['internal']['y'])
y1 = np.array(RS['ext1']['y'])
z1 = np.array(RS['ext1']['ens_z'])
CFGS = ['no_fam', 'no_cgi', 'no_towers', 'towers_only', 'fam_cgi']
SHORT = {'no_fam': '-fam', 'no_cgi': '-cgi', 'no_towers': '-towers',
         'towers_only': 'towers only', 'fam_cgi': 'fam+cgi'}
full_rows = [r for r in rk if r['fold'] == 0 and r['seed'] in (0, 1, 2, 3)]

def cell(cfg):
    return full_rows if cfg == 'full' else [r for r in abl if r['cfg'] == cfg]

def m_seed(cfg, layer, key):
    return np.array([r[layer][key] for r in cell(cfg)], float)

def lab(ax, L):
    ax.set_title(L, fontweight='bold', fontsize=11, loc='left', color=K.DARK)

SPANS = [('A', 0, 2, 0, 2), ('B', 0, 2, 2, 4), ('C', 0, 2, 4, 5), ('D', 2, 3, 0, 1),
         ('E', 2, 3, 1, 2), ('F', 2, 3, 2, 4), ('G', 2, 3, 4, 5), ('H', 3, 5, 0, 2),
         ('I', 3, 5, 2, 4), ('J', 3, 5, 4, 5)]
K.grid_check(SPANS, 5, 5)

fig = plt.figure(figsize=(11.2, 13.4))
G = gs.GridSpec(5, 5, figure=fig, hspace=0.55, wspace=0.7)

def _pol(r, deg):
    t = np.radians(deg)
    return r * np.cos(t), r * np.sin(t)

# ---------- A 2x2: internal per-seed AUPRC delta lollipops (7/8 above baseline) ----------
ax = fig.add_subplot(G[0:2, 0:2])
Xi = dd['X_internal']
mbc = CM.compute_metrics(Y, cgi_score(Xi))
dA = []
for s in range(8):
    a = np.array(RS['internal']['per_seed_raw'][str(s)], float)
    ok = ~np.isnan(a)
    dA.append(CM.auprc(Y[ok], a[ok]) - mbc['AUPRC'])
dA = np.array(dA)
yy = np.arange(8)[::-1]
ax.hlines(yy, 0, dA, color=K.ROSE2, lw=2.2)
ax.scatter(dA, yy, s=42, color=[K.SAND if v > 0 else K.ROSE for v in dA],
           ec='white', lw=0.6, zorder=3)
for yi, v in zip(yy, dA):
    ax.text(v + 0.0015, yi, f'{v:+.3f}', fontsize=4.6, va='center', color=K.DARK)
ax.axvline(0, color=K.DARK, lw=0.7)
t_int = next(t for t in JR['table'] if t['layer'] == 'internal' and t['metric'] == 'AUPRC')
ax.text(0.03, 0.96, f"pooled +{t_int['delta']:.4f}, CI [{t_int['ci_lo']:.4f}, {t_int['ci_hi']:.4f}]",
        transform=ax.transAxes, fontsize=5.2, ha='left', va='top', color=K.DARK,
        bbox=dict(fc='white', ec=K.ROSE2, lw=0.5, alpha=0.9, pad=1.2))
ax.set_yticks(yy); ax.set_yticklabels([f's{i}' for i in range(8)], fontsize=6)
ax.set_xlabel('int AUPRC delta vs cgi35 (7/8 seeds above)', fontsize=6.3)
ax.set_xlim(min(dA.min() * 1.4, -0.004), dA.max() * 1.18)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'A')

# ---------- B 2x2 3D: identity-oracle array (4 named models x 2 layers), L+ spec ----------
ax = fig.add_subplot(G[0:2, 2:4], projection='3d')
mods = ['RandForest', 'XGBoost', 'DeepAllergenPair', 'LogReg']
ZM = np.zeros((4, 2))
for i, m_ in enumerate(mods):
    rows = [r for r in named if r['cfg'] == m_]
    ZM[i, 0] = float(np.mean([r['internal']['AUPRC'] for r in rows]))
    ZM[i, 1] = float(np.mean([r['ext1']['AUPRC'] for r in rows]))
SC3 = ['#3D1E96', '#7B1FA2', '#B91E8A', '#F03A6A']
z0 = ZM.min() * 0.96
for li in range(2):
    for mi in range(4):
        v = ZM[mi, li]
        ax.bar3d(mi - 0.3, li - 0.28, z0, 0.6, 0.56, max(v - z0, 1e-4),
                 color=SC3[mi], alpha=0.88 if li == 0 else 0.62, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels(['RF', 'XGB', 'DAP', 'LR'], fontsize=5.4)
ax.set_yticks(range(2)); ax.set_yticklabels(['int', 'HO'], fontsize=5.5)
ax.set_zlabel('AUPRC', fontsize=6)
ax.set_zlim(z0, ZM.max() * 1.01)
ax.text2D(0.02, 0.95, f'z starts {z0:.3f} (trunc.)', transform=ax.transAxes,
          fontsize=5.2, color=K.DARK)
ax.view_init(elev=20, azim=-56)
try:
    ax.set_box_aspect((1.1, 0.75, 0.7), zoom=1.18)
except TypeError:
    ax.set_box_aspect((1.1, 0.75, 0.7))
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_pane_color((0.98, 0.965, 0.99, 1.0))
    a._axinfo['grid'].update(color=K.ROSE2, linewidth=0.3)
ax.tick_params(colors=K.DARK, labelsize=5.2, pad=1)
lab(ax, 'B')

# ---------- C 2x1 tall: NNK two-bar with value labels ----------
ax = fig.add_subplot(G[0:2, 4])
t_n = next(t for t in JR['table'] if t['layer'] == 'internal_ens' and t['metric'] == 'NNK')
ax.bar([0, 1], [t_n['model_mean'], t_n['base_mean']], width=0.55,
       color=[K.ROSE, K.LAV2], alpha=0.9)
ax.text(0, t_n['model_mean'] * 1.03, f"{t_n['model_mean']:.0f}", ha='center', fontsize=5.4, color=K.DARK)
ax.text(1, t_n['base_mean'] * 1.03, f"{t_n['base_mean']:.0f}", ha='center', fontsize=5.4, color=K.DARK)
ax.set_xticks([0, 1]); ax.set_xticklabels(['CARE', t_n['baseline']], fontsize=6)
ax.set_ylabel('int NNK (lower = better)', fontsize=6.3)
ax.set_ylim(0, t_n['model_mean'] * 1.2)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'C')

# ---------- D 1x1: calibration losses grouped ----------
ax = fig.add_subplot(G[2, 0])
t_e = next(t for t in JR['table'] if t['layer'] == 'internal' and t['metric'] == 'ECE10')
t_b = next(t for t in JR['table'] if t['layer'] == 'internal' and t['metric'] == 'Brier')
xx = np.arange(2)
ax.bar(xx - 0.19, [t_e['model_mean'], t_b['model_mean']], width=0.36, color=K.ROSE, label='CARE')
ax.bar(xx + 0.19, [t_e['base_mean'], t_b['base_mean']], width=0.36, color=K.LAV2, label=t_e['baseline'])
ax.set_xticks(xx); ax.set_xticklabels(['ECE10', 'Brier'], fontsize=5.8)
ax.set_ylabel('int value', fontsize=6.0)
ax.legend(fontsize=3.9, loc='upper left', frameon=False)
ax.tick_params(labelsize=5.0)
K.despine(ax); K.grid_y(ax)
lab(ax, 'D')

# ---------- E 1x1: knockout per-seed strip ----------
ax = fig.add_subplot(G[2, 1])
for yi, c in enumerate(CFGS):
    sd = m_seed(c, 'ext1', 'AUPRC') - m_seed('full', 'ext1', 'AUPRC')
    ax.scatter(sd, np.full(4, yi), s=18, color=K.ROSE if sd.mean() < 0 else K.SAND,
               alpha=0.85, lw=0, zorder=3)
ax.axvline(0, color=K.DARK, lw=0.7)
ax.set_yticks(range(len(CFGS)))
ax.set_yticklabels([SHORT[c] for c in CFGS], fontsize=4.4)
ax.set_xlabel('HO AUPRC delta', fontsize=6.0)
ax.tick_params(labelsize=5.0)
K.despine(ax); K.grid_x(ax)
lab(ax, 'E')

# ---------- F 1x2: ext3 P4 honest histogram ----------
ax = fig.add_subplot(G[2, 2:4])
E3 = RS['ext3']
med = np.array(E3['ens_z_median'])
y3 = np.array(E3['y'])
pos_idx = np.where(y3 == 1)[0]
p4 = med[pos_idx[3]]
ax.hist(med[y3 == 0], bins=12, color=K.LAV2, alpha=0.75, label='neg (n=18)')
ax.axvline(p4, color=K.ROSE, lw=1.6, label=f'P4 {p4:.2f}')
for k in range(3):
    ax.axvline(med[pos_idx[k]], color=K.NAVY, lw=1.2, ls='--', alpha=0.8)
ax.text(0.99, 0.85, 'navy dashes = P1-P3', transform=ax.transAxes, fontsize=4.6,
        ha='right', color=K.DARK)
ax.set_xlabel('block z', fontsize=6.3)
ax.set_ylabel('rows', fontsize=6.3)
ax.legend(fontsize=4.6, loc='upper left', frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'F')

# ---------- G 1x1 RING: snapshot pool 39 kept / 1 excluded ----------
ax = fig.add_subplot(G[2, 4])
segsG = [('kept 39', 39, K.LAV2), ('excl. 1', 1, K.ROSE)]
totG = sum(v for _, v, _ in segsG)
avo = 300.0 - 10.0 * len(segsG)
ao = -240.0
for nm, v, col in segsG:
    wo = avo * v / totG
    th = np.linspace(ao, ao + wo, 48)
    pts = [_pol(1.0, t) for t in th] + [_pol(0.52, t) for t in th[::-1]]
    ax.add_patch(Poly(pts, closed=True, fc=col, ec='white', lw=1.0))
    ao += wo + 10.0
ax.text(0, 0, '40\nsnaps', fontsize=6.5, ha='center', va='center', color=K.DARK)
ax.text(_pol(1.24, -90)[0], _pol(1.24, -90)[1], 'kept 39', fontsize=5.0,
        ha='center', color=K.DARK)
ax.text(_pol(1.24, 60)[0], _pol(1.24, 60)[1], 'excl. 1', fontsize=5.0,
        ha='center', color=K.DARK)
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
ax.axis('off')
lab(ax, 'G')

# ---------- H 2x2: screening dumbbells (all recorded screens) ----------
ax = fig.add_subplot(G[3:5, 0:2])
scr = led['screens']
rows_ = []
for s_ in scr:
    rows_.append((s_['name'] + ' TI', s_['TI'], s_['refTI'], s_['verdict']))
    rows_.append((s_['name'] + ' HO', s_['HO'], s_['refHO'], s_['verdict']))
yy = np.arange(len(rows_))[::-1]
for yi, (nm, v, ref, vd) in zip(yy, rows_):
    col = K.NAVY if vd == 'PASS' else K.ROSE
    ax.plot([min(v, ref), max(v, ref)], [yi, yi], color=K.ROSE2, lw=1.8)
    ax.scatter([v], [yi], s=28, color=col, zorder=3)
    ax.scatter([ref], [yi], s=30, facecolor='none', edgecolor=K.SAND, lw=1.1, zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows_], fontsize=5.2)
ax.set_xlabel('AUPRC (dot = screen, ring = ref; navy = pass, rose = fail)', fontsize=6.0)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'H')

# ---------- I 2x2: internal AUROC per-seed deltas (strip + mean line) ----------
ax = fig.add_subplot(G[3:5, 2:4])
t_a = next(t for t in JR['table'] if t['layer'] == 'internal' and t['metric'] == 'AUROC')
mba = CM.compute_metrics(Y, cgi_score(Xi))['AUROC']
sdI = []
for s in range(8):
    a = np.array(RS['internal']['per_seed_raw'][str(s)], float)
    ok = ~np.isnan(a)
    sdI.append(CM.compute_metrics(Y[ok], a[ok])['AUROC'] - mba)
sdI = np.array(sdI)
jit = rng.uniform(-0.14, 0.14, 8)
ax.scatter(sdI, np.arange(8) + jit, s=40, color=np.where(sdI > 0, K.SAND, K.ROSE),
           alpha=0.9, lw=0, zorder=3)
meanI = float(sdI.mean())
ax.axvline(meanI, color='#3D1E96', lw=1.4, ls='--')
ax.axvline(0, color=K.DARK, lw=0.7)
ax.text(meanI - 0.001, 7.6, f'mean {meanI:+.3f}', fontsize=4.8, ha='right', color='#3D1E96')
for s_, v in zip(range(8), sdI):
    ax.text(v - 0.002, s_ + jit[list(range(8)).index(s_)] if False else s_, f'{v:+.3f}',
            fontsize=4.4, va='center', ha='right', color=K.DARK)
ax.set_yticks(range(8)); ax.set_yticklabels([f's{i}' for i in range(8)], fontsize=5.5)
ax.set_xlabel(f"int AUROC delta vs cgi35 (pooled {t_a['delta']:+.3f})", fontsize=6.2)
ax.set_xlim(min(sdI.min() * 1.35, -0.004), sdI.max() * 1.25)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'I')

# ---------- J 2x1 tall: per-seed ECE10 delta (honest calibration disclosure) ----------
ax = fig.add_subplot(G[3:5, 4])
ece_base = t_e = next(t for t in JR['table'] if t['layer'] == 'internal' and t['metric'] == 'ECE10')['base_mean']
ece = []
for s in range(8):
    a = np.array(RS['internal']['per_seed_raw'][str(s)], float)
    ok = ~np.isnan(a)
    ece.append(CM.compute_metrics(Y[ok], a[ok])['ECE10'] - ece_base)
ece = np.array(ece)
yy = np.arange(8)[::-1]
ax.barh(yy, ece, color=K.ROSE, alpha=0.75, height=0.55)
for yi, v in zip(yy, ece):
    ax.text(v - 0.004, yi, f'{v:+.3f}', fontsize=4.2, va='center', ha='right', color=K.DARK)
ax.axvline(0, color=K.DARK, lw=0.7)
ax.set_yticks(yy); ax.set_yticklabels([f's{i}' for i in range(8)], fontsize=5.5)
ax.set_xlabel('int ECE10 delta vs cgi35', fontsize=6.2)
ax.set_xlim(ece.min() * 1.25, ece.max() * 1.6 + 0.01)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'J')

fig.savefig(OUT / 'FigS1_extended.png', dpi=600, bbox_inches='tight')
fig.savefig(OUT / 'FigS1_extended.pdf', bbox_inches='tight')
print('FigS1 v3 done (10 panels A-J, sum=25, ring G + 3D B)')
