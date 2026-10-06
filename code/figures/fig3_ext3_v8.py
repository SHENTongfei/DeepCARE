# CARE Fig3 v8: ext3 ONLY. 5x5=25, 11 panels A-K. L DELETED, B ring heatmap 2x3 ENLARGED,
# radial labels via SOP iron-041 standard (start=-300,end=0 gap at 3 o'clock, yticks side=right).
# Letters row-major (iron 047). CLEAN SINGLE WRITE (iron 052).
import os, sys, json, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'C:/Users/TS/Codex/ABRACE/code')
import _carefig as K
K.style()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import care_model as CM  # noqa
from care_judge import fam_prior_score

ROOT = pathlib.Path(r'C:\Users\TS\Codex\ABRACE')
D = ROOT / 'data'
OUT = ROOT / 'figures' / 'output' / 'assembled_figures'
OUT.mkdir(parents=True, exist_ok=True)

RS = json.load(open(D / 'care_scores_robust.json'))
E3 = RS['ext3']
pid3 = [str(p) for p in E3['pid']]
y3 = np.array(E3['y'])
snap = np.array(E3['snap_z'])
med = np.array(E3['ens_z_median'])
pos_idx = np.where(y3 == 1)[0]
pairs4 = [pid3[i] for i in pos_idx]
rng = np.random.default_rng(7)
groups = {}
for i in pos_idx:
    a, b = pid3[i].split('|')
    groups[pid3[i]] = [j for j, p in enumerate(pid3)
                       if y3[j] == 0 and (a in p.split('|') or b in p.split('|'))]
PCOL = {'Cor a 1|Cor a 12': '#3D1E96', 'Cor a 13|Cor a 2': '#B91E8A',
        'Tri a 12|Tri a 29': '#F03A6A', 'Tri a 30|Tri a 37': '#FD7E4D'}
dd = np.load(D / 'care_dataset_v9.npz', allow_pickle=True)
X3 = dd['X_ext3']

SPANS = [('A', 0, 2, 0, 2), ('B', 0, 2, 2, 5), ('C', 2, 3, 0, 1), ('D', 2, 3, 1, 2),
         ('E', 2, 3, 2, 4), ('F', 2, 3, 4, 5), ('G', 3, 5, 0, 2), ('H', 3, 4, 2, 4),
         ('I', 3, 4, 4, 5), ('J', 4, 5, 2, 4), ('K', 4, 5, 4, 5)]
K.grid_check(SPANS, 5, 5)

fig = plt.figure(figsize=(11.2, 13.4))
G = gs.GridSpec(5, 5, figure=fig, hspace=0.58, wspace=0.72,
                height_ratios=[1.16, 1.16, 1, 1, 1])

def lab(ax, L):
    ax.set_title(L, fontweight='bold', fontsize=11, loc='left', color=K.DARK)

# ============ A 2x2: horizontal raincloud per pair (X = snapshot z value, Y = pair row) ============
ax = fig.add_subplot(G[0:2, 0:2])
yy = np.arange(4)[::-1]
for k, i in enumerate(pos_idx):
    yk = yy[k]
    vals = np.sort(snap[:, i])
    kde_x = np.linspace(vals.min() - 0.2, vals.max() + 0.2, 140)
    dens = gaussian_kde(vals)(kde_x)
    dens = dens / dens.max() * 0.17
    ax.fill_between(kde_x, yk + 0.08, yk + 0.08 + dens, color=PCOL[pairs4[k]], alpha=0.40, lw=0)
    ax.plot(kde_x, yk + 0.08 + dens, color=PCOL[pairs4[k]], lw=1)
    jit = rng.uniform(-0.055, 0.055, 39)
    ax.scatter(vals, np.full(39, yk) + jit, s=9, color=PCOL[pairs4[k]], alpha=0.6, lw=0, zorder=3)
    q1, qm, q3 = np.percentile(vals, [25, 50, 75])
    ax.plot([qm, qm], [yk - 0.30, yk - 0.10], color=K.DARK, lw=2.6, solid_capstyle='butt')
    ax.plot([q1, q3], [yk - 0.20, yk - 0.20], color=K.DARK, lw=1.0)
ax.set_ylim(-0.55, 3.62)
ax.set_yticks(yy)
ax.set_yticklabels([p.replace('|', ' | ') for p in pairs4], fontsize=6)
lo = float(min(snap[:, i].min() for i in pos_idx)) - 0.35
hi = float(max(snap[:, i].max() for i in pos_idx)) + 0.35
ax.set_xlim(lo, hi)
ax.set_xlabel('snapshot z', fontsize=6.5)
ax.tick_params(axis='x', labelsize=5.5)
K.despine(ax)
lab(ax, 'A')

# ============ B 2x3 ENLARGED: ring heatmap (SOP iron-041 standard) ============
from pycirclize import Circos
Lm = np.stack([snap[:, i] for i in pos_idx])          # (4 rings, 39 sectors)
Lmn = np.zeros_like(Lm)
for ri in range(Lm.shape[0]):
    rmin, rmax = Lm[ri].min(), Lm[ri].max()
    Lmn[ri] = (Lm[ri] - rmin) / max(rmax - rmin, 1e-9)
CMAP_B = plt.cm.colors.LinearSegmentedColormap.from_list('sfB',
    ['#FDE725', '#FD7E4D', '#B91E8A', '#7B1FA2', '#3D1E96'])
n_secB, n_ringB = 39, 4
sec_labB = [f'S{i+1}' for i in range(n_secB)]
circB = Circos({sec_labB[i]: n_ringB for i in range(n_secB)}, start=-300, end=0, space=1.0)
for i, sector in enumerate(circB.sectors):
    sector.text(sector.name, r=100.5, size=6.0, color=K.DARK,
                adjust_rotation=True, orientation='vertical')
    trB = sector.add_track((14, 99))
    trB.axis(ec='white', lw=0.3)
    trB.heatmap(Lmn[:, i].reshape(-1, 1), vmin=0, vmax=1, cmap=CMAP_B,
                rect_kws=dict(ec='white', lw=0.25))
    if i == n_secB - 1:  # radial axis labels INTO the gap at 3 o'clock (iron 041)
        trB.yticks([r + 0.5 for r in range(n_ringB)], ['P4', 'P3', 'P2', 'P1'],
                   vmin=0, vmax=n_ringB, side='right', tick_length=2.0,
                   label_size=14, label_margin=0.15)
figB = circB.plotfig(dpi=300)
figB.set_size_inches(8.4, 8.4)
figB.axes[0].set_position([0.0, 0.0, 1.0, 1.0])  # polar fills canvas -> ring diameter max
import io as _io
_bufB = _io.BytesIO()
figB.savefig(_bufB, format='png', dpi=300, bbox_inches='tight', pad_inches=0.02,
             facecolor='white')
plt.close(figB)
_bufB.seek(0)
from PIL import Image as _ImB
_arrB = np.asarray(_ImB.open(_bufB).convert('RGB'))
_nzB = np.where(_arrB.min(axis=2) < 245)
ax = fig.add_subplot(G[0:2, 2:5])
ax.imshow(_arrB[int(_nzB[0].min()):int(_nzB[0].max()) + 1,
                int(_nzB[1].min()):int(_nzB[1].max()) + 1], aspect='equal')
ax.axis('off')
lab(ax, 'B')
# colourbar OUT of the ring render: horizontal, standalone row in the band below B
posB = ax.get_position()
caxM = fig.add_axes([posB.x0 + 0.015, posB.y0 - 0.0225, posB.width * 0.97, 0.0045])
cbM = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_B,
                   norm=plt.cm.colors.Normalize(vmin=0, vmax=1)),
                   cax=caxM, orientation='horizontal')
cbM.set_ticks([0, 1]); cbM.set_ticklabels(['low', 'high'])
cbM.ax.tick_params(labelsize=6.5, colors=K.DARK)
cbM.outline.set_visible(False)
fig.text(posB.x0 + posB.width * 0.5, posB.y0 - 0.0305, 'snapshot z (norm.)',
         fontsize=6.8, color=K.DARK, ha='center', va='top')

# ============ C 1x1: block ECDF ============
ax = fig.add_subplot(G[2, 0])
xs_g = np.linspace(med.min() - 0.4, med.max() + 0.4, 240)
ax.fill_between(xs_g, 0, K.kecdf(xs_g, med), color=K.LAV2, alpha=0.18, lw=0)
ax.plot(xs_g, K.kecdf(xs_g, med), color='#B91E8A', lw=1.3)
for k, i in enumerate(pos_idx):
    ax.scatter([med[i]], [float((med < med[i]).mean())], s=32, color=PCOL[pairs4[k]],
               ec='white', lw=0.8, zorder=5)
ax.set_xlabel('block z (n=22)', fontsize=6.5)
ax.set_ylabel('ECDF', fontsize=6.5)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'C')

# ============ D 1x1: ranked dot 22 rows ============
ax = fig.add_subplot(G[2, 1])
rank_o = np.argsort(med)
for k_, i_ in enumerate(rank_o):
    yi = 21 - k_
    is_clin = y3[i_] == 1
    col = PCOL.get(pid3[i_], '#B91E8A') if is_clin else K.ROSE2
    ax.scatter([med[i_]], [yi], s=30 if is_clin else 16, color=col,
               marker='D' if is_clin else 'o', zorder=3,
               linewidth=0.5, edgecolor='white' if is_clin else 'none')
ax.set_yticks([21, 11, 1])
ax.set_yticklabels(['#22', '#12', '#2'], fontsize=4.6)
ax.set_ylabel('rank (low z -> high z)', fontsize=5.8)
ax.set_xlabel('block z', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'D')

# ============ E 1x2: CGI coverage ============
ax = fig.add_subplot(G[2, 2:4])
mem_hit, mem_all = set(), set()
for i in range(len(pid3)):
    for m_ in pid3[i].split('|'):
        mem_all.add(m_)
        if X3[i, 4] > 0:
            mem_hit.add(m_)
n_hit_rows = int((X3[:, 4] >= 0).sum())
bars = [('members', len(mem_hit), len(mem_all)), ('rows', n_hit_rows, 22)]
xx = np.arange(2)
for xi, (nm, hit, tot) in enumerate(bars):
    ax.barh(xi, hit, color='#F03A6A', height=0.4)
    ax.barh(xi, tot - hit, left=hit, color=K.ROSE2, alpha=0.25, height=0.4)
    ax.text(tot + 0.4, xi, f'{hit}/{tot}', fontsize=5.2, va='center', color=K.DARK)
ax.set_yticks(xx); ax.set_yticklabels([b[0] for b in bars], fontsize=6)
ax.set_yticks(xx); ax.set_yticklabels([b[0] for b in bars], fontsize=6)
ax.set_xlim(0, 27)
ax.set_xlabel('with CGI hit', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'E')

# ============ F 1x1: slopegraph ============
ax = fig.add_subplot(G[2, 4])
for p in pairs4[:3]:
    mem = groups[p]
    if not mem:
        continue
    i = pos_idx[list(pairs4).index(p)]
    neg_med = float(np.median(med[mem]))
    ax.plot([0, 1], [med[i], neg_med], color=PCOL[p], lw=1.8, marker='o', ms=4,
            mec='white', zorder=3)
ax.set_xlim(-0.3, 1.3)
ax.set_xticks([0, 1]); ax.set_xticklabels(['DeepCARE', 'neg\nmed'], fontsize=5.0)
ax.set_ylabel('z', fontsize=6.0)
ax.tick_params(labelsize=4.6)
K.despine(ax)
lab(ax, 'F')

# ============ G 2x2: trajectories ============
ax = fig.add_subplot(G[3:5, 0:2])
for k, i in enumerate(pos_idx):
    ax.plot(np.arange(39), snap[:, i], color=PCOL[pairs4[k]], lw=0.8, alpha=0.55)
    ax.scatter(np.arange(39), snap[:, i], s=6, color=PCOL[pairs4[k]], alpha=0.6, lw=0)
    ax.axhline(med[i], color=PCOL[pairs4[k]], lw=1.4, ls='--')
    ax.text(39.6, med[i], f'P{k+1}', fontsize=4.6, color=PCOL[pairs4[k]], va='center')
ax.legend(handles=[Line2D([], [], color=PCOL[p], lw=2, label=f'P{k+1} = {p}')
                   for k, p in enumerate(pairs4)],
          fontsize=4.6, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
ax.set_xlabel('snapshot id (39 healthy)', fontsize=6.5)
ax.set_ylabel('z', fontsize=6.5)
ax.set_xlim(-1, 44)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'G')

# ============ H 1x2: rank box + strip ============
ax = fig.add_subplot(G[3, 2:4])
boxes = []
for i in pos_idx:
    ranks = (snap > snap[:, i][:, None]).sum(axis=1) + 1
    boxes.append(ranks)
bp = ax.boxplot(boxes, widths=0.45, patch_artist=True, positions=range(4),
                boxprops=dict(facecolor=K.ROSE2, edgecolor='none'),
                medianprops=dict(color=K.DARK, lw=1.6),
                whiskerprops=dict(color=K.DARK, lw=0.7),
                capprops=dict(color=K.DARK, lw=0.7),
                flierprops=dict(markersize=2.5, markerfacecolor='#B91E8A', markeredgecolor='none'))
for k, i in enumerate(pos_idx):
    ranks = (snap > snap[:, i][:, None]).sum(axis=1) + 1
    ax.scatter(rng.uniform(-0.1, 0.1, 39) + k, ranks, s=7, color=PCOL[pairs4[k]],
               alpha=0.55, lw=0, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels([f'P{k+1}' for k in range(4)], fontsize=5.5)
ax.set_ylabel('rank in 22-row block', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'H')

# ============ I 1x1: fam-prior vs DeepCARE dumbbell ============
ax = fig.add_subplot(G[3, 4])
fp = fam_prior_score(X3[pos_idx])
fpx = (fp - fp.mean()) / (fp.std() + 1e-9)
for k in range(4):
    a, b_ = sorted([fpx[k], med[pos_idx[k]]])
    ax.plot([a, b_], [k, k], color=K.ROSE2, lw=2.2, zorder=1)
    ax.scatter([fpx[k]], [k], s=26, color='#FDB731', zorder=3, label='fam prior' if k == 0 else '')
    ax.scatter([med[pos_idx[k]]], [k], s=30, color='#F03A6A', zorder=4, label='DeepCARE' if k == 0 else '')
ax.set_yticks(range(4)); ax.set_yticklabels([f'P{k+1}' for k in range(4)], fontsize=5.5)
ax.set_xlabel('z in ext3 block', fontsize=6)
ax.legend(fontsize=4.4, loc='lower center', bbox_to_anchor=(0.62, 1.01), ncol=2, frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'I')

# ============ J 1x2: composition (zero rows dropped) ============
ax = fig.add_subplot(G[4, 2:4])
comp = [('clinical pairs', 4, '#F03A6A')]
for p in pairs4:
    if groups[p]:
        comp.append((f"{p.split('|')[0]} partners", len(groups[p]), '#FDB731'))
yl = np.arange(len(comp))[::-1]
for yi, (nm, n_, c_) in zip(yl, comp):
    ax.barh(yi, n_, color=c_, height=0.55)
    ax.text(n_ + 0.12, yi, str(n_), fontsize=5, va='center', color=K.DARK)
ax.set_yticks(yl); ax.set_yticklabels([nm for nm, _, _ in comp], fontsize=5.5)
ax.set_xlabel('rows in ext3 block (n=22)', fontsize=6.3)
ax.set_xlim(0, 7.2)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'J')

# ============ K 1x1: spread box ============
ax = fig.add_subplot(G[4, 4])
boxes = [snap[:, i] for i in pos_idx]
bp = ax.boxplot(boxes, widths=0.5, patch_artist=True, positions=range(4),
                boxprops=dict(facecolor=K.ROSE2, edgecolor='none'),
                medianprops=dict(color=K.DARK, lw=1.6),
                whiskerprops=dict(color=K.DARK, lw=0.7),
                capprops=dict(color=K.DARK, lw=0.7),
                flierprops=dict(markersize=2.5, markerfacecolor='#B91E8A', markeredgecolor='none'))
for k, i in enumerate(pos_idx):
    ax.scatter(np.full(39, k) + rng.uniform(-0.14, 0.14, 39), snap[:, i], s=6,
               color=PCOL[pairs4[k]], alpha=0.5, lw=0, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels([f'P{k+1}' for k in range(4)], fontsize=5.5)
ax.set_ylabel('snapshot z spread', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'K')

fig.savefig(OUT / 'Fig3_ext3.png', dpi=600, bbox_inches='tight')
fig.savefig(OUT / 'Fig3_ext3.pdf', bbox_inches='tight')
print('Fig3 v8: 11 panels A-K (L deleted), B ring heatmap 2x3 with iron-041 radial labels')
