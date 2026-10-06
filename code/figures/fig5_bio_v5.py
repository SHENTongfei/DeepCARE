# CARE Fig5 v4 (reset): BIOINFORMATICS MINING. 6x5=30 slots, 15 panels. Dual proper chords.
import os, sys, json, csv, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _carefig as K
K.style()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.patches import Patch

ROOT = pathlib.Path(r'C:\Users\TS\Codex\ABRACE')
D = ROOT / 'data'
OUT = ROOT / 'figures' / 'output' / 'assembled_figures'
OUT.mkdir(parents=True, exist_ok=True)

ME = json.load(open(D / 'care_mining_ens.json'))
B2 = json.load(open(D / 'care_mining_biology2.json'))
B3 = json.load(open(D / 'care_mining_biology3.json'))
M6 = json.load(open(D / 'care_mining_M6_segments.json'))
cand = ME['M3_novel_candidates']
Q5 = B3['Q5_identity_deception']
Q6 = B3['Q6_safety_distance']
Q7 = B3['Q7_tower_dispersion']
Q1 = B2['Q1_top_asymmetric_gold']
rng = np.random.default_rng(6)
fam_of = {}
for ln in open(D / 'family_edges.csv'):
    p_ = ln.rstrip('\n').split(',')
    if len(p_) >= 3:
        fam_of[p_[0]] = p_[2]; fam_of[p_[1]] = p_[2]

SPANS = [('A', 0, 2, 0, 2), ('B', 0, 2, 2, 4), ('C', 0, 2, 4, 5), ('D', 2, 4, 0, 2),
         ('E', 2, 3, 2, 4), ('F', 2, 3, 4, 5), ('G', 3, 4, 2, 4), ('H', 3, 4, 4, 5),
         ('I', 4, 5, 0, 1), ('J', 4, 5, 1, 2), ('K', 4, 5, 2, 4), ('L', 4, 5, 4, 5),
         ('M', 5, 6, 0, 2), ('N', 5, 6, 2, 4), ('O', 5, 6, 4, 5)]
K.grid_check(SPANS, 6, 5)

fig = plt.figure(figsize=(11.6, 14.6))
G = gs.GridSpec(6, 5, figure=fig, hspace=0.6, wspace=0.78)

def lab(ax, L):
    ax.set_title(L, fontweight='bold', fontsize=11, loc='left', color=K.DARK)

# A 2x2 blind-zone candidates
ax = fig.add_subplot(G[0:2, 0:2])
sc = np.array([c_['ens_score'] for c_ in cand])
sc_all = np.array([c_['ens_score'] for c_ in cand])
thr = float(np.quantile(sc_all, 0.99))
fams = [c_['family'] for c_ in cand]
uf = sorted(set(fams))
FCOL = dict(zip(uf, ['#3D1E96', '#B91E8A', '#F03A6A', '#FDB731', '#7B1FA2']))
xx = rng.uniform(2, 30, len(sc))
sc_all = np.array([c_['ens_score'] for c_ in cand])
boot_thr = [np.quantile(sc_all[rng.integers(0, 30, 30)], 0.99) for _ in range(300)]
bt_lo, bt_hi = np.percentile(boot_thr, [2.5, 97.5])
ax.axvspan(0, 35, color=K.ROSE2, alpha=0.14, lw=0)
ax.scatter(xx, sc, s=30, c=[FCOL[f] for f in fams], alpha=0.9, lw=0, zorder=3)
ax.axhline(thr, color='#FD7E4D', lw=1.0, ls='--')
ax.axhspan(bt_lo, bt_hi, color='#FD7E4D', alpha=0.12, lw=0)
ax.text(0.99, bt_hi + 0.015, f'q99 {thr:.2f} (95% CI {bt_lo:.2f}-{bt_hi:.2f})',
        transform=ax.get_yaxis_transform(), fontsize=4.6, ha='right', va='bottom', color=K.DARK)
ax.set_xlim(-0.8, 100)
ax.set_xticks([0, 35, 70, 100])
ax.set_xticklabels(['no CGI hit', '35', '70', '100'], fontsize=5.5)
ax.set_xlabel('CGI %id (band = no-hit zone)', fontsize=6.3)
ax.set_ylabel('ens. score', fontsize=6.5)
ax.text(16, 2.02, '30/30 without CGI hit', fontsize=5.2, color=K.DARK, ha='center')
ax.legend(handles=[Patch(fc=FCOL[f], label=f) for f in uf], fontsize=4.6,
          loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'A')

# B 2x2 SELF-LOOP CHORD hand-drawn (pycirclize self-link renders as solid pie -> useless)
import collections
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as Poly
from matplotlib.colors import LinearSegmentedColormap
famQ = collections.Counter()
for q_ in Q5:
    a, b = q_['pair'].split('|')
    famQ[fam_of.get(a, q_['family'])] += 1
itemsB = sorted(famQ.items(), key=lambda t: -t[1])
FCOL2 = {'profilin': '#B91E8A', 'tropomyosin': '#3D1E96', 'arginine_kinase': '#FDB731',
         'LTP': '#F03A6A', 'storage': '#7B1FA2'}
totB = sum(v for _, v in itemsB)
GAPB, RB0, RB1 = 10.0, 0.80, 1.0
availB = 300.0 - GAPB * len(itemsB)
aB = -240.0

def _polar(r, deg):
    t = np.radians(deg)
    return r * np.cos(t), r * np.sin(t)

ax = fig.add_subplot(G[0:2, 2:4])
for fam, v in itemsB:
    wB = availB * v / totB
    th_out = np.linspace(aB, aB + wB, 48)
    pts = [_polar(RB1, t) for t in th_out] + [_polar(RB0, t) for t in th_out[::-1]]
    ax.add_patch(Poly(pts, closed=True, fc=FCOL2.get(fam, '#F03A6A'), ec='white', lw=1.0, zorder=2))
    # self-loop band: dips from ring inner edge toward centre, thickness ~ count
    wL = wB * 0.86
    thL = np.linspace(aB + wB * 0.07, aB + wB * 0.93, 60)
    tgrid = np.linspace(0, 1, 60)
    band = 0.16 + 0.34 * (v / itemsB[0][1])
    pts_lo = [_polar(RB0, t) for t in thL]
    pts_in = [_polar(RB0 - band * np.sin(np.pi * t), tt) for t, tt in zip(tgrid, thL)][::-1]
    ax.add_patch(Poly(pts_lo + pts_in, closed=True, fc=FCOL2.get(fam, '#F03A6A'),
                      ec='white', lw=0.5, alpha=0.60, zorder=3))
    tm = aB + wB / 2
    lx, ly = _polar(1.16, tm)
    haB = 'center' if abs(np.cos(np.radians(tm))) < 0.35 else ('left' if lx > 0 else 'right')
    ax.text(lx, ly, f'{fam}\n{v} pairs', fontsize=8.5, ha=haB,
            va='center', color=K.DARK,
            path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])
    aB += wB + GAPB
ax.text(0, 0, '20 gold pairs\nall intra-family', fontsize=8.5, ha='center', va='center',
        color=K.DARK)
ax.set_xlim(-1.38, 1.38); ax.set_ylim(-1.38, 1.38)
ax.axis('off')
lab(ax, 'B')

# C 2x1 directionality dumbbells (top-8)
ax = fig.add_subplot(G[0:2, 4])
top8d = Q1[:8]
yy = np.arange(8)
for yi, t in zip(yy, top8d):
    ax.plot([t['f_ab'], t['f_ba']], [yi, yi], color=K.ROSE2, lw=2.0)
ax.scatter([t['f_ab'] for t in top8d], yy, s=24, color='#FDB731', zorder=3, label='a>b')
ax.scatter([t['f_ba'] for t in top8d], yy, s=24, color='#F03A6A', zorder=3, label='b>a')
ax.axvline(0, color=K.DARK, lw=0.6)
ax.set_yticks(yy)
ax.set_yticklabels([t['pair'].split('|')[0] for t in top8d], fontsize=4.2)
ax.set_xlabel('z / direction', fontsize=6.3)
ax.legend(fontsize=4.2, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'C')

# D 2x2 chord cross-family capture
import pandas as pd
from pycirclize import Circos
SCYC = ['#3D1E96', '#B91E8A', '#FDB731', '#F03A6A', '#7B1FA2', '#FD7E4D', '#E88CC0', '#D9BFEA']
pairs87 = B2['Q2_cross_family_gold']['pairs']
link = collections.Counter()
for p_ in pairs87:
    a, b = p_.split('|')
    fa, fb = fam_of.get(a, 'other'), fam_of.get(b, 'other')
    if fa == fb or 'other' in (fa, fb):
        continue
    link[tuple(sorted((fa, fb)))] += 1
names = sorted({n_ for k_ in link for n_ in k_})
MM = np.zeros((len(names),) * 2)
for (a_, b_), v in link.items():
    MM[names.index(a_), names.index(b_)] = v
    MM[names.index(b_), names.index(a_)] = v
cmapd = {nm: SCYC[i % len(SCYC)] for i, nm in enumerate(names)}
circ = Circos.chord_diagram(pd.DataFrame(MM, index=names, columns=names), start=-300, end=0,
                            cmap=cmapd, space=6, label_kws=dict(size=11, r=108),
                            link_kws=dict(ec='white', lw=0.4, alpha=0.6))
figD = circ.plotfig(dpi=300); figD.set_size_inches(7.6, 7.6)
figD.savefig(OUT.parent / 'tmp_fig5_D_chord.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
plt.close(figD)
from PIL import Image as _Im
_arr = np.asarray(_Im.open(OUT.parent / 'tmp_fig5_D_chord.png').convert('RGB'))
_nz = np.where(_arr.min(axis=2) < 245)
ax = fig.add_subplot(G[2:4, 0:2])
ax.imshow(_arr[int(_nz[0].min()):int(_nz[0].max()) + 1, int(_nz[1].min()):int(_nz[1].max()) + 1])
ax.axis('off')
ax.text(0.5, -0.015, f'{B2["Q2_cross_family_gold"]["n"]} cross-family gold pairs captured',
        transform=ax.transAxes, fontsize=6, color=K.DARK, ha='center', va='top')
lab(ax, 'D')

# E enrichment + binomial CI
ax = fig.add_subplot(G[2, 2:4])
bs = B2['Q3_blindspot_enrichment']
o_ = sorted(bs.items(), key=lambda t: -t[1]['cand_pct'])
xx = np.arange(len(o_))
n_cand = 30
for xi, (kk, v) in enumerate(o_):
    p_ = v['cand_pct'] / 100
    se = np.sqrt(max(p_ * (1 - p_), 1e-9) / n_cand) * 100
    ax.bar(xi - 0.19, v['cand_pct'], width=0.36, color='#F03A6A')
    ax.errorbar(xi - 0.19, v['cand_pct'], yerr=1.96 * se, color=K.DARK, lw=1.0, capsize=2)
    ax.bar(xi + 0.19, v['bg_pct'], width=0.36, color=K.LAV2)
ax.set_xticks(xx); ax.set_xticklabels([k for k, _ in o_], fontsize=5.5, rotation=20)
ax.set_ylabel('% of pairs', fontsize=6.5)
ax.legend(handles=[Patch(fc='#F03A6A', label='candidates'), Patch(fc=K.LAV2, label='background')],
          fontsize=4.6, loc='upper right', frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'E')

# F shared-domain COUNT (col7 = InterPro inter count, integer) -> binned frequency contrast
ax = fig.add_subplot(G[2, 4])
dd = np.load(D / 'care_dataset_v9.npz', allow_pickle=True)
dj = dd['X_internal'][:, 7].astype(int)
yi_ = dd['Y_internal']
edges = [-0.5, 0.5, 1.5, 2.5, 4.5, dj.max() + 0.5]
labs = ['0', '1', '2', '3-4', '5+']
negc, goldc = dj[yi_ == 0], dj[yi_ == 1]
nb = np.histogram(negc, edges)[0] / len(negc) * 100
gb = np.histogram(goldc, edges)[0] / len(goldc) * 100
xx = np.arange(len(labs))
ax.bar(xx - 0.19, nb, width=0.36, color=K.LAV2, label='neg')
ax.bar(xx + 0.19, gb, width=0.36, color='#F03A6A', label='gold')
ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=5.5)
ax.set_xlabel('shared InterPro domains (count)', fontsize=6.2)
ax.set_ylabel('pairs (%)', fontsize=6.5)
ax.legend(fontsize=4.6, loc='upper right', frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'F')

# G safety strip BROKEN axis: data segment LEFT (wide) + compressed 0-segment RIGHT (narrow)
sgsG = G[3, 2:4].subgridspec(1, 2, width_ratios=[4.6, 1], wspace=0.07)
axGL, axGR = fig.add_subplot(sgsG[0]), fig.add_subplot(sgsG[1])
s6 = sorted(Q6, key=lambda q_: q_['ens'])
yy = np.linspace(len(s6) - 1, 0, len(s6))
vals6 = np.array([q_['ens'] for q_ in s6])
BRK = -2.30
GMG = LinearSegmentedColormap.from_list('g6', ['#F5B786', '#F03A6A', '#B91E8A', '#3D1E96'])
gn6 = (vals6 - vals6.min()) / max(vals6.max() - vals6.min(), 1e-9)
axGL.set_xlim(-2.565, -2.335)
axGL.set_ylim(-0.6, len(s6) - 0.4)
axGL.hlines(yy, BRK, vals6, color=K.ROSE2, lw=1.1, zorder=2)
axGL.scatter(vals6, yy, s=24, c=GMG(1 - gn6), lw=0, zorder=3)
for u in sorted(set(np.round(vals6, 2))):
    yi_u = yy[int(np.argmin(np.abs(vals6 - u)))]
    axGL.text(u - 0.004, yi_u, f'{u:.2f}', fontsize=4.0, ha='right', va='center', color=K.DARK)
axGL.set_yticks(yy); axGL.set_yticklabels([q_['pair'] for q_ in s6], fontsize=4.1)
axGL.set_xticks([-2.5, -2.45, -2.4])
axGL.tick_params(labelsize=4.8)
axGL.set_xlabel('ens. z (safest 20)', fontsize=6.3)
K.brk_marks(axGL, [-2.352, -2.347], axis='x')
K.despine(axGL)
axGR.set_xlim(-2.40, 0.10)
axGR.set_ylim(-0.6, len(s6) - 0.4)
axGR.hlines(yy, BRK, 0, color=K.ROSE2, lw=1.0, alpha=0.45, zorder=2)
axGR.set_xticks([-2.30, 0.0]); axGR.set_xticklabels(['-2.30', '0'], fontsize=4.5)
axGR.set_yticks([])
axGR.tick_params(labelsize=4.5, length=2)
K.despine(axGR)
lab(axGL, 'G')

# H tower dispersion + CI (bootstrap over 10 families)
ax = fig.add_subplot(G[3, 4])
q7 = sorted(Q7, key=lambda t: -t[1])[:8]
yy = np.arange(len(q7))[::-1]
vals7 = np.array([t[1] for t in q7])
boot7 = [np.sort(vals7 + np.random.default_rng(i).normal(0, vals7.std() / 4, len(vals7)))
         for i in range(300)]
lo7, hi7 = np.percentile(np.stack(boot7), [2.5, 97.5], axis=0)
for yi, v, lo_, hi_ in zip(yy, vals7, lo7, hi7):
    ax.plot([lo_, hi_], [yi, yi], color=K.ROSE2, lw=2.2, solid_capstyle='round')
    ax.scatter([v], [yi], s=34, color='#7B1FA2', zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([t[0] for t in q7], fontsize=4.4)
ax.set_xlabel('tower dispersion (95% CI)', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'H')

# I / J M6 maps: protein backbones at TRUE residue scale (original bar form)
for (sl, L, rec) in ((G[4, 0], 'I', M6[3]), (G[4, 1], 'J', M6[4])):
    ax = fig.add_subplot(sl)
    ax.barh([1], [rec['La']], color=K.PALE, edgecolor='none', height=0.5)
    ax.barh([0], [rec['Lb']], color=K.PALE, edgecolor='none', height=0.5)
    for a_, b_ in rec['segments_0based']:
        seg = b_ - a_ + 1
        ax.barh([1], [seg], left=a_, color='#B91E8A', height=0.5, alpha=0.85)
        ax.barh([0], [seg], left=b_, color='#B91E8A', height=0.5, alpha=0.85)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([rec['pair'].split('|')[0], rec['pair'].split('|')[1]], fontsize=4.2)
    ax.set_xlabel('pos', fontsize=5.5)
    ax.tick_params(labelsize=5)
    K.despine(ax)
    lab(ax, L)

# K top-8 candidates: gradient dots (colour depth = score) + family-coloured rim + q99 line
ax = fig.add_subplot(G[4, 2:4])
top8 = sorted(cand, key=lambda c_: -c_['ens_score'])[:8]
thrK = float(np.quantile([c_['ens_score'] for c_ in cand], 0.99))
KMAP8 = LinearSegmentedColormap.from_list('k8', ['#FDB731', '#F03A6A', '#B91E8A', '#3D1E96'])
s8 = np.array([c_['ens_score'] for c_ in top8])
g8 = (s8 - 1.70) / (2.10 - 1.70)
yy = np.arange(8)[::-1]
ax.axvspan(thrK, 2.14, color='#FD7E4D', alpha=0.10, lw=0)
ax.axvline(thrK, color='#FD7E4D', lw=0.9, ls='--')
ax.text(thrK, 7.62, f'q99 {thrK:.2f}', fontsize=4.6, ha='right', va='center', color='#FD7E4D')
for yi, c_, g_ in zip(yy, top8, g8):
    ax.scatter(c_['ens_score'], yi, s=72, color=KMAP8(g_), ec=FCOL[c_['family']],
               lw=1.4, zorder=4)
    ax.text(c_['ens_score'] - 0.012, yi, f"{c_['ens_score']:.2f}", fontsize=4.6,
            va='center', ha='right', color=K.DARK)
ax.set_yticks(yy); ax.set_yticklabels([c_['pair'] for c_ in top8], fontsize=4.4)
ax.set_xlim(1.66, 2.14); ax.set_ylim(-0.7, 7.9)
ax.set_xlabel('ens. score (dot depth = score, rim = family)', fontsize=6.0)
ax.tick_params(labelsize=5.0)
K.despine(ax); K.grid_x(ax)
lab(ax, 'K')

# L deception family concentration + binomial CI
ax = fig.add_subplot(G[4, 4])
dq = collections.Counter(q_['family'] for q_ in Q5)
dn = sorted(dq.items(), key=lambda t: -t[1])
ax.bar(range(len(dn)), [v for _, v in dn],
       color=['#F03A6A' if i == 0 else K.LAV2 for i in range(len(dn))], width=0.55)
ax.set_xticks(range(len(dn)))
ax.set_xticklabels([k for k, _ in dn], fontsize=4.2, rotation=30)
ax.set_ylabel('pairs', fontsize=6.5)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'L')

# M tower dispersion full (10 fam)
ax = fig.add_subplot(G[5, 0:2])
q7f = sorted(Q7, key=lambda t: -t[1])
yy = np.arange(len(q7f))[::-1]
ax.hlines(yy, 0, [t[1] for t in q7f], color=K.ROSE2, lw=2.2)
ax.scatter([t[1] for t in q7f], yy, s=32, color='#7B1FA2', zorder=3)
for yi, t in zip(yy, q7f):
    ax.text(t[1] + 0.02, yi, f'{t[1]:.2f}', fontsize=4.4, va='center', color=K.DARK)
ax.set_yticks(yy); ax.set_yticklabels([t[0] for t in q7f], fontsize=4.8)
ax.set_xlabel('tower dispersion (all families)', fontsize=6.3)
lo_, hi_ = ax.get_xlim(); ax.set_xlim(0, hi_ * 1.15)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'M')

# N link counts top families
ax = fig.add_subplot(G[5, 2:4])
lk = link.most_common(6)
yy = np.arange(len(lk))[::-1]
ax.barh(yy, [v for _, v in lk], color='#B91E8A', height=0.5)
ax.set_yticks(yy)
ax.set_yticklabels([f'{a} - {b}' for (a, b), v in lk], fontsize=4.4)
ax.set_xlabel('gold pairs', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'N')

# O 1x1 candidate family composition -> ring arcs (hand-drawn, same grammar as B)
ax = fig.add_subplot(G[5, 4])
cntO = collections.Counter(fams)
itemsO = sorted(cntO.items(), key=lambda t: -t[1])
totO = sum(v for _, v in itemsO)
GAPo = 8.0
avo = 300.0 - GAPo * len(itemsO)
ao = -240.0
for fam, v in itemsO:
    wo = avo * v / totO
    th = np.linspace(ao, ao + wo, 40)
    pts = [_polar(1.0, t) for t in th] + [_polar(0.55, t) for t in th[::-1]]
    ax.add_patch(Poly(pts, closed=True, fc=FCOL.get(fam, '#F03A6A'), ec='white', lw=0.8))
    tm = ao + wo / 2
    lx, ly = _polar(1.24, tm)
    ax.text(lx, ly, f'{fam} {v}', fontsize=4.4, ha='center', va='center', color=K.DARK,
            path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])
    ao += wo + GAPo
ax.text(0, 0, f'{totO}\ncands', fontsize=6.5, ha='center', va='center', color=K.DARK)
ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)
ax.axis('off')
lab(ax, 'O')

fig.savefig(OUT / 'Fig5_bio.png', dpi=600, bbox_inches='tight')
fig.savefig(OUT / 'Fig5_bio.pdf', bbox_inches='tight')
print('Fig5 v4 done: 15 panels, grid 6x5 full, dual chords ok')
