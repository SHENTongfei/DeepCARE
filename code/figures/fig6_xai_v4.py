# CARE Fig6 v4 (reset): INTERPRETABLE DL. 6x5=30 slots, 14 panels.
# Ring #4 = ring-stacked excess IG (F+ form + rule 046); ring #5 = ring heatmap (I+ 041 trio).
# CARE Fig6 v4 (reset): XAI
import matplotlib.patheffects as pe
import os, sys, json, pathlib
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
M6 = json.load(open(D / 'care_mining_M6_segments.json'))
blocks = list(ME['M1_ig_block_totals'].keys())
tot = np.array([ME['M1_ig_block_totals'][b] for b in blocks])
posm = np.array([ME['M1_ig_pos_mean'][b] for b in blocks])
negm = np.array([ME['M1_ig_neg_mean'][b] for b in blocks])
BSHORT = [b.replace('tower_', '') for b in blocks]
fams = list(ME['M2_gate_by_family'].keys())
views = ME['M2_view_names']
V = np.array([ME['M2_gate_by_family'][f] for f in fams], dtype=float).T

SPANS = [('A', 0, 2, 0, 2), ('B', 0, 2, 2, 4), ('C', 0, 2, 4, 5), ('D', 2, 4, 0, 1),
         ('E', 2, 4, 1, 2), ('F', 2, 3, 2, 4), ('G', 2, 3, 4, 5), ('H', 3, 4, 2, 4),
         ('I', 3, 4, 4, 5), ('J', 4, 5, 0, 2), ('K', 4, 5, 2, 3), ('L', 4, 5, 3, 4),
         ('M', 4, 5, 4, 5), ('N', 5, 6, 0, 3), ('O', 5, 6, 3, 5)]
K.grid_check(SPANS, 6, 5)

fig = plt.figure(figsize=(11.6, 14.6))
G = gs.GridSpec(6, 5, figure=fig, hspace=0.6, wspace=0.8)

def lab(ax, L):
    ax.set_title(L, fontweight='bold', fontsize=11, loc='left', color=K.DARK)

# A 2x2 ring-stacked excess IG (pos+neg stack; radius ~ excess, labels = raw) — rule 046 + F+
order = np.argsort(-tot)
tot_o = tot[order]; lab_o = [BSHORT[i] for i in order]
stack = np.stack([np.maximum(posm[order], 0), np.maximum(negm[order], 0)], axis=1)
exc = 0.30 + 0.70 * (tot_o - tot_o.min()) / max(tot_o.max() - tot_o.min(), 1e-9)
band = stack * (exc / np.maximum(stack.sum(axis=1), 1e-9))[:, None]
from pycirclize import Circos
sec = {lab_o[i]: 4 for i in range(len(lab_o))}
circ = Circos(sec, start=0, end=300, space=4)
vmax = float(band.sum(axis=1).max() * 1.05 + 1e-9)
for i, s_ in enumerate(circ.sectors):
    s_.text(s_.name, r=102, size=13, color=K.DARK, adjust_rotation=True, orientation='vertical',
            path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    tr = s_.add_track((46, 100))
    tr.axis(ec='white', lw=0.3)
    bots = np.concatenate([[0], np.cumsum(band[i])[:-1]]).astype(float)
    for j, col in enumerate(('#F03A6A', '#FDB731')):
        tr.bar(np.array([1.5]), np.array([float(band[i, j])]), width=3.0,
               bottom=float(bots[j]), color=col, ec='white', lw=0.5, vmin=0.0, vmax=vmax)
    tr.text(f'{tot_o[i]:.1e}', x=1.5, r=37 + (5 if i % 2 else 0), size=9.5, color=K.DARK,
            adjust_rotation=True, orientation='vertical')
figA = circ.plotfig(dpi=300); figA.set_size_inches(7.2, 7.2)
figA.axes[0].set_position([0.0, 0.05, 1.0, 0.95])
figA.legend(handles=[Patch(fc='#F03A6A', label='pos IG'), Patch(fc='#FDB731', label='neg IG')],
            loc='upper left', frameon=False, fontsize=13, bbox_to_anchor=(0.02, 0.985))
figA.text(0.5, 0.005, 'radius ~ excess total IG (labels = raw)', fontsize=13, color=K.DARK, ha='center')
figA.savefig(OUT.parent / 'tmp_fig6_A_ring.png', dpi=300, bbox_inches='tight', pad_inches=0.06)
plt.close(figA)
from PIL import Image as _Im
_arr = np.asarray(_Im.open(OUT.parent / 'tmp_fig6_A_ring.png').convert('RGB'))
_nz = np.where(_arr.min(axis=2) < 245)
ax = fig.add_subplot(G[0:2, 0:2])
ax.imshow(_arr[int(_nz[0].min()):int(_nz[0].max()) + 1, int(_nz[1].min()):int(_nz[1].max()) + 1])
ax.axis('off')
lab(ax, 'A')

# B 2x2 ring heatmap of gate routing (041 trio: gap 3 o'clock, labels in gap, colorbar outside top)
Vn = (V - V.min()) / max(V.max() - V.min(), 1e-9)
o2B = np.argsort(tot)            # ring order = C-panel y order (inner->outer = bottom->top)
Vn = Vn[o2B]
SEQ9 = plt.cm.colors.LinearSegmentedColormap.from_list('sf9', K.SEQ)
circ = Circos({fams[i]: 11 for i in range(len(fams))}, start=-240, end=60, space=2.2)
for i, s_ in enumerate(circ.sectors):
    s_.text(s_.name, r=99, size=14, color=K.DARK, adjust_rotation=True, orientation='vertical',
            path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    tr = s_.add_track((24, 96))
    tr.axis(ec='white', lw=0.3)
    tr.heatmap(Vn[:, i].reshape(-1, 1), vmin=0, vmax=1, cmap=SEQ9,
               rect_kws=dict(ec='white', lw=0.3))
    if i == len(fams) - 1:
        tr.yticks([r + 0.5 for r in range(len(views))],
                  [f'T{k+1}' for k in range(len(views))][::-1],
                  vmin=0, vmax=len(views), side='right', tick_length=1.6,
                  label_size=10.5, label_margin=0.12)
figB = circ.plotfig(dpi=300); figB.set_size_inches(9.0, 9.0)
figB.axes[0].set_position([0.0, 0.0, 1.0, 1.0])
import io as _ioB
_bufB6 = _ioB.BytesIO()
figB.savefig(_bufB6, format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close(figB)
_bufB6.seek(0)
_arr = np.asarray(_Im.open(_bufB6).convert('RGB'))
_nz = np.where(_arr.min(axis=2) < 245)
ax = fig.add_subplot(G[0:2, 2:4])
ax.imshow(_arr[int(_nz[0].min()):int(_nz[0].max()) + 1, int(_nz[1].min()):int(_nz[1].max()) + 1])
ax.axis('off')
lab(ax, 'B')
posB6 = ax.get_position()
caxB6 = fig.add_axes([posB6.x0 + 0.015, posB6.y0 - 0.0225, posB6.width * 0.97, 0.0045])
cbB6 = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ9,
                    norm=plt.cm.colors.Normalize(vmin=0, vmax=1)),
                    cax=caxB6, orientation='horizontal')
cbB6.set_ticks([0, 1]); cbB6.set_ticklabels(['low', 'high'])
cbB6.ax.tick_params(labelsize=6.5, colors=K.DARK)
cbB6.outline.set_visible(False)
fig.text(posB6.x0 + posB6.width * 0.56, posB6.y0 - 0.0305, 'gate weight · T1-T11: inner→outer = C bottom→top',
         fontsize=6.0, color=K.DARK, ha='center', va='top')

# C 2x1 pos-fraction of tower IG (all towers in one [0,1] frame; dot depth = total IG)
ax = fig.add_subplot(G[0:2, 4])
o2 = np.argsort(tot)
fracC = posm[o2] / np.maximum(posm[o2] + negm[o2], 1e-9)
lgtC = np.log10(np.maximum(tot[o2], 1e-9))
CMC = plt.cm.colors.LinearSegmentedColormap.from_list('c6', ['#FDB731', '#F03A6A', '#B91E8A', '#3D1E96'])
gnC = (lgtC - lgtC.min()) / max(lgtC.max() - lgtC.min(), 1e-9)
ax.axvline(0.5, color=K.DARK, lw=0.6, ls='--')
ax.scatter(fracC, np.arange(len(o2)), s=66, c=CMC(gnC), ec='white', lw=0.6, zorder=3)
for k, f_ in enumerate(fracC):
    ax.text(f_, k + 0.36, f'{f_ * 100:.0f}%', fontsize=4.4, ha='center', color=K.DARK)
ax.set_yticks(np.arange(len(o2)))
ax.set_yticklabels([BSHORT[i] for i in o2], fontsize=4.8)
ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.7, len(o2) - 0.3)
ax.set_xlabel('pos fraction of tower IG (dot depth = total IG, log)', fontsize=5.8)
ax.legend(handles=[Patch(fc='#B91E8A', label='deep = high total IG')], fontsize=4.2,
          loc='lower right', frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'C')

# D 2x1 IG share lollipop on LOG x (99.9% vs 0.001-0.02% spread over 5 decades)
ax = fig.add_subplot(G[2:4, 0])
share = tot / tot.sum() * 100
o3 = np.argsort(share)
xminD = share[o3][0] * 0.5
ax.hlines(range(len(o3)), xminD, share[o3], color=K.ROSE2, lw=2)
ax.scatter(share[o3], range(len(o3)), s=30, color='#B91E8A', zorder=3)
ax.scatter([share[o3[-1]]], [len(o3) - 1], s=46, color='#F03A6A', zorder=4)
for k, s_ in zip(range(len(o3)), share[o3]):
    ax.text(s_ * 1.8, k, f'{s_:.3g}%', fontsize=4.4, va='center', color=K.DARK)
ax.set_xscale('log')
ax.set_xlim(xminD, 500)
ax.set_yticks(range(len(o3)))
ax.set_yticklabels([BSHORT[i] for i in o3], fontsize=5.2)
ax.set_xlabel('IG share (%) — log', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'D')

# E 2x1 modal gate view per family (transposed: family names on y, colour = view)
ax = fig.add_subplot(G[2:4, 1])
Vf = np.array([ME['M2_gate_by_family'][f] for f in fams], dtype=float)
modal = [views[int(np.argmax(Vf[i]))].replace('tower_', '') for i in range(len(fams))]
uniq = sorted(set(modal))
UCOL = dict(zip(uniq, ['#3D1E96', '#B91E8A', '#F03A6A', '#FDB731', '#7B1FA2', '#FD7E4D']))
for yi, f in enumerate(fams):
    ax.scatter(uniq.index(modal[yi]), yi, s=50, color=UCOL[modal[yi]], ec='white',
               lw=0.5, zorder=3)
ax.set_yticks(range(len(fams)))
ax.set_yticklabels(fams, fontsize=4.6)
ax.set_ylim(-0.8, len(fams) - 0.2)
ax.set_xticks(range(len(uniq)))
ax.set_xticklabels(uniq, fontsize=5.0, rotation=25, ha='right')
ax.set_xlim(-0.7, len(uniq) - 0.3)
ax.set_xlabel('modal view per family', fontsize=6.0)
K.despine(ax); K.grid_x(ax)
lab(ax, 'E')

# F/G/H M6 maps: protein backbones at TRUE residue scale (original bar form)
def _m6_colinear(ax, rec, nfs=5):
    ax.barh([1], [rec['La']], color=K.PALE, edgecolor='none', height=0.5)
    ax.barh([0], [rec['Lb']], color=K.PALE, edgecolor='none', height=0.5)
    for a_, b_ in rec['segments_0based']:
        if a_ >= rec['La'] or b_ >= rec['Lb']:
            continue  # out-of-bounds segment (data anomaly, e.g. M6[1] [273,273] vs Lb=96)
        seg = b_ - a_ + 1
        ax.barh([1], [seg], left=a_, color='#B91E8A', height=0.5, alpha=0.85)
        ax.barh([0], [seg], left=b_, color='#B91E8A', height=0.5, alpha=0.85)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([rec['pair'].split('|')[0], rec['pair'].split('|')[1]], fontsize=nfs)
    ax.tick_params(labelsize=5)
    K.despine(ax)

ax = fig.add_subplot(G[2, 2:4])
_m6_colinear(ax, M6[0], nfs=5)
ax.set_xlabel(f"residue pos (prof. mean {M6[0]['profile_mean']:.2f})", fontsize=5.8)
ax.tick_params(labelsize=5.5)
lab(ax, 'F')

ax = fig.add_subplot(G[2, 4])
_m6_colinear(ax, M6[1], nfs=4.4)
ax.set_xlabel('pos', fontsize=5.5)
ax.tick_params(labelsize=5)
lab(ax, 'G')

ax = fig.add_subplot(G[3, 2:4])
_m6_colinear(ax, M6[2], nfs=5)
ax.set_xlabel('residue pos', fontsize=5.8)
ax.tick_params(labelsize=5.5)
lab(ax, 'H')

# I 1x1 gate entropy BROKEN axis (data 0.77-0.92; compressed 0->0.75 left, data right)
sgsI = G[3, 4].subgridspec(1, 2, width_ratios=[1, 5.2], wspace=0.07)
axIL, axIR = fig.add_subplot(sgsI[0]), fig.add_subplot(sgsI[1])
P = V / np.maximum(V.sum(axis=0, keepdims=True), 1e-9)
Hn = -(P * np.log(P + 1e-12)).sum(axis=0) / np.log(len(views))
o5 = np.argsort(Hn)
BRKI = 0.75
yyI = np.arange(len(o5))
axIL.hlines(yyI, 0, BRKI, color=K.ROSE2, lw=1.0, alpha=0.5, zorder=2)
axIL.set_xlim(-0.02, BRKI + 0.025)
axIL.set_xticks([0, BRKI]); axIL.set_xticklabels(['0', '0.75'], fontsize=4.4)
axIL.set_yticks(yyI); axIL.set_yticklabels([fams[i] for i in o5], fontsize=4.0)
axIL.set_ylim(-0.6, len(o5) - 0.4)
axIL.tick_params(labelsize=4.4, length=2)
K.despine(axIL)
axIR.hlines(yyI, BRKI, Hn[o5], color=K.ROSE2, lw=1.6, zorder=2)
axIR.scatter(Hn[o5], yyI, s=24, color='#7B1FA2', zorder=3)
for k, hv in zip(yyI, Hn[o5]):
    axIR.text(hv + 0.013, k, f'{hv:.2f}', fontsize=3.9, va='center', color=K.DARK)
axIR.set_xlim(BRKI, 0.985)
axIR.set_ylim(-0.6, len(o5) - 0.4)
axIR.set_yticks([])
axIR.set_xticks([0.8, 0.9])
axIR.tick_params(labelsize=4.6)
axIR.set_xlabel('gate entropy (norm.)', fontsize=6.0)
K.brk_marks(axIR, [0.7545, 0.7585], axis='x')
K.despine(axIR)
lab(axIL, 'I')

# J 1x2 log2 pos/neg IG ratio
ax = fig.add_subplot(G[4, 0:2])
ratio = np.log2(posm / np.maximum(negm, 1e-9))
o6 = np.argsort(ratio)
ax.barh(range(len(o6)), ratio[o6],
        color=['#F03A6A' if ratio[o6][i] > 0 else '#FDB731' for i in range(len(o6))], height=0.55)
ax.axvline(0, color=K.DARK, lw=0.6)
ax.set_yticks(range(len(o6)))
ax.set_yticklabels([BSHORT[i] for i in o6], fontsize=5.2)
ax.set_xlabel('log2 pos/neg IG (rose = gold-specific)', fontsize=6.2)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'J')

# K 1x1 anchors log lollipop (fam/cgi among towers)
ax = fig.add_subplot(G[4, 2])
lg = np.log10(tot)
o4 = np.argsort(lg)
cbar = ['#FDB731' if ('fam' in blocks[i] or blocks[i] == 'cgi') else '#B91E8A' for i in o4]
ax.hlines(range(len(o4)), 0, lg[o4], color=K.ROSE2, lw=1.6)
ax.scatter(lg[o4], range(len(o4)), s=26, color=cbar, zorder=3)
ax.set_yticks(range(len(o4)))
ax.set_yticklabels([BSHORT[i] for i in o4], fontsize=4.4)
ax.set_xlabel('log10 total IG (gold = anchors)', fontsize=6.2)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'K')

# L 1x1 mean gate share bars
ax = fig.add_subplot(G[4, 3])
gw = V.mean(axis=1)
o7 = np.argsort(gw)
ax.hlines(range(len(o7)), 0, gw[o7] * 100, color=K.ROSE2, lw=2)
ax.scatter(gw[o7] * 100, range(len(o7)), s=26, color='#B91E8A', zorder=3)
ax.set_yticks(range(len(o7)))
ax.set_yticklabels([v.replace('tower_', '') for v in [views[i] for i in o7]], fontsize=4.2)
ax.set_xlabel('mean gate share (%)', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'L')

# M 1x1 entropy vs dispersion scatter (greedy collision-free labels, all families)
ax = fig.add_subplot(G[4, 4])
B3 = json.load(open(D / 'care_mining_biology3.json'))
disp = {t[0]: t[1] for t in B3['Q7_tower_dispersion']}
ptsM = [(Hn[fams.index(f)], disp[f], f) for f in fams if f in disp]
xs_, ys_ = [p[0] for p in ptsM], [p[1] for p in ptsM]
ax.scatter(xs_, ys_, s=28, color='#B91E8A', alpha=0.85, lw=0)
placedM = []
for x_, y_, f in sorted(ptsM, key=lambda t: (t[1], t[0])):
    for dx, dy in ((7, 3), (7, -10), (7, 13), (-34, 3), (12, 3), (12, 13)):
        cx = x_ + dx / 813.0   # 1 x-unit ~= 813 pt in this panel
        cy = y_ + dy / 262.0   # 1 y-unit ~= 262 pt
        if all(abs(cx - px) > 0.048 or abs(cy - py) > 0.040 for px, py in placedM):
            break
    ax.annotate(f, (x_, y_), fontsize=3.8, xytext=(dx, dy),
                textcoords='offset points', color=K.DARK)
    placedM.append((cx, cy))
ax.set_xlabel('gate entropy', fontsize=6.3)
ax.set_ylabel('tower dispersion', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'M')

# N 1x3 Pareto: log10 IG bars + cumulative line (twin axis)
ax = fig.add_subplot(G[5, 0:3])
o8 = np.argsort(-tot)
cum = np.cumsum(tot[o8]) / tot.sum() * 100
lgn = np.log10(np.maximum(tot[o8], 1e-9))
ax.bar(range(len(o8)), lgn, color='#F03A6A', alpha=0.8, width=0.6, zorder=2)
ax.set_ylabel('total IG (log10)', fontsize=6.5)
ax.set_ylim(0, 8.6)
ax2N = ax.twinx()
ax2N.plot(range(len(o8)), cum, color='#3D1E96', lw=1.4, marker='o', ms=3.2, zorder=3)
ax2N.set_ylim(0, 108)
ax2N.set_ylabel('cumulative IG (%)', fontsize=6.2, color='#3D1E96')
ax2N.tick_params(labelsize=5, colors='#3D1E96')
ax2N.spines['top'].set_visible(False)
ax.set_xticks(range(len(o8)))
ax.set_xticklabels([BSHORT[i] for i in o8], fontsize=4.4, rotation=30, ha='right')
ax.tick_params(labelsize=5.5)
K.despine(ax)
lab(ax, 'N')

# O 1x2 gate mini-heatmap families x top views
ax = fig.add_subplot(G[5, 3:5])
topv = np.argsort(-V.mean(axis=1))[:5]
Vm = V[topv]
Vmn = (Vm - Vm.min()) / max(Vm.max() - Vm.min(), 1e-9)
ramp2 = plt.cm.colors.LinearSegmentedColormap.from_list('sf2', K.SEQ)
imO = ax.imshow(Vmn, cmap=ramp2, aspect='auto')
cbO = fig.colorbar(imO, ax=ax, fraction=0.05, pad=0.03)
cbO.ax.tick_params(labelsize=4.5, colors=K.DARK)
cbO.outline.set_visible(False)
cbO.set_label('gate (norm.)', fontsize=5.0, color=K.DARK)
ax.set_xticks(range(len(fams)))
ax.set_xticklabels([f[:7] for f in fams], fontsize=4.2, rotation=40, ha='right')
ax.set_yticks(range(5))
ax.set_yticklabels([views[i].replace('tower_', '') for i in topv], fontsize=4.4)
K.despine(ax, keep_bottom=False, keep_left=False)
ax.tick_params(length=0)
lab(ax, 'O')

fig.savefig(OUT / 'Fig6_xai.png', dpi=600, bbox_inches='tight')
fig.savefig(OUT / 'Fig6_xai.pdf', bbox_inches='tight')
print('Fig6 v4 done: 15 panels, grid 6x5 full, ring-stack + ring-heatmap ok')
