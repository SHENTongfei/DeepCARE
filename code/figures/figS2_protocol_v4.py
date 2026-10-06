# CARE FigS2 v4 (reset): PROTOCOL & AUDIT. 5x5=25 slots, 12 panels A-L, position-correct.
import os, sys, json, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'C:/Users/TS/Codex/ABRACE/code')
import _carefig as K
K.style()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

ROOT = pathlib.Path(r'C:\Users\TS\Codex\ABRACE')
D = ROOT / 'data'
OUT = ROOT / 'figures' / 'output' / 'assembled_figures'
OUT.mkdir(parents=True, exist_ok=True)

rk = json.load(open(D / 'care_cv_results_ranknet.json'))
led = json.load(open(D / 'inno_ledger.json'))
rng = np.random.default_rng(32)
import csv as _csv
prov = list(_csv.DictReader(open(D / 'ext3_pairs.tsv'), delimiter='\t'))

SPANS = [('A', 0, 2, 0, 2), ('B', 0, 2, 2, 4), ('C', 0, 2, 4, 5), ('D', 2, 3, 0, 2),
         ('E', 2, 4, 2, 4), ('F', 2, 3, 4, 5), ('G', 3, 4, 0, 2), ('H', 3, 4, 4, 5),
         ('I', 4, 5, 0, 2), ('J', 4, 5, 2, 4), ('K', 4, 5, 4, 5)]
K.grid_check(SPANS, 5, 5)

fig = plt.figure(figsize=(11.2, 13.4))
G = gs.GridSpec(5, 5, figure=fig, hspace=0.55, wspace=0.7)

def lab(ax, L):
    ax.set_title(L, fontweight='bold', fontsize=11, loc='left', color=K.DARK)

# ---------- A 2x2 (r0-1,c0-1): all recorded screens (verdicts from inno_ledger) ----------
ax = fig.add_subplot(G[0:2, 0:2])
scrA = led['screens']
rowsA = []
for s_ in scrA:
    rowsA.append((s_['name'] + ' TI', s_['TI'], s_['refTI'], s_['verdict']))
    rowsA.append((s_['name'] + ' HO', s_['HO'], s_['refHO'], s_['verdict']))
yyA = np.arange(len(rowsA))[::-1]
for yi, (nm, v, ref, vd) in zip(yyA, rowsA):
    col = K.NAVY if vd == 'PASS' else K.ROSE
    ax.plot([min(v, ref), max(v, ref)], [yi, yi], color=K.ROSE2, lw=1.8)
    ax.scatter([v], [yi], s=28, color=col, zorder=3)
    ax.scatter([ref], [yi], s=30, facecolor='none', edgecolor=K.SAND, lw=1.1, zorder=3)
n_pass = sum(1 for s_ in scrA if s_['verdict'] == 'PASS')
ax.set_yticks(yyA); ax.set_yticklabels([r[0] for r in rowsA], fontsize=5.0)
ax.set_xlabel(f'AUPRC (navy = pass {n_pass}/{len(scrA)}, rose = fail; ring = ref)', fontsize=6.0)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'A')

# ---------- B 2x2 (r0-1,c2-4): seed x fold heatmap ----------
ax = fig.add_subplot(G[0:2, 2:4])
Mh = np.zeros((8, 5))
for r in rk:
    Mh[r['seed'], r['fold']] = r['internal']['AUPRC']
ramp = plt.cm.colors.LinearSegmentedColormap.from_list('sf', K.SEQ)
im = ax.imshow(Mh, cmap=ramp, aspect='auto')
ax.set_xticks(range(5)); ax.set_xticklabels([f'f{i}' for i in range(5)], fontsize=5.5)
ax.set_yticks(range(8)); ax.set_yticklabels([f's{i}' for i in range(8)], fontsize=5.5)
for i in range(8):
    for j in range(5):
        ax.text(j, i, f'{Mh[i, j]:.3f}'.lstrip('0'), ha='center', va='center',
                fontsize=3.6, color='white' if Mh[i, j] > 0.12 else K.DARK)
cax = ax.inset_axes([1.04, 0, 0.05, 1])
cb = plt.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=4.5, colors=K.DARK)
cb.outline.set_visible(False)
ax.set_xlabel('fold', fontsize=6.3); ax.set_ylabel('seed', fontsize=6.3)
K.despine(ax, keep_bottom=False, keep_left=False)
ax.tick_params(length=0)
lab(ax, 'B')

# ---------- C 2x1 (r0-1,c4): holdout split ----------
ax = fig.add_subplot(G[0:2, 4])
ax.barh([0], [284], color=K.LAV2, height=0.45)
ax.barh([0], [73], left=284, color=K.NAVY, height=0.45)
ax.text(142, 0, '284 train', ha='center', va='center', fontsize=5, color=K.DARK)
ax.text(320, 0, '73', ha='center', va='center', fontsize=5, color='white')
ax.set_yticks([])
ax.set_xlabel('gold pairs (family-stratified 80/20)', fontsize=5.8)
ax.tick_params(labelsize=5.5)
K.despine(ax)
lab(ax, 'C')

# ---------- E 1x2 (r2,c0-1): per-seed temperature ----------
ax = fig.add_subplot(G[2, 0:2])
temps = {}
for r in rk:
    temps.setdefault(r['seed'], []).append(r.get('temp', np.nan))
tv = [float(np.nanmean(temps[s])) for s in range(8)]
ax.hlines(range(8), 0, tv, color=K.ROSE2, lw=2)
ax.scatter(tv, range(8), s=26, color='#B91E8A', zorder=3)
ax.set_yticks(range(8)); ax.set_yticklabels([f's{i}' for i in range(8)], fontsize=5.5)
ax.set_xlabel('val temperature', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'D')

# ---------- F 2x2 (r2-3,c2-4): protocol-fix timeline ----------
ax = fig.add_subplot(G[2:4, 2:4])
events = [('ext1/ext2 train contamination', 0), ('holdout 284/73 rebuilt', 1),
          ('fam-LOO fold bug', 2), ('val split leakage', 3),
          ('extra-feat mis-routing', 4), ('label-conflict negs', 5),
          ('snapshot tag pollution', 6), ('snap39 diverged', 7)]
ax.hlines(0, -0.4, 7.4, color=K.ROSE2, lw=1.6)
for i, (nm, x_) in enumerate(events):
    col = '#FDB731' if i % 2 else '#B91E8A'
    ax.scatter([x_], [0], s=50, color=col, zorder=3)
    ax.text(x_, 0.16 if i % 2 == 0 else -0.16, nm, fontsize=4.8, ha='center',
            va='bottom' if i % 2 == 0 else 'top', color=K.DARK, rotation=18)
ax.set_ylim(-0.55, 0.55)
ax.set_xlim(-0.9, 8.0)
ax.set_xticks(range(8))
ax.set_xticklabels([f'{i+1}' for i in range(8)], fontsize=5)
ax.set_yticks([])
K.despine(ax, keep_bottom=True, keep_left=False)
lab(ax, 'E')

# ---------- D 1x2 (r2,c4): data funnel ----------
ax = fig.add_subplot(G[2, 4])
funnel = [('IUIS', 1168), ('gold', 357), ('hard neg', 9065), ('HO pairs', 313), ('ext3', 22)]
yy = np.arange(len(funnel))[::-1]
ax.hlines(yy, 0, [v for _, v in funnel], color=K.ROSE2, lw=2)
ax.scatter([v for _, v in funnel], yy, s=26, color='#B91E8A', zorder=3)
ax.set_xscale('log')
ax.set_xticks([1, 10, 100, 1000, 10000]); ax.minorticks_off()
ax.set_yticks(yy); ax.set_yticklabels([k for k, _ in funnel], fontsize=5.0)
ax.set_xlabel('rows (log)', fontsize=6.2)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_x(ax)
lab(ax, 'F')

# ---------- H 1x2 (r3,c0-2): CGI coverage ----------
ax = fig.add_subplot(G[3, 0:2])
ax.barh([0], [263], color=K.ROSE2, height=0.45)
ax.barh([0], [50], left=263, color=K.NAVY, height=0.45)
ax.text(131, 0, 'no CGI hit 263', ha='center', va='center', fontsize=5, color=K.DARK)
ax.text(288, 0, 'hit 50', ha='center', va='center', fontsize=5, color='white')
ax.set_yticks([])
ax.set_xlabel('HO pairs CGI coverage', fontsize=6.2)
ax.tick_params(labelsize=5.5)
K.despine(ax)
lab(ax, 'G')

# ---------- G 1x2 (r3,c4): model parts ----------
ax = fig.add_subplot(G[3, 4])
ME = json.load(open(D / 'care_mining_ens.json'))
views = ME['M2_view_names']
ax.barh([0], [len(views)], color='#FDB731', height=0.4)
ax.text(len(views) + 0.15, 0, f'{len(views)} views', va='center', fontsize=5.0, color=K.DARK)
ax.barh([1], [9], color='#B91E8A', height=0.4)
ax.text(9.15, 1, '9 towers', va='center', fontsize=5.0, color=K.DARK)
ax.barh([2], [4], color='#F03A6A', height=0.4)
ax.text(4.15, 2, '4 domain dims', va='center', fontsize=5.0, color=K.DARK)
ax.set_yticks([])
ax.set_xlabel('model parts', fontsize=6.2)
ax.set_xlim(0, 13)
ax.tick_params(labelsize=5.5)
K.despine(ax)
lab(ax, 'H')

# ---------- I 1x2 (r4,c4): ext3 provenance ----------
ax = fig.add_subplot(G[4, 4])
n_pmc = len({r['source'] for r in prov})
ax.bar([0, 1], [n_pmc, len(prov)], width=0.5, color=['#FDB731', '#B91E8A'])
ax.text(0, n_pmc + 0.1, str(n_pmc), ha='center', fontsize=5.2, color=K.DARK)
ax.text(1, len(prov) + 0.1, str(len(prov)), ha='center', fontsize=5.2, color=K.DARK)
ax.set_xticks([0, 1]); ax.set_xticklabels(['PMC ids', 'pairs'], fontsize=5.5)
ax.set_ylim(0, 5)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'K')

# ---------- J 1x2 (r4,c0-2): class balance ----------
ax = fig.add_subplot(G[4, 0:2])
ax.barh([0], [284], color='#F03A6A', height=0.45, label='gold pos (284)')
ax.barh([0], [9065], left=284, color=K.ROSE2, height=0.45, label='hard neg (9065)')
ax.barh([0], [3000], left=284 + 9065, color=K.LAV2, height=0.45, label='random neg (3000)')
ax.set_yticks([])
ax.set_xlabel('internal rows (pre-fold)', fontsize=6.2)
ax.legend(fontsize=4.4, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
ax.tick_params(labelsize=5.5)
K.despine(ax)
lab(ax, 'I')

# ---------- K 1x2 (r4,c2-4): fused vs per-seed HO ----------
ax = fig.add_subplot(G[4, 2:4])
S = json.load(open(D / 'care_scores_robust.json'))
from sklearn.metrics import average_precision_score
y1 = np.array(S['ext1']['y'])
sp = [average_precision_score(y1, np.array(z)) for z in S['ext1']['per_seed_raw_mean'] if z]
fe = average_precision_score(y1, np.array(S['ext1']['ens_z']))
ax.scatter(np.full(8, 0) + rng.uniform(-0.12, 0.12, 8), sp, s=26, color='#B91E8A', zorder=3)
ax.scatter([1], [fe], s=46, color='#F03A6A', marker='D', zorder=4)
ax.set_xticks([0, 1]); ax.set_xticklabels(['8 seeds', 'ens fused'], fontsize=6)
ax.set_ylabel('HO AUPRC', fontsize=6.3)
ax.tick_params(labelsize=5.5)
K.despine(ax); K.grid_y(ax)
lab(ax, 'J')

# ---------- L 1x1... wait c4 used by I. L at r4? no slot left.
# SPANS uses I at r4c4. Adjust: L replaced - fold sizes are visualized via B heatmap axes.
# Keep 11 panels; sum = 25 holds (I occupies r4c4; J c0-2; K c2-4).
fig.savefig(OUT / 'FigS2_protocol.png', dpi=600, bbox_inches='tight')
fig.savefig(OUT / 'FigS2_protocol.pdf', bbox_inches='tight')
print('FigS2 v4 done (11 panels, sum=25)')
