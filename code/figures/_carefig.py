# CARE figure kit: sandnavy palette, borderless/filled style, benchmark helpers
# (aligned with MicroRATING _v4common.py per figure-SOP rule 036).
import json, math, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---- sunsetflare palette (visual_system.md §1 落日焰霞 sunsetflare) ----
SAND   = '#FDB731'   # 琥珀 warm accent / contrast end (落日焰霞 6)
CREAM  = '#FDE9B8'   # pale amber tint (light fills, derived from FDB731)
LAV    = '#B91E8A'   # 品红 main series (落日焰霞 3)
LAV2   = '#E88CC0'   # light magenta tint (secondary/sticks)
ROSE   = '#7B1FA2'   # 紫红 middle series (落日焰霞 2)
ROSE2  = '#D9BFEA'   # light violet tint (fills)
NAVY   = '#F03A6A'   # 玫红 important result (落日焰霞 4)
DARK   = '#3D1E96'   # 靛紫 deepest anchor / text emphasis (落日焰霞 1)
PALE   = '#F5F1E8'   # near-white warm bg (sandnavy-derived, fills only)
# divergent for +/- values: NAVY <-> white <-> SAND
# sequential light->dark: CREAM -> LAV2 -> LAV -> NAVY -> DARK

SEQ = ['#FDE9B8', '#FD7E4D', '#B91E8A', '#3D1E96']   # sunsetflare light->dark
DIVERGE = ['#3D1E96', '#B91E8A', '#FDE725']   # sunsetflare purple->magenta->yellow

def style():
    rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.linewidth': 0.0,           # borderless: no spines anywhere
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.color': DARK, 'ytick.color': DARK,
        'axes.labelcolor': DARK, 'text.color': DARK,
        'axes.edgecolor': DARK,
        'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })

def despine(ax, keep_bottom=True, keep_left=False):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_visible(keep_bottom)
    if keep_left:
        ax.spines['left'].set_visible(True)

def grid_y(ax, lw=0.4, alpha=0.35):
    ax.grid(axis='y', linewidth=lw, alpha=alpha, color=LAV2)
    ax.set_axisbelow(True)

def grid_x(ax, lw=0.4, alpha=0.35):
    ax.grid(axis='x', linewidth=lw, alpha=alpha, color=LAV2)
    ax.set_axisbelow(True)

# ---- broken-axis helpers (experience.md II + rule 039) ----
def brk_marks(ax, xs, axis='x', mark='///'):
    """Double-slash break symbols at axis break positions."""
    import matplotlib.transforms as mtransforms
    if axis == 'x':
        for x in xs:
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(x, -0.02, mark, transform=trans, ha='center', va='top',
                    fontsize=7, color=DARK)
    else:
        for y in xs:
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(-0.02, y, mark, transform=trans, ha='right', va='center',
                    fontsize=7, color=DARK, rotation=90)

def boot_ci(vals, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    boots = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)

def lum(hexcolor):
    h = hexcolor.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299*r + 0.587*g + 0.114*b) / 255

def text_color_for(bg_hex):
    return 'white' if lum(bg_hex) < 0.55 else DARK

def size_norm(vals, smin, smax):
    v = np.asarray(vals, dtype=float)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return np.full(len(v), (smin + smax) / 2)
    return smin + (v - lo) / (hi - lo) * (smax - smin)

def seq_ramp(n=256):
    """sunsetflare continuous ramp (pale amber -> indigo)."""
    import matplotlib.colors as mc
    return mc.LinearSegmentedColormap.from_list('sunsetflare_seq', ['#FDE9B8', '#FDB731', '#F03A6A', '#3D1E96'])

def div_ramp():
    import matplotlib.colors as mc
    return mc.LinearSegmentedColormap.from_list('sandnavy_div', DIVERGE)

# ---- ranked lollipop with broken axis (experience.md III) ----
def ranked_lollipop(ax, labels, vals, color=LAV, anchor=(0.0, 0.05),
                    data_seg=(0.30, 0.80), hero_label=None, plain=False):
    """Two-segment broken lollipop per rule 039: EVERY row gets the compressed
    0->anchor stick + its data-segment stick. plain=True draws simple 0-based
    (NO break marks, NO rescale)."""
    order = np.argsort(-np.asarray(vals))
    labels = [labels[i] for i in order]
    v = np.asarray(vals, dtype=float)[order]
    n = len(v)
    ys = np.arange(n)[::-1]
    if hero_label:
        for yi, lb in zip(ys, labels):
            if lb == hero_label:
                ax.axhspan(yi - 0.42, yi + 0.42, color=PALE, alpha=0.75, zorder=0)
    if plain:
        for yi, vv in zip(ys, v):
            ax.plot([0, vv], [yi, yi], color=LAV2, lw=1.6, zorder=2)
            ax.scatter([vv], [yi], s=26, color=color, zorder=3)
        ax.set_xlim(0, max(v) * 1.12)
    else:
        a0, a1 = anchor
        d0, d1 = data_seg
        for yi, vv in zip(ys, v):
            if vv <= a1:
                ax.plot([0, vv], [yi, yi], color=LAV2, lw=1.6, zorder=2)
                ax.scatter([vv], [yi], s=26, color=color, zorder=3)
            else:
                ax.plot([0, a1], [yi, yi], color=LAV2, lw=1.6, zorder=2)
                ax.plot([d0, vv], [yi, yi], color=LAV2, lw=1.6, zorder=2)
                ax.scatter([vv], [yi], s=26, color=color, zorder=3)
        ax.set_xticks([a0, a1, d0, d1])
        ax.set_xticklabels([f'{t:.2f}' for t in (a0, a1, d0, d1)])
        brk_marks(ax, [(a1 + d0) / 2], axis='x')
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    despine(ax)
    grid_x(ax)


def broken_bars(ax, labels, vals, anchor=(0.0, 0.02), data_seg=(0.05, 0.60),
                color=LAV, base_ref=None):
    """Horizontal two-segment broken bars (rule 039): 0-anchored compressed
    base segment + data segment; break hatching spans all rows."""
    n = len(vals)
    ys = np.arange(n)[::-1]
    a0, a1 = anchor
    d0, d1 = data_seg
    bar_h = 0.6
    for yi, vv in zip(ys, vals):
        v1 = min(vv, a1)
        if v1 > 0:
            ax.barh(yi, v1, height=bar_h, color=color, zorder=2)
        if vv > d0:
            ax.barh(yi, vv - d0, height=bar_h, left=d0, color=color, zorder=2)
    import matplotlib.patches as mpatches
    ax.add_patch(mpatches.Rectangle((a1, -0.55), d0 - a1, n - 0.0 + 0.1,
                 facecolor='white', edgecolor=LAV2, hatch='///', lw=0, zorder=1))
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=6)
    if abs(d0 - a1) < 1e-9:   # contiguous rescale break (no discarded range)
        ax.set_xticks([a1, d1])
        ax.set_xticklabels([f'{t:.2f}' for t in (a1, d1)])
        brk_marks(ax, [a1], axis='x')
    else:
        ax.set_xticks([a0, a1, d0, d1])
        ax.set_xticklabels([f'{t:.2f}' for t in (a0, a1, d0, d1)])
        brk_marks(ax, [(a1 + d0) / 2], axis='x')
    if base_ref is not None:
        ax.axvline(base_ref, color=SAND, lw=0.8, ls='--')
    despine(ax); grid_x(ax)


# ---- ridge (peak-normalized, experience.md IV) ----
def ridge(ax, groups, base_offset=1.0, peak_h=0.45, color_hi=LAV, color_lo=LAV2):
    from scipy.stats import gaussian_kde
    for gi, (name, vals) in enumerate(groups):
        if len(vals) < 3 or np.std(vals) < 1e-9:
            continue
        xs = np.linspace(min(vals), max(vals), 160)
        kde = gaussian_kde(np.asarray(vals, dtype=float))
        dens = kde(xs)
        dens = dens / dens.max() * peak_h          # peak-normalize (rule: half row height)
        y0 = gi * base_offset
        ax.fill_between(xs, y0, y0 + dens, color=color_hi, alpha=0.45, lw=0)
        ax.plot(xs, y0 + dens, color=NAVY, lw=0.9)
        ax.axhline(y0, color=LAV2, lw=0.5)
        ax.text(0.99, y0 + peak_h * 0.7, name, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=6.5, color=DARK)
    despine(ax)

# ---- value-in-bubble matrix (experience.md III final form) ----
def bubble_matrix(ax, mat, row_labels, col_labels, cmap_lo=CREAM, cmap_hi=NAVY,
                  smin=200, smax=560, fmt='{:.2f}'):
    v = np.asarray(mat, dtype=float)
    nr, nc = v.shape
    # per-column normalization (cross-metric scales differ)
    for j in range(nc):
        col = v[:, j]
        lo, hi = col.min(), col.max()
        for i in range(nr):
            frac = (col[i] - lo) / (hi - lo) if hi > lo else 0.5
            size = smax * (0.72 + 0.28 * frac)   # size INSensitive (rule 040)
            bg = cm_lerp(cmap_lo, cmap_hi, frac)  # color SENSITIVE (full range)
            ax.scatter([j], [nr - 1 - i], s=size, color=bg, zorder=2,
                       edgecolors='none')
            tc = text_color_for(bg)
            ax.text(j, nr - 1 - i, fmt.format(col[i]), ha='center', va='center',
                    fontsize=5.6, color=tc, zorder=3)
    ax.set_xticks(range(nc)); ax.set_xticklabels(col_labels, fontsize=6, rotation=20, ha='right')
    ax.set_yticks(range(nr)); ax.set_yticklabels(row_labels[::-1], fontsize=6.5)
    ax.set_xlim(-0.6, nc - 0.4); ax.set_ylim(-0.6, nr - 0.4)
    despine(ax)

def cm_lerp(lo_hex, hi_hex, frac):
    def h2rgb(h):
        h = h.lstrip('#')
        return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)])
    lo, hi = h2rgb(lo_hex), h2rgb(hi_hex)
    c = (lo + (hi - lo) * np.clip(frac, 0, 1)).astype(int)
    return f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'

# ---- calibration curve (filled, 0-1 convention never broken) ----
def calibration(ax, y_true, y_prob, color=LAV, n_bins=10):
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    obs, pred = [], []
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob <= bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1] + 1e-9)
        if m.sum() < 3:
            continue
        obs.append(y_true[m].mean()); pred.append(y_prob[m].mean())
    ax.plot([0, 1], [0, 1], color=LAV2, lw=0.8, ls='--')
    ax.plot(pred, obs, color=color, lw=1.3, marker='o', ms=3)
    ax.fill_between(pred, obs, pred, color=color, alpha=0.15, lw=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    despine(ax); grid_y(ax)

# ---- ECDF ----
def ecdf(ax, groups, colors=None):
    for gi, (name, vals) in enumerate(groups):
        v = np.sort(np.asarray(vals))
        cdf = np.arange(1, len(v) + 1) / len(v)
        c = colors[gi] if colors else LAV
        ax.plot(v, cdf, lw=1.2, color=c, label=name)
    despine(ax); grid_y(ax)

# ---- plain horizontal bars (no break; zero-family disclosure variant) ----
def plain_bars(ax, labels, vals, color=LAV, base_ref=None):
    n = len(vals)
    ys = np.arange(n)[::-1]
    ax.barh(ys, vals, height=0.6, color=color, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=6)
    if base_ref is not None:
        ax.axvline(base_ref, color=SAND, lw=0.8, ls='--')
    despine(ax); grid_x(ax)

# ---- smoothing helpers (professor: smooth curves, robust to outliers) ----
def smooth_y(y, window=7):
    """Rolling-median then rolling-mean smoothing, reflect-padded edges
    (no edge shrink artifacts)."""
    v = np.asarray(y, dtype=float)
    if len(v) < window or window < 3:
        return v
    h = window // 2
    vp = np.concatenate((v[h:0:-1], v, v[-2:-2-h:-1]))
    med = np.array([np.median(vp[i:i + window]) for i in range(len(v))])
    medp = np.concatenate((med[h:0:-1], med, med[-2:-2-h:-1]))
    return np.convolve(medp, np.ones(window) / window, mode='valid')

def calib_smooth(ax, y_true, y_prob, color=ROSE, n_bins=10):
    """Robust calibration: wide bins -> stable means -> isotonic (monotone,
    outlier-robust) -> PCHIP smooth curve. Fill = gap between diagonal and
    smoothed curve (NOT between raw points)."""
    from sklearn.isotonic import IsotonicRegression
    obs, pred = [], []
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)
    qs = np.unique(np.quantile(yp, np.linspace(0, 1, n_bins + 1)))
    for i in range(len(qs) - 1):
        m = (yp >= qs[i]) & (yp <= qs[i + 1] + (1e-9 if i == len(qs) - 2 else 0))
        if m.sum() < 5:
            continue
        pred.append(yp[m].mean()); obs.append(yt[m].mean())
    pred, obs = np.asarray(pred), np.asarray(obs)
    iso = IsotonicRegression(out_of_bounds='clip').fit(pred, obs)
    pu = np.unique(pred)
    ou = iso.predict(pu)
    xs = np.linspace(max(pu.min(), 0.02), min(pu.max(), 0.98), 150)
    if len(pu) >= 3:
        from scipy.interpolate import PchipInterpolator
        smooth = PchipInterpolator(pu, ou)(xs)
    else:
        smooth = iso.predict(xs)
    diag = xs
    ax.plot([0, 1], [0, 1], color=LAV2, lw=0.8, ls='--')
    ax.fill_between(xs, smooth, diag, color=color, alpha=0.15, lw=0)
    ax.plot(xs, smooth, color=color, lw=1.6)
    ax.scatter(pred, obs, s=9, color=color, alpha=0.55, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    despine(ax); grid_y(ax)


def ecdf(ax, groups, colors=None):
    for gi, (name, vals) in enumerate(groups):
        v = np.sort(np.asarray(vals))
        cdf = np.arange(1, len(v) + 1) / len(v)
        c = colors[gi] if colors else LAV
        ax.plot(v, cdf, lw=1.2, color=c, label=name)
    despine(ax); grid_y(ax)

# ---- discrete-category axis centering (rule 044 universal) ----
def cat_axis(ax, n_cats, axis='x', margin=0.5):
    """Center discrete categories: n=2 -> 1/4 & 3/4; n=3 -> 1/6, 3/6, 5/6.
    Call BEFORE plotting; then place categories at 1..n_cats."""
    ticks = list(range(1, n_cats + 1))
    if axis == 'x':
        ax.set_xticks(ticks)
        ax.set_xlim(1 - margin, n_cats + margin)
    else:
        ax.set_yticks(ticks)
        ax.set_ylim(1 - margin, n_cats + margin)

# ---- smooth curve helpers (professor: smoother, fill follows smoothed curve) ----
def pchip_smooth(xs, ys, n=200, every=3):
    """PCHIP smoothing on deduped, sorted anchors (strictly increasing x)."""
    from scipy.interpolate import PchipInterpolator
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    keep = np.concatenate([[True], np.diff(xs) > 0])
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 2:
        return np.asarray([]), np.asarray([])
    p = PchipInterpolator(xs, ys)
    xn = np.linspace(xs.min(), xs.max(), n)
    return xn, p(xn)

def kecdf(xs_grid, vals):
    """Kernel-smoothed ECDF (monotone, no staircase)."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(np.asarray(vals, dtype=float))
    cdf = np.array([kde.integrate_box_1d(-np.inf, x) for x in xs_grid])
    return cdf


# ---- grid coverage assert (SOP stage-5 mechanical check) ----
def grid_check(spans, R, C):
    """spans: list of (name, r0, r1, c0, c1) half-open. Assert no overlap + full coverage."""
    grid = [[0] * C for _ in range(R)]
    for name, r0, r1, c0, c1 in spans:
        for r in range(r0, r1):
            for c in range(c0, c1):
                assert grid[r][c] == 0, f'overlap at r{r}c{c}: {name}'
                grid[r][c] = 1
    total = sum(sum(row) for row in grid)
    assert total == R * C, f'unfilled slots: {R*C - total}'
    return True
