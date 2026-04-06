"""
AI-generated code.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

os.makedirs("figures", exist_ok=True)

data = np.load("errordiagresults.npy", allow_pickle=True).item()

sum_as     = data["as"].sum(axis=1)
sum_bs     = data["bs"].sum(axis=1)
n_zeros_bs = (data["bs"] == 0).sum(axis=1)

ref_errors = np.load("errordistributionCLR.npy")
ref_sum_errors = ref_errors.sum(axis=1)
lo = np.percentile(ref_sum_errors, 0.5)
hi = np.percentile(ref_sum_errors, 99.5)

def make_cells(err1, err2):
    """Classify simulations into 4 cells.
    'Inside' = sum error within [0.5th, 99.5th] percentile of errordistributionCLR.
    """
    valid = ~(np.isnan(err1) | np.isnan(err2))
    inside1 = (err1 >= lo) & (err1 <= hi)
    inside2 = (err2 >= lo) & (err2 <= hi)
    return [
        valid &  inside1 &  inside2,   # both inside
        valid & ~inside1 &  inside2,   # CL(P) outside, CL(R̂) inside
        valid &  inside1 & ~inside2,   # CL(P) inside, CL(R̂) outside
        valid & ~inside1 & ~inside2,   # both outside
    ]

err_P     = data["sum_err_P"]
err_J     = data["sum_err_J"]
err_JP    = data["sum_err_JP"]
err_JPmin = data["sum_err_JPmin"]

cells_J     = make_cells(err_P, err_J)
cells_JP    = make_cells(err_P, err_JP)
cells_JPmin = make_cells(err_P, err_JPmin)

better_J     = np.abs(err_J)     < np.abs(err_P)
better_JP    = np.abs(err_JP)    < np.abs(err_P)
better_JPmin = np.abs(err_JPmin) < np.abs(err_P)

# Per-method metadata
METHODS = [
    ("J",     cells_J,     better_J,     f"Known $J$ ($\\gamma={data['GAMMA_J']}$)",                    "J"),
    ("JP",    cells_JP,    better_JP,    f"$J_P$ ($\\gamma={data['GAMMA_J_P']}$)",                      "JP"),
    ("JPmin", cells_JPmin, better_JPmin, f"$J_{{P,\\min}}$ ($\\gamma={data['GAMMA_J_P_MIN']}$)",        "JPmin"),
]

CELL_LABELS = [
    "Both inside",
    "CL(P) outside, CL($\\hat{R}$) inside",
    "CL(P) inside, CL($\\hat{R}$) outside",
    "Both outside",
]

COLORS = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]  # blue, green, red, orange


# --- 3D density figures (one per method) ---
def plot_cells_3d(cells, title, fig):
    fig.suptitle(title, y=0.98, fontsize=11)
    for i, (mask, label) in enumerate(zip(cells, CELL_LABELS)):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        x, y = sum_as[mask], sum_bs[mask]
        ax.set_title(f"{label}\n(n={mask.sum()})", fontsize=8, pad=4)
        ax.set_xlabel(r"$\sum a_s$", fontsize=7, labelpad=2)
        ax.set_ylabel(r"$\sum b_s$", fontsize=7, labelpad=2)
        ax.set_zlabel("density", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        if mask.sum() < 10:
            continue
        H, xedges, yedges = np.histogram2d(x, y, bins=30, density=True)
        xc = (xedges[:-1] + xedges[1:]) / 2
        yc = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(xc, yc)
        ax.plot_surface(X, Y, H.T, cmap="Blues", edgecolor="none", alpha=0.9)

for _, cells, _, method_label, fname_suffix in METHODS:
    fig = plt.figure(figsize=(13, 9))
    plot_cells_3d(cells, r"Density of $(\sum a_s,\ \sum b_s)$" + f" — {method_label}", fig)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.3, hspace=0.35)
    # fig.savefig(f"figures/diag_3d_{fname_suffix}.png", dpi=150, bbox_inches="tight")
# plt.show()


# --- Most likely outcome map: (sum_a, sum_b) ---
def most_likely_outcome_map(cells, ax, title, bins=50):
    x, y = sum_as, sum_bs
    x_edges = np.linspace(x.min(), x.max(), bins + 1)
    y_edges = np.linspace(y.min(), y.max(), bins + 1)

    counts = np.zeros((4, bins, bins), dtype=int)
    for k, mask in enumerate(cells):
        counts[k], _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])

    total = counts.sum(axis=0)
    winner = np.argmax(counts, axis=0)

    img = np.ones((bins, bins, 4))
    cmap_colors = [mcolors.to_rgba(c) for c in COLORS]
    for i in range(bins):
        for j in range(bins):
            if total[i, j] > 0:
                img[i, j] = cmap_colors[winner[i, j]]

    ax.imshow(
        img.transpose(1, 0, 2),
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel(r"$\sum a_s$", fontsize=9)
    ax.set_ylabel(r"$\sum b_s$", fontsize=9)
    ax.set_title(title, fontsize=10)
    patches = [mpatches.Patch(color=COLORS[k], label=CELL_LABELS[k]) for k in range(4)]
    ax.legend(handles=patches, fontsize=7, loc="upper right")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (_, cells, _, method_label, _) in zip(axes, METHODS):
    most_likely_outcome_map(cells, ax, f"Most Likely Outcome — {method_label}")
fig.tight_layout()
# fig.savefig("figures/diag_most_likely_outcome.png", dpi=150, bbox_inches="tight")
# plt.show()


# --- Probability heatmaps per cell (one figure per cell, 3 subplots per figure) ---
CELL_FILENAMES = [
    "diag_prob_both_inside",
    "diag_prob_P_outside",
    "diag_prob_Rhat_outside",
    "diag_prob_both_outside",
]

def prob_heatmap_row(all_cells, cell_idx, cell_label, filename, bins=25):
    x = sum_as
    y = sum_bs
    x_edges = np.linspace(x.min(), x.max(), bins + 1)
    y_edges = np.linspace(y.min(), y.max(), bins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(cell_label, fontsize=12)

    for i, (ax, (_, cells, _, method_label, _)) in enumerate(zip(axes, all_cells)):
        total_counts = np.zeros((bins, bins), dtype=int)
        cell_counts  = np.zeros((bins, bins), dtype=int)
        for k, mask in enumerate(cells):
            h, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])
            total_counts += h.astype(int)
            if k == cell_idx:
                cell_counts = h.astype(int)

        prob = np.full((bins, bins), np.nan)
        mask_nonempty = total_counts > 0
        prob[mask_nonempty] = cell_counts[mask_nonempty] / total_counts[mask_nonempty]

        im = ax.imshow(
            prob.T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="auto",
            interpolation="nearest",
            cmap="YlOrRd",
            norm=plt.matplotlib.colors.PowerNorm(gamma=0.4, vmin=0, vmax=1),
        )
        ax.set_xlabel(r"$\sum a_s$", fontsize=10)
        ax.set_title(method_label, fontsize=10)
        if i > 0:
            ax.set_yticklabels([])

    fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.12, wspace=0.08)
    fig.text(0.01, 0.5, r"$\sum b_s$", va="center", rotation="vertical", fontsize=10)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Probability", fontsize=10)
    # fig.savefig(f"figures/{filename}.png", dpi=150, bbox_inches="tight")
    # plt.show()

for idx, (label, fname) in enumerate(zip(CELL_LABELS, CELL_FILENAMES)):
    prob_heatmap_row(METHODS, idx, label, fname)


# --- Summary tables ---
CELL_NOTES = ["Both inside", "P outside only", "R̂ outside only", "Both outside"]

print(f"\nReference interval: [{lo:.1f}, {hi:.1f}]")
print(f"{'─'*62}")

for name, cells, better, method_label, _ in METHODS:
    total = sum(m.sum() for m in cells)
    print(f"\n{method_label}  (n={total})")
    print(f"  {'Cell':<22} {'Count':>7}  {'Share':>6}  Note")
    print(f"  {'─'*55}")
    for i, (mask, note) in enumerate(zip(cells, CELL_NOTES)):
        count = mask.sum()
        share = count / total if total > 0 else 0
        if i in (0, 3) and count > 0:
            pct_worse = (~better[mask]).sum() / count
            extra = f"R̂ worse in {pct_worse:.1%} of cases"
        elif i == 1:
            extra = "Recovery helps"
        else:
            extra = "Recovery worsens"
        print(f"  {note:<22} {count:>7}  {share:>5.1%}  {extra}")

# --- Scatter plots: features vs error metrics ---
rel_err_J     = data["rel_err_J"]
rel_err_JP    = data["rel_err_JP"]
rel_err_JPmin = data["rel_err_JPmin"]

as_      = data["as"]
bs_      = data["bs"]
J_hats   = data["J_hats"]
J_P_mins = data["J_P_mins"]
B_ts     = data["B_ts"]

threshold = 0.2

def longest_run_below(arr, thr):
    below = (arr < thr).astype(int)
    max_run = current = 0
    for v in below:
        current = current + 1 if v else 0
        max_run = max(max_run, current)
    return max_run

mean_as      = as_.mean(axis=1)
mean_bs      = bs_.mean(axis=1)
std_as       = as_.std(axis=1)
min_at       = as_.min(axis=1)
n_small      = (as_ < threshold).sum(axis=1)
longest_run  = np.array([longest_run_below(row, threshold) for row in as_])
std_bt       = bs_.std(axis=1)
n_overflow   = (bs_ > 0).sum(axis=1)

scatter_features = [
    (mean_as,              r"$\mathrm{mean}(a_t)$"),
    (std_as,               r"$\mathrm{std}(a_t)$"),
    (min_at,               r"$\min(a_t)$"),
    (n_small,              f"# $a_t <$ {threshold}"),
    (longest_run,          f"Longest run $a_t <$ {threshold}"),
    (mean_as + mean_bs,    r"$\mathrm{mean}(a_t) + \mathrm{mean}(b_t)$"),
    (std_bt,               r"$\mathrm{std}(b_t)$"),
    (n_overflow,           r"# $b_t > 0$"),
]

scatter_method_groups = [
    (rel_err_J,    r"$|\mathrm{err}(\mathrm{CL}(\hat{R},J)) - \mathrm{err}(\mathrm{CL}(R))|$"),
    (err_J,        r"$\mathrm{err}(\mathrm{CL}(\hat{R},J))$"),
    (rel_err_JP,   r"$|\mathrm{err}(\mathrm{CL}(\hat{R},J_P)) - \mathrm{err}(\mathrm{CL}(R))|$"),
    (err_JP,       r"$\mathrm{err}(\mathrm{CL}(\hat{R},J_P))$"),
    (rel_err_JPmin, r"$|\mathrm{err}(\mathrm{CL}(\hat{R},J_{P,\min})) - \mathrm{err}(\mathrm{CL}(R))|$"),
    (err_JPmin,    r"$\mathrm{err}(\mathrm{CL}(\hat{R},J_{P,\min}))$"),
]

_n_cols = 4
_n_rows = int(np.ceil(len(scatter_features) / _n_cols))

for _i, (_err_vals, _ylabel) in enumerate(scatter_method_groups):
    _valid  = ~np.isnan(_err_vals)
    _y_vals = np.abs(_err_vals[_valid])
    fig, axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 4.5, _n_rows * 5))
    axes = axes.flatten()
    fig.suptitle(_ylabel + " vs features", fontsize=13)
    for ax, (fvals, flabel) in zip(axes, scatter_features):
        ax.scatter(fvals[_valid], _y_vals, s=3, alpha=0.3, color="#1565C0")
        ax.set_xlabel(flabel, fontsize=12)
        ax.set_yscale("log")
        ax.tick_params(labelsize=9)
    for ax in axes[len(scatter_features):]:
        ax.set_visible(False)
    fig.tight_layout()
    # fig.savefig(f"figures/diag_scatter_{_i}.png", dpi=150, bbox_inches="tight")
    # plt.show()


