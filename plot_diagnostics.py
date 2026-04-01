"""
Completely AI-generated code.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

data = np.load("when_to_use_results.npy", allow_pickle=True).item()

sum_as      = data["sum_as"]
n_zeros_bs  = data["n_zeros_bs"]
sum_bs      = data["sum_bs"]
cells_kj    = data["cells_kj"]   # (both_inside, P_out_only, Rhat_out_only, both_outside)
cells_ej    = data["cells_ej"]

CELL_LABELS = [
    "Both inside",
    "CL(P) outside, CL($\\hat{R}$) inside",
    "CL(P) inside, CL($\\hat{R}$) outside",
    "Both outside",
]


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


# --- Figur 1: Known J — 3D density ---
fig1 = plt.figure(figsize=(13, 9))
plot_cells_3d(
    cells_kj,
    r"Density of $(\sum a_s,\ \sum b_s)$" + f" — Known $J$ ($\\gamma={data['GAMMA_KNOWN_J']}$)",
    fig1,
)
fig1.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.3, hspace=0.35)

# --- Figur 2: Estimated J — 3D density ---
fig2 = plt.figure(figsize=(13, 9))
plot_cells_3d(
    cells_ej,
    r"Density of $(\sum a_s,\ \sum b_s)$" + f" — Estimated $\\hat{{J}}$ ($\\gamma={data['GAMMA_EST_J']}$)",
    fig2,
)
fig2.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.3, hspace=0.35)

fig1.savefig("figures/diag_3d_kj.pdf", bbox_inches="tight")
fig2.savefig("figures/diag_3d_ej.pdf", bbox_inches="tight")
plt.show()

# --- Figur 3 & 4: Troligaste utfall givet (sum_a, sum_b) ---
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

COLORS = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]  # blue, green, red, orange

def most_likely_outcome_map(cells, ax, title, bins=60):
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

fig3, axes = plt.subplots(1, 2, figsize=(13, 5))
most_likely_outcome_map(
    cells_kj, axes[0],
    f"Most Likely Outcome — Known $J$ ($\\gamma={data['GAMMA_KNOWN_J']}$)",
)
most_likely_outcome_map(
    cells_ej, axes[1],
    f"Most Likely Outcome — Estimated $\\hat{{J}}$ ($\\gamma={data['GAMMA_EST_J']}$)",
)
fig3.tight_layout()
fig3.savefig("figures/diag_most_likely.pdf", bbox_inches="tight")
plt.show()

# --- Figur 5 & 6: Troligaste utfall med n_zeros_bs på y-axeln ---
def most_likely_outcome_map_nzeros(cells, ax, title, bins=60):
    x, y = sum_as, n_zeros_bs
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
    ax.set_ylabel(r"$\#\{b_s = 0\}$", fontsize=9)
    ax.set_title(title, fontsize=10)

    patches = [mpatches.Patch(color=COLORS[k], label=CELL_LABELS[k]) for k in range(4)]
    ax.legend(handles=patches, fontsize=7, loc="upper right")

fig5, axes = plt.subplots(1, 2, figsize=(13, 5))
most_likely_outcome_map_nzeros(
    cells_kj, axes[0],
    f"Most Likely Outcome ($\\#\\{{b_s=0\\}}$) — Known $J$ ($\\gamma={data['GAMMA_KNOWN_J']}$)",
)
most_likely_outcome_map_nzeros(
    cells_ej, axes[1],
    f"Most Likely Outcome ($\\#\\{{b_s=0\\}}$) — Estimated $\\hat{{J}}$ ($\\gamma={data['GAMMA_EST_J']}$)",
)
fig5.tight_layout()
fig5.savefig("figures/diag_most_likely_nzeros.pdf", bbox_inches="tight")
plt.show()

# --- Figur 6–9: Probability heatmaps per cell ---
CELL_FILENAMES = [
    "diag_prob_both_inside",
    "diag_prob_P_outside",
    "diag_prob_Rhat_outside",
    "diag_prob_both_outside",
]

def prob_heatmap(cells_kj, cells_ej, cell_idx, cell_label, filename, bins=25):
    x = sum_as
    y = sum_bs
    x_edges = np.linspace(x.min(), x.max(), bins + 1)
    y_edges = np.linspace(y.min(), y.max(), bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(cell_label, fontsize=12)

    for ax, cells, method_label in zip(
        axes,
        [cells_kj, cells_ej],
        [f"Known $J$ ($\\gamma={data['GAMMA_KNOWN_J']}$)",
         f"Estimated $\\hat{{J}}$ ($\\gamma={data['GAMMA_EST_J']}$)"],
    ):
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
        fig.colorbar(im, ax=ax, label="Probability")
        ax.set_xlabel(r"$\sum a_s$", fontsize=10)
        ax.set_ylabel(r"$\sum b_s$", fontsize=10)
        ax.set_title(method_label, fontsize=10)

    fig.tight_layout()
    fig.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    plt.show()

for idx, (label, fname) in enumerate(zip(CELL_LABELS, CELL_FILENAMES)):
    prob_heatmap(cells_kj, cells_ej, idx, label, fname)
