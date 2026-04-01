"""
AI-generated code for plotting.
"""


"""
Saves plots studying the effect of gamma on CL(R_hat) (recovered reportings only).

Plot 1: Distribution of total reserve error — one figure per eta,
        subplots for each gamma.

Plot 2: Bias per occurrence period — one figure, subplots for each eta,
        lines for each gamma.

Plot 3: MSEP bar chart — one figure per eta, bars for each gamma.

Two datasets:
  A) Known J  : results_known_j.npy
                ETA=[1.05, 1.1, 1.2, 1.5], GAMMA=[0, 1.0, 10.0, 100.0]

  B) Est. J   : gammaresults_est_j.npy (GAMMA=[0.01, 0.05, 0.1])
              + results_est_j.npy      (gamma=0, eta != 1.5)
                combined ETA=[1.05, 1.1, 1.2], GAMMA=[0, 0.01, 0.05, 0.1]
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────────────────────────────
res_kj  = np.load("results_known_j.npy",      allow_pickle=True).item()
res_gej = np.load("gammaresults_est_j.npy",   allow_pickle=True).item()
res_ej  = np.load("results_est_j.npy",        allow_pickle=True).item()

ETA_KJ   = res_kj["ETA_LIST"]    # [1.05, 1.1, 1.2, 1.5]
GAMMA_KJ = res_kj["GAMMA_LIST"]  # [0, 1.0, 10.0, 100.0]
err_kj   = res_kj["errors"]

# Combine est-J: gamma=0 from results_est_j (eta != 1.5) + gammaresults_est_j
ETA_EJ   = [e for e in res_gej["ETA_LIST"]]   # [1.05, 1.1, 1.2]
GAMMA_EJ = [0] + list(res_gej["GAMMA_LIST"])  # [0, 0.01, 0.05, 0.1]
err_ej   = {}
for eta in ETA_EJ:
    err_ej[(eta, 0)] = res_ej["errors"][(eta, 0)]
    for gamma in res_gej["GAMMA_LIST"]:
        err_ej[(eta, gamma)] = res_gej["errors"][(eta, gamma)]

# ── Helpers ───────────────────────────────────────────────────────────────────
outdir = "figures"
os.makedirs(outdir, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    plt.close()

n_occ = np.stack(err_kj[(ETA_KJ[0], GAMMA_KJ[0])]).shape[2]
x_occ = np.arange(n_occ)

# Colors: one per gamma
COLORS_KJ = ["#7f7f7f", "#4C72B0", "#DD8452", "#55A868"]   # 4 gammas
COLORS_EJ = ["#7f7f7f", "#4C72B0", "#DD8452", "#55A868"]   # 4 gammas


# ── Plot 1: Histogram of total reserve error (recovered only) ─────────────────
def plot_hist(eta_list, gamma_list, errors, colors, tag):
    for eta in eta_list:
        n_g = len(gamma_list)
        fig, axes = plt.subplots(1, n_g, figsize=(5 * n_g, 5), sharey=False)
        fig.suptitle(
            fr"Distribution of total reserve — CL($\hat{{R}}$)  ($\eta={eta}$)",
            fontsize=13,
        )
        for ax, gamma, color in zip(axes, gamma_list, colors):
            data = np.stack(errors[(eta, gamma)])   # (N_SIM, 3, n_occ)
            vals = data[:, 2, :].sum(axis=1).astype(float)
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            ax.hist(vals, bins=60, alpha=0.7, color=color, density=True,
                    range=(lo, hi))
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axvline(vals.mean(), color="red", linewidth=1.8, linestyle=":",
                       alpha=0.9, label=f"Mean = {vals.mean():.0f}")
            ax.set_xlim(lo, hi)
            ax.set_title(fr"$\gamma = {gamma}$", fontsize=12)
            ax.set_xlabel(r"Total reserve error  (predicted $-$ true)")
            ax.legend(fontsize=9)
        axes[0].set_ylabel("Density")
        plt.tight_layout()
        savefig(f"gamma_{tag}_hist_eta{eta}")

plot_hist(ETA_KJ, GAMMA_KJ, err_kj, COLORS_KJ, "known_j")
plot_hist(ETA_EJ, GAMMA_EJ, err_ej, COLORS_EJ, "est_j")


# ── Plot 2: Bias per occurrence period (recovered only) ───────────────────────
def plot_bias(eta_list, gamma_list, errors, colors, tag):
    fig, axes = plt.subplots(
        1, len(eta_list),
        figsize=(5 * len(eta_list), 4),
        sharey=True, squeeze=False,
    )
    fig.suptitle(
        fr"Bias per occurrence period — CL($\hat{{R}}$)",
        fontsize=13,
    )
    for ax, eta in zip(axes[0], eta_list):
        for gamma, color in zip(gamma_list, colors):
            data = np.stack(errors[(eta, gamma)])
            bias = data[:, 2, :].mean(axis=0)    # (n_occ,)
            ax.plot(x_occ, bias, marker="o", markersize=3, color=color,
                    label=fr"$\gamma={gamma}$")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(fr"$\eta = {eta:.2f}$", fontsize=11)
        ax.set_xlabel("Occurrence period (relative)")
    axes[0][0].set_ylabel(r"Bias  $\mathrm{E}[\hat{R} - R]$")
    axes[0][-1].legend(fontsize=9)
    plt.tight_layout()
    savefig(f"gamma_{tag}_bias")

plot_bias(ETA_KJ, GAMMA_KJ, err_kj, COLORS_KJ, "known_j")
plot_bias(ETA_EJ, GAMMA_EJ, err_ej, COLORS_EJ, "est_j")


# ── Plot 3: MSEP bar chart per eta (recovered only) ───────────────────────────
def plot_msep(eta_list, gamma_list, errors, colors, tag):
    for eta in eta_list:
        mseps = []
        for gamma in gamma_list:
            data = np.stack(errors[(eta, gamma)])
            total_err = data[:, 2, :].sum(axis=1)
            mseps.append(float((total_err ** 2).mean()))
        fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(gamma_list) + 2), 5))
        x = np.arange(len(gamma_list))
        ax.bar(x, mseps, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([fr"$\gamma={g}$" for g in gamma_list])
        ax.set_ylabel(r"Total MSEP  $\mathrm{E}[(\hat{R}-R)^2]$")
        ax.set_title(
            fr"MSEP vs $\gamma$ — CL($\hat{{R}}$)  ($\eta={eta}$)",
            fontsize=12,
        )
        plt.tight_layout()
        savefig(f"gamma_{tag}_msep_eta{eta}")

plot_msep(ETA_KJ, GAMMA_KJ, err_kj, COLORS_KJ, "known_j")
plot_msep(ETA_EJ, GAMMA_EJ, err_ej, COLORS_EJ, "est_j")

print("Done.")
