"""
Visualize simulation results from datageneration.py.
Usage:
    python plots.py results_known_j.npy          # plots all eta/gamma combos
    python plots.py results_known_j.npy 1.10 10.0 # pick specific eta, gamma
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# ── Load ─────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python plots.py <results.npy> [eta] [gamma]")
    sys.exit(1)

results    = np.load(sys.argv[1], allow_pickle=True).item()
errors     = results["errors"]
ETA_LIST   = results["ETA_LIST"]
GAMMA_LIST = results["GAMMA_LIST"]

eta_arg   = float(sys.argv[2]) if len(sys.argv) > 2 else None
gamma_arg = float(sys.argv[3]) if len(sys.argv) > 3 else None

# If eta/gamma given, validate and narrow the lists
if eta_arg is not None:
    if eta_arg not in ETA_LIST:
        print(f"eta={eta_arg} not in {ETA_LIST}")
        sys.exit(1)
    ETA_PLOT = [eta_arg]
else:
    ETA_PLOT = ETA_LIST

if gamma_arg is not None:
    if gamma_arg not in GAMMA_LIST:
        print(f"gamma={gamma_arg} not in {GAMMA_LIST}")
        sys.exit(1)
    GAMMA_PLOT = [gamma_arg]
else:
    GAMMA_PLOT = GAMMA_LIST

colors = ["#4C72B0", "#DD8452", "#55A868"]
labels = ["CL(P)", "CL(R)", "CL(R̂)"]
n_occ  = np.stack(errors[(ETA_LIST[0], GAMMA_LIST[0])]).shape[2]

LAST_N_MSEP = 5


# ── 1. Histogram of total reserve error ──────────────────────────────────────
for eta in ETA_PLOT:
    for gamma in GAMMA_PLOT:
        data         = np.stack(errors[(eta, gamma)])   # (N_SIM, 3, n_occ)
        total_errors = data.sum(axis=2)                 # (N_SIM, 3)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        fig.suptitle(f"Distribution of total reserve error  (η={eta}, γ={gamma})", fontsize=13)

        for k, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            vals = total_errors[:, k].astype(float)
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            ax.hist(vals, bins=60, alpha=0.7, color=color, density=True, range=(lo, hi))
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axvline(vals.mean(), color="red", linewidth=1.8, linestyle=":",
                       alpha=0.9, label=f"Mean = {vals.mean():.0f}")
            ax.set_xlim(lo, hi)
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("Total reserve error  (predicted − true)")
            ax.legend(fontsize=9)

        axes[0].set_ylabel("Density")
        plt.tight_layout()
        plt.show()


# ── 2. Bias per occurrence period ────────────────────────────────────────────
for gamma in GAMMA_PLOT:
    fig, axes = plt.subplots(1, len(ETA_PLOT), figsize=(5 * len(ETA_PLOT), 4),
                             sharey=True, squeeze=False)
    fig.suptitle(f"Bias per occurrence period  (γ={gamma})", fontsize=13)

    for ax, eta in zip(axes[0], ETA_PLOT):
        data = np.stack(errors[(eta, gamma)])
        bias = data.mean(axis=0)
        x = np.arange(n_occ)

        for k in range(3):
            ax.plot(x, bias[k], marker="o", markersize=3, color=colors[k], label=labels[k])

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(f"η = {eta:.2f}", fontsize=11)
        ax.set_xlabel("Occurrence period (relative)")

    axes[0][0].set_ylabel("Bias  E[predicted − true]")
    axes[0][-1].legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ── 3. Histogram of errors, last occurrence period ──────────────────────────
for eta in ETA_PLOT:
    for gamma in GAMMA_PLOT:
        data      = np.stack(errors[(eta, gamma)])
        last_vals = data[:, :, -1]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        fig.suptitle(f"Error distribution, last occurrence period (i={n_occ})  "
                     f"(η={eta}, γ={gamma})", fontsize=13)

        for k, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            vals = last_vals[:, k].astype(float)
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            ax.hist(vals, bins=60, alpha=0.7, color=color, density=True, range=(lo, hi))
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axvline(vals.mean(), color="red", linewidth=1.8, linestyle=":",
                       alpha=0.9, label=f"Mean = {vals.mean():.0f}")
            ax.set_xlim(lo, hi)
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("Reserve error  (predicted − true)")
            ax.legend(fontsize=9)

        axes[0].set_ylabel("Density")
        plt.tight_layout()
        plt.show()


# ── 4a. Total MSEP — all 3 methods ──────────────────────────────────────────
for eta in ETA_PLOT:
    for gamma in GAMMA_PLOT:
        data         = np.stack(errors[(eta, gamma)])
        total_errors = data.sum(axis=2)          # (N_SIM, 3)
        msep         = (total_errors ** 2).mean(axis=0)  # (3,)

        fig, ax = plt.subplots(figsize=(6, 5))
        x     = np.arange(3)
        width = 0.5

        for k in range(3):
            ax.bar(x[k], msep[k], width=width, color=colors[k], alpha=0.8, label=labels[k])

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Total MSEP  E[(predicted − true)²]")
        ax.set_title(f"Total MSEP — all methods  (η={eta}, γ={gamma})", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.show()


# ── 4b. Total MSEP — CL(R) vs CL(R̂) only ───────────────────────────────────
for eta in ETA_PLOT:
    for gamma in GAMMA_PLOT:
        data         = np.stack(errors[(eta, gamma)])
        total_errors = data.sum(axis=2)          # (N_SIM, 3)
        msep         = (total_errors ** 2).mean(axis=0)  # (3,)

        fig, ax = plt.subplots(figsize=(5, 5))
        x      = np.arange(2)
        width2 = 0.5

        for idx, k in enumerate([1, 2]):
            ax.bar(x[idx], msep[k], width=width2, color=colors[k], alpha=0.8, label=labels[k])

        ax.set_xticks(x)
        ax.set_xticklabels([labels[1], labels[2]])
        ax.set_ylabel("Total MSEP  E[(predicted − true)²]")
        ax.set_title(f"Total MSEP — CL(R) vs CL(R̂)  (η={eta}, γ={gamma})", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.show()
