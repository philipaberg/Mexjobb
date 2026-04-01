import sys
import os
import numpy as np
import matplotlib.pyplot as plt

results    = np.load(sys.argv[1], allow_pickle=True).item()
errors     = results["errors"]
ETA_LIST   = results["ETA_LIST"]
GAMMA_LIST = results["GAMMA_LIST"]

tag = os.path.splitext(os.path.basename(sys.argv[1]))[0].replace("results_", "")

outdir = "figures"
os.makedirs(outdir, exist_ok=True)

colors = ["#4C72B0", "#DD8452", "#55A868"]
labels = ["CL(P)", "CL(R)", r"CL($\hat{R}$)"]
n_occ  = np.stack(errors[(ETA_LIST[0], GAMMA_LIST[0])]).shape[2]


def savefig(name):
    plt.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    plt.close()


for eta in ETA_LIST:
    for gamma in GAMMA_LIST:
        data         = np.stack(errors[(eta, gamma)])
        total_errors = data.sum(axis=2)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        fig.suptitle(fr"Distribution of total reserve error  ($\eta$={eta}, $\gamma$={gamma})", fontsize=13)
        for k, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            vals = total_errors[:, k].astype(float)
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            ax.hist(vals, bins=60, alpha=0.7, color=color, density=True, range=(lo, hi))
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axvline(vals.mean(), color="red", linewidth=1.8, linestyle=":", alpha=0.9, label=f"Mean = {vals.mean():.0f}")
            ax.set_xlim(lo, hi)
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("Total reserve error  (predicted \u2212 true)")
            ax.legend(fontsize=9)
        axes[0].set_ylabel("Density")
        plt.tight_layout()
        savefig(f"{tag}_hist_total_eta{eta}_gamma{gamma}")


for gamma in GAMMA_LIST:
    fig, axes = plt.subplots(1, len(ETA_LIST), figsize=(5 * len(ETA_LIST), 4), sharey=True, squeeze=False)
    fig.suptitle(fr"Bias per occurrence period  ($\gamma$={gamma})", fontsize=13)
    for ax, eta in zip(axes[0], ETA_LIST):
        data = np.stack(errors[(eta, gamma)])
        bias = data.mean(axis=0)
        x = np.arange(n_occ)
        for k in range(3):
            ax.plot(x, bias[k], marker="o", markersize=3, color=colors[k], label=labels[k])
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(fr"$\eta$ = {eta:.2f}", fontsize=11)
        ax.set_xlabel("Occurrence period (relative)")
    axes[0][0].set_ylabel(r"Bias  $E[\hat{R} - R]$")
    axes[0][-1].legend(fontsize=9)
    plt.tight_layout()
    savefig(f"{tag}_bias_gamma{gamma}")


for eta in ETA_LIST:
    for gamma in GAMMA_LIST:
        data      = np.stack(errors[(eta, gamma)])
        last_vals = data[:, :, -1]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        fig.suptitle(fr"Error distribution, last occurrence period ($i={n_occ}$)  ($\eta$={eta}, $\gamma$={gamma})", fontsize=13)
        for k, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            vals = last_vals[:, k].astype(float)
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            ax.hist(vals, bins=60, alpha=0.7, color=color, density=True, range=(lo, hi))
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axvline(vals.mean(), color="red", linewidth=1.8, linestyle=":", alpha=0.9, label=f"Mean = {vals.mean():.0f}")
            ax.set_xlim(lo, hi)
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("Reserve error  (predicted \u2212 true)")
            ax.legend(fontsize=9)
        axes[0].set_ylabel("Density")
        plt.tight_layout()
        savefig(f"{tag}_hist_last_eta{eta}_gamma{gamma}")


for eta in ETA_LIST:
    for gamma in GAMMA_LIST:
        data         = np.stack(errors[(eta, gamma)])
        total_errors = data.sum(axis=2)
        msep         = (total_errors ** 2).mean(axis=0)

        fig, ax = plt.subplots(figsize=(6, 5))
        for k in range(3):
            ax.bar(k, msep[k], width=0.5, color=colors[k], alpha=0.8, label=labels[k])
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"Total MSEP  $E[(\hat{R}-R)^2]$")
        ax.set_title(fr"Total MSEP — all methods  ($\eta$={eta}, $\gamma$={gamma})", fontsize=12)
        ax.legend()
        plt.tight_layout()
        savefig(f"{tag}_msep_all_eta{eta}_gamma{gamma}")

        fig, ax = plt.subplots(figsize=(5, 5))
        for idx, k in enumerate([1, 2]):
            ax.bar(idx, msep[k], width=0.5, color=colors[k], alpha=0.8, label=labels[k])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([labels[1], labels[2]])
        ax.set_ylabel(r"Total MSEP  $E[(\hat{R}-R)^2]$")
        ax.set_title(fr"Total MSEP — CL(R) vs CL($\hat{{R}}$)  ($\eta$={eta}, $\gamma$={gamma})", fontsize=12)
        ax.legend()
        plt.tight_layout()
        savefig(f"{tag}_msep_clr_eta{eta}_gamma{gamma}")

print("Done.")
