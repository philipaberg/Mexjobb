"""
Usage: python plots.py etagammaresults.npy
AI-generated
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#os.makedirs("figures", exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python plots.py <results.npy>")
    sys.exit(1)

# Load the results file
results    = np.load(sys.argv[1], allow_pickle=True).item()
errors     = results["errors"]
ETA_LIST   = results["ETA_LIST"]
GAMMA_LIST = results["GAMMA_LIST"]

# Method names, in the order they are stored in the errors array
labels = ["CL(P)", "CL(R)", "CL(R̂, J)", "CL(R̂, J_P)", "CL(R̂, J_P_min)"]

# LaTeX versions for plot titles and tick labels
latex_labels = [
    r"$\mathrm{CL}(P)$",
    r"$\mathrm{CL}(R)$",
    r"$\mathrm{CL}(\hat{R}, J)$",
    r"$\mathrm{CL}(\hat{R}, J_P)$",
    r"$\mathrm{CL}(\hat{R}, J_{P,\min})$",
]

# CL(P) and CL(R) do not depend on gamma, so we use any gamma to look them up
any_gamma = GAMMA_LIST[0]


# ── Helper: compute mean or RMSE for one (eta, gamma) pair ───────────────────

def get_mean(eta, gamma):
    data         = np.stack(errors[(eta, gamma)])  # shape: (N_SIM, n_methods, n_occ)
    total_errors = data.sum(axis=2)                # sum over accident years → (N_SIM, n_methods)
    return total_errors.mean(axis=0)               # average over simulations → (n_methods,)

def get_rmse(eta, gamma):
    data         = np.stack(errors[(eta, gamma)])      # shape: (N_SIM, n_methods, n_occ)
    total_errors = data.sum(axis=2)                    # sum over accident years → (N_SIM, n_methods)
    return np.sqrt((total_errors ** 2).mean(axis=0))   # RMSE over simulations → (n_methods,)


# ── Reference table: CL(P) and CL(R) by eta (gamma-invariant) ────────────────

for metric_name, metric_fn in [("Mean error", get_mean), ("RMSE", get_rmse)]:
    print(f"\n# {metric_name} — CL(P) and CL(R) reference (does not vary with gamma)")
    print("\t" + "\t".join(f"eta={eta}" for eta in ETA_LIST))

    for method_idx in [0, 1]:  # 0 = CL(P), 1 = CL(R)
        values = [metric_fn(eta, any_gamma)[method_idx] for eta in ETA_LIST]
        print(labels[method_idx] + "\t" + "\t".join(f"{v:.0f}" for v in values))


# ── Main table: recovery methods by eta (rows) and gamma (columns) ────────────

for metric_name, metric_fn in [("Mean error", get_mean), ("RMSE", get_rmse)]:
    print(f"\n# {metric_name} — recovery methods")
    print("eta\tmethod\t" + "\t".join(f"gamma={g}" for g in GAMMA_LIST))

    for eta in ETA_LIST:
        for method_idx in [2, 3, 4]:  # 2 = CL(R̂,J), 3 = CL(R̂,J_P), 4 = CL(R̂,J_P_min)
            values = [metric_fn(eta, gamma)[method_idx] for gamma in GAMMA_LIST]
            print(f"{eta}\t{labels[method_idx]}\t" + "\t".join(f"{v:.0f}" for v in values))


# ── Plotting helpers ──────────────────────────────────────────────────────────

def get_total_errors(eta, gamma):
    """Returns array of shape (N_SIM, n_methods): total error per simulation."""
    data = np.stack(errors[(eta, gamma)])  # (N_SIM, n_methods, n_occ)
    return data.sum(axis=2)

def clip_1_99(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return arr[(arr >= lo) & (arr <= hi)], lo, hi

ETA_PLOT   = 1.2
GAMMA_PLOT = 100
COLORS     = ["steelblue", "darkorange", "green"]

# ── Plot 1: Distribution of total error — CL(P), CL(R), CL(R̂,J) ─────────────
# eta=1.2, gamma=100, one subplot per method

total_errors = get_total_errors(ETA_PLOT, GAMMA_PLOT)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(r"Distribution of total error  ($\eta=" + f"{ETA_PLOT}$, $\\gamma={GAMMA_PLOT}$)")
for ax, method_idx, color in zip(axes, [0, 1, 2], COLORS):
    clipped, *_ = clip_1_99(total_errors[:, method_idx])
    ax.hist(clipped, bins=50, color=color, density=True)
    ax.axvline(clipped.mean(), color="red", linestyle="--", linewidth=1.5)
    ax.text(0.97, 0.97, f"Mean = {clipped.mean():.0f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)
    ax.set_title(f"Distribution of {latex_labels[method_idx]} error")
    ax.set_xlabel("Total error")
axes[0].set_ylabel("Density")
plt.tight_layout()
#plt.savefig("figures/plot1_dist_clp_clr_clrj.png", dpi=150)
plt.show()

# ── Plot 2: RMSE bar chart — CL(P), CL(R), CL(R̂,J) ─────────────────────────

rmse_vals = get_rmse(ETA_PLOT, GAMMA_PLOT)[[0, 1, 2]]
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar([latex_labels[i] for i in [0, 1, 2]], rmse_vals, color=COLORS)
ax.set_title(r"RMSE  ($\eta=" + f"{ETA_PLOT}$, $\\gamma={GAMMA_PLOT}$)")
ax.set_ylabel("RMSE")
plt.tight_layout()
#plt.savefig("figures/plot2_rmse_clp_clr_clrj.png", dpi=150)
plt.show()

# ── Plots 3–5: Distribution by gamma + RMSE by gamma for each recovery method ─

def plot_by_gamma(method_idx, fname_prefix):
    """Subplots of clipped histogram per gamma, shared x-axis range."""
    n = len(GAMMA_LIST)

    # Collect clipped data and find global x-limits
    clipped_data = {}
    global_lo, global_hi = np.inf, -np.inf
    for gamma in GAMMA_LIST:
        arr, lo, hi = clip_1_99(get_total_errors(ETA_PLOT, gamma)[:, method_idx])
        clipped_data[gamma] = arr
        global_lo = min(global_lo, lo)
        global_hi = max(global_hi, hi)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Distribution of {latex_labels[method_idx]} error by $\\gamma$  ($\\eta={ETA_PLOT}$)")
    for ax, gamma in zip(axes, GAMMA_LIST):
        arr = clipped_data[gamma]
        ax.hist(arr, bins=50, range=(global_lo, global_hi), color="steelblue", density=True)
        ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.5)
        ax.text(0.97, 0.97, f"Mean = {arr.mean():.0f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
        ax.set_title(f"$\\gamma={gamma}$")
        ax.set_xlabel("Total error")
        ax.set_xlim(global_lo, global_hi)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    axes[0].set_ylabel("Density")
    plt.tight_layout(w_pad=2.5)
    #plt.savefig(f"figures/{fname_prefix}_dist_by_gamma.png", dpi=150)
    plt.show()

    # RMSE bar chart by gamma
    rmse_vals = [get_rmse(ETA_PLOT, gamma)[method_idx] for gamma in GAMMA_LIST]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([str(g) for g in GAMMA_LIST], rmse_vals, color="steelblue")
    ax.set_title(f"RMSE of {latex_labels[method_idx]} by $\\gamma$  ($\\eta={ETA_PLOT}$)")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("RMSE")
    plt.tight_layout()
    #plt.savefig(f"figures/{fname_prefix}_rmse_by_gamma.png", dpi=150)
    plt.show()

plot_by_gamma(2, "plot3_clrj")     # CL(R̂,J)
plot_by_gamma(3, "plot4_clrjp")    # CL(R̂,J_P)
plot_by_gamma(4, "plot5_clrjpmin") # CL(R̂,J_P_min)

# ── Plot 6: All three recovery methods together — eta=1.2, gamma=100 ──────────

recovery_indices = [2, 3, 4]

# 6a: Subplots per recovery method, shared x-axis range
clipped_data_6a = {}
global_lo_6a, global_hi_6a = np.inf, -np.inf
for method_idx in recovery_indices:
    arr, lo, hi = clip_1_99(total_errors[:, method_idx])
    clipped_data_6a[method_idx] = arr
    global_lo_6a = min(global_lo_6a, lo)
    global_hi_6a = max(global_hi_6a, hi)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(r"Distribution of total error — recovery methods  ($\eta=" + f"{ETA_PLOT}$, $\\gamma={GAMMA_PLOT}$)")
for ax, method_idx, color in zip(axes, recovery_indices, COLORS):
    arr = clipped_data_6a[method_idx]
    ax.hist(arr, bins=50, range=(global_lo_6a, global_hi_6a), color=color, density=True)
    ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.5)
    ax.text(0.97, 0.97, f"Mean = {arr.mean():.0f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)
    ax.set_title(f"Distribution of {latex_labels[method_idx]} error")
    ax.set_xlabel("Total error")
    ax.set_xlim(global_lo_6a, global_hi_6a)
axes[0].set_ylabel("Density")
plt.tight_layout()
#plt.savefig("figures/plot6a_recovery_dist.png", dpi=150)
plt.show()

# 6b: RMSE bar chart
rmse_vals = get_rmse(ETA_PLOT, GAMMA_PLOT)[recovery_indices]
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar([latex_labels[i] for i in recovery_indices], rmse_vals, color=COLORS)
ax.set_title(r"RMSE — recovery methods  ($\eta=" + f"{ETA_PLOT}$, $\\gamma={GAMMA_PLOT}$)")
ax.set_ylabel("RMSE")
plt.tight_layout()
#plt.savefig("figures/plot6b_recovery_rmse.png", dpi=150)
#plt.show()