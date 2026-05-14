import os
import numpy as np
import matplotlib.pyplot as plt

COLORS = ["salmon", "steelblue", "mediumseagreen"]
os.makedirs("finalfigures", exist_ok=True)

# ── Load datasets ─────────────────────────────────────────────────────────────

eg      = np.load("etagammaresults.npy", allow_pickle=True).item()
eg_errors     = eg["errors"]
ETA_LIST      = eg["ETA_LIST"]
GAMMA_LIST    = eg["GAMMA_LIST"]

bk_results = np.load("benktanderresults.npy", allow_pickle=True)
bk_valid   = [r for r in bk_results if not np.isnan(r["err_CL"]).all()]

lb_results = np.load("labelresults.npy", allow_pickle=True)
lb_valid   = [r for r in lb_results if not np.isnan(r["err_CL"]).all()]

print(f"benktander valid: {len(bk_valid)} / {len(bk_results)}")
print(f"label    valid:   {len(lb_valid)} / {len(lb_results)}")

KAPPAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 100.0]

# benktander arrays
bk_err_P     = np.array([r["err_P"]        for r in bk_valid])
bk_err_BK_P  = np.array([r["err_BK_P"]     for r in bk_valid])
bk_err_BF_P  = np.array([r["err_BF_P"]     for r in bk_valid])
bk_err_R     = np.array([r["err_R"]        for r in bk_valid])
bk_err_R_17  = np.array([r["err_R_17"]     for r in bk_valid])
bk_err_R_24  = np.array([r["err_R_24"]     for r in bk_valid])
bk_err_CL    = np.array([r["err_CL"]       for r in bk_valid])
bk_n_occ     = len(bk_valid[0]["err_R"])
bk_err_BK    = {k: np.array([r[f"err_BK_kappa_{k}"] for r in bk_valid]) for k in KAPPAS}
bk_err_BK_ma = np.array([r["err_BK_mean_a"] for r in bk_valid])

# label arrays
lb_err_CL    = np.array([r["err_CL"]  for r in lb_valid])
lb_err_BK    = {k: np.array([r[f"err_BK_kappa_{k}"] for r in lb_valid]) for k in KAPPAS}
lb_err_BK_ma = np.array([r["err_BK_mean_a"] for r in lb_valid])

# ── Shared helpers ────────────────────────────────────────────────────────────

def plot_hist(ax, data, label, color, bins):
    mean = data.mean()
    rmse = np.sqrt((data ** 2).mean())
    ax.hist(data, bins=bins, color=color, edgecolor="black", alpha=0.8, density=True)
    ax.axvline(mean, color="red", linestyle="--", linewidth=1.5)
    ax.text(
        0.97, 0.95,
        f"RMSE: {rmse:.0f}\nMean: {mean:.0f}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
    )
    ax.set_title(label)
    ax.set_xlabel("Total error")

def clip_1_99(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return arr[(arr >= lo) & (arr <= hi)], lo, hi

def get_eg_total_errors(eta, gamma):
    data = np.stack(eg_errors[(eta, gamma)])
    return data.sum(axis=2)

def get_eg_mean(eta, gamma):
    data         = np.stack(eg_errors[(eta, gamma)])
    total_errors = data.sum(axis=2)
    return total_errors.mean(axis=0)

def get_eg_rmse(eta, gamma):
    data         = np.stack(eg_errors[(eta, gamma)])
    total_errors = data.sum(axis=2)
    return np.sqrt((total_errors ** 2).mean(axis=0))


# ═══════════════════════════════════════════════════════════════════════════════
# plots.py plots  (etagammaresults.npy,  eta=1.2, gamma=100)
# ═══════════════════════════════════════════════════════════════════════════════

labels = ["CL(P)", "CL(R)", "CL(R̂, J)", "CL(R̂, J_P)", "CL(R̂, J_P_min)"]
latex_labels = [
    r"$\mathrm{CL}(P)$",
    r"$\mathrm{CL}(R)$",
    r"$\mathrm{CL}(\hat{R}, J)$",
    r"$\mathrm{CL}(\hat{R}, J_P)$",
    r"$\mathrm{CL}(\hat{R}, J_{P,\min})$",
]
any_gamma = GAMMA_LIST[0]

# ── Table: CL(P) and CL(R) reference (gamma-invariant) ───────────────────────
for metric_name, metric_fn in [("Mean error", get_eg_mean), ("RMSE", get_eg_rmse)]:
    print(f"\n# {metric_name} — CL(P) and CL(R) reference (does not vary with gamma)")
    print("\t" + "\t".join(f"eta={eta}" for eta in ETA_LIST))
    for method_idx in [0, 1]:
        values = [metric_fn(eta, any_gamma)[method_idx] for eta in ETA_LIST]
        print(labels[method_idx] + "\t" + "\t".join(f"{v:.0f}" for v in values))

# ── Table: recovery methods by eta and gamma ──────────────────────────────────
for metric_name, metric_fn in [("Mean error", get_eg_mean), ("RMSE", get_eg_rmse)]:
    print(f"\n# {metric_name} — recovery methods")
    print("eta\tmethod\t" + "\t".join(f"gamma={g}" for g in GAMMA_LIST))
    for eta in ETA_LIST:
        for method_idx in [2, 3, 4]:
            values = [metric_fn(eta, gamma)[method_idx] for gamma in GAMMA_LIST]
            print(f"{eta}\t{labels[method_idx]}\t" + "\t".join(f"{v:.0f}" for v in values))

ETA_PLOT   = 1.2
GAMMA_PLOT = 100

total_errors = get_eg_total_errors(ETA_PLOT, GAMMA_PLOT)

# ── EG Plot 1: Histograms — CL(P), CL(R), CL(R̂,J) ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(r"Distribution of total error  ($\eta=" + f"{ETA_PLOT}$, $\\gamma={GAMMA_PLOT}$)")
for ax, method_idx, color in zip(axes, [0, 1, 2], COLORS):
    full = total_errors[:, method_idx]
    clipped, *_ = clip_1_99(full)
    ax.hist(clipped, bins=50, color=color, edgecolor="black", alpha=0.8, density=True)
    ax.axvline(full.mean(), color="red", linestyle="--", linewidth=1.5)
    ax.text(0.97, 0.97, f"Mean = {full.mean():.0f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))
    ax.set_title(f"Distribution of {latex_labels[method_idx]} error")
    ax.set_xlabel("Total error")
axes[0].set_ylabel("Density")
plt.tight_layout()
# plt.savefig("finalfigures/eg_hist_clp_clr_clrj.png", dpi=150)
# plt.show()

# ── EG Plot 2: RMSE bar chart — CL(P), CL(R), CL(R̂,J) ──────────────────────
rmse_vals = get_eg_rmse(ETA_PLOT, GAMMA_PLOT)[[0, 1, 2]]
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar([latex_labels[i] for i in [0, 1, 2]], rmse_vals, color=COLORS)
ax.set_title(r"RMSE  ($\eta=" + f"{ETA_PLOT}$, $\\gamma={GAMMA_PLOT}$)")
ax.set_ylabel("RMSE")
plt.tight_layout()
# plt.savefig("finalfigures/eg_rmse_clp_clr_clrj.png", dpi=150)
# plt.show()

# ── EG Plot 6a: Histograms — recovery methods (shared axes) ──────────────────
recovery_indices = [2, 3, 4]
clipped_data = {}
global_lo, global_hi = np.inf, -np.inf
for idx in recovery_indices:
    arr, lo, hi = clip_1_99(total_errors[:, idx])
    clipped_data[idx] = arr
    global_lo = min(global_lo, lo)
    global_hi = max(global_hi, hi)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle(r"Distribution of total error — recovery methods  ($\eta=" + f"{ETA_PLOT}$, $\\gamma={GAMMA_PLOT}$)")
for ax, idx, color in zip(axes, recovery_indices, COLORS):
    full = total_errors[:, idx]
    arr = clipped_data[idx]
    ax.hist(arr, bins=50, range=(global_lo, global_hi), color=color, edgecolor="black", alpha=0.8, density=True)
    ax.axvline(full.mean(), color="red", linestyle="--", linewidth=1.5)
    ax.text(0.97, 0.97, f"Mean = {full.mean():.0f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))
    ax.set_title(f"Distribution of {latex_labels[idx]} error")
    ax.set_xlabel("Total error")
    ax.set_xlim(global_lo, global_hi)
axes[0].set_ylabel("Density")
plt.tight_layout()
# plt.savefig("finalfigures/eg_hist_recovery.png", dpi=150)
# plt.show()

# ── EG Plot 6b: RMSE bar chart — recovery methods ────────────────────────────
rmse_vals = get_eg_rmse(ETA_PLOT, GAMMA_PLOT)[recovery_indices]
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar([latex_labels[i] for i in recovery_indices], rmse_vals, color=COLORS)
ax.set_title(r"RMSE — recovery methods  ($\eta=" + f"{ETA_PLOT}$, $\\gamma={GAMMA_PLOT}$)")
ax.set_ylabel("RMSE")
plt.tight_layout()
# plt.savefig("finalfigures/eg_rmse_recovery.png", dpi=150)
# plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# plot_results.py plots  (benktanderresults.npy)
# ═══════════════════════════════════════════════════════════════════════════════

bk_total_R   = bk_err_R.sum(axis=1)
bk_total_CL  = bk_err_CL.sum(axis=1)
bk_total_BK1 = bk_err_BK[1.0].sum(axis=1)

# ── Table: RMSE and mean total reserve error ──────────────────────────────────
bk_methods = {
    "CL P": bk_err_P, "BK P": bk_err_BK_P, "BF P": bk_err_BF_P,
    "CL (true R)": bk_err_R, "CL (recovered)": bk_err_CL,
}
for k in KAPPAS:
    bk_methods[f"Weighted BK κ={k}"] = bk_err_BK[k]
bk_methods["BK κ=mean(a)"] = bk_err_BK_ma

col_w = 20
print(f"\n{'Method':<{col_w}} {'RMSE':>12} {'Mean error':>12}")
print("-" * (col_w + 26))
for name, arr in bk_methods.items():
    total = arr.sum(axis=1)
    print(f"{name:<{col_w}} {np.sqrt((total**2).mean()):>12.0f} {total.mean():>12.0f}")

# ── BK Plot 1: Bar chart – mean error per occurrence period ──────────────────
periods = np.arange(1, bk_n_occ + 1)
w, n = 0.35, 3
periods_plot  = periods[:-n]
err_R_17_plot = bk_err_R_17.mean(axis=0)[:-n]
err_R_24_plot = bk_err_R_24.mean(axis=0)[:-n]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(periods_plot - w / 2, err_R_17_plot, w, label="Recovery error (17-period)", color="steelblue")
ax.bar(periods_plot + w / 2, err_R_24_plot, w, label="Recovery error (24-period)", color="tomato")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Occurrence period")
ax.set_ylabel("Mean error")
ax.set_title("Mean recovery error per occurrence period: 17 vs 24 periods")
ax.set_xticks(periods_plot)
ax.legend()
plt.tight_layout()
# plt.savefig("finalfigures/bk_bar_occurrence.png", dpi=150)
# plt.show()

# ── BK Plot 2: Histograms — CL (true R), CL (recovered), BK κ=1 ──────────────
all_vals = np.concatenate([bk_total_R, bk_total_CL, bk_total_BK1])
xmin, xmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
bins = np.linspace(xmin, xmax, 61)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
plot_hist(axes[0], bk_total_R,   "Total error – CL (true R)",                          "salmon",         bins)
axes[0].set_ylabel("Density")
plot_hist(axes[1], bk_total_CL,  "Total error – CL (recovered R)",                     "steelblue",      bins)
plot_hist(axes[2], bk_total_BK1, "Total error – Benktander κ=1.0 (recovered R)",        "mediumseagreen", bins)
for ax in axes:
    ax.set_xlim(xmin, xmax)
plt.tight_layout()
# plt.savefig("finalfigures/bk_hist_cl_bk.png", dpi=150)
# plt.show()

# ── BK Plot 4: Distribution of J_P and J_P_min ───────────────────────────────
bk_jp     = np.array([r["J_P"]     for r in bk_valid])
bk_jp_min = np.array([r["J_P_min"] for r in bk_valid])
jp_bins = np.arange(0, max(bk_jp.max(), bk_jp_min.max()) + 2) - 0.5

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle(r"Distribution of estimates $J_P$ and $J_{P,\min}$ ($\eta=1.1$)")
for ax, data, label, color in zip(
    axes,
    [bk_jp, bk_jp_min],
    [r"$J_P$", r"$J_{P,\min}$"],
    ["steelblue", "mediumseagreen"],
):
    ax.hist(data, bins=jp_bins, color=color, edgecolor="black", alpha=0.8, density=True)
    ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.5)
    ax.text(0.97, 0.97, f"Mean = {data.mean():.1f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))
    ax.set_title(f"Distribution of {label}")
    ax.set_xlabel("Estimated value")
    ax.set_xticks(np.arange(0, jp_bins[-1] + 1))
axes[0].set_ylabel("Density")
plt.tight_layout()
#plt.savefig("finalfigures/bk_jp_distribution.png", dpi=150)
#plt.show()

# ── BK Plot 3: RMSE and Mean error vs kappa ───────────────────────────────────
kappas_plot = [k for k in KAPPAS if k <= 10.0]
bk_rmse = [np.sqrt((bk_err_BK[k].sum(axis=1) ** 2).mean()) for k in kappas_plot]
bk_mean = [bk_err_BK[k].sum(axis=1).mean()                 for k in kappas_plot]

bk_ref = {
    "CL (κ=0)":              bk_err_BK[0],
    "BK κ=mean(a)":          bk_err_BK_ma,
    "BF asymptotic (κ=100)": bk_err_BK[100.0],
}
ref_colors = {"CL (κ=0)": "tomato", "BK κ=mean(a)": "mediumseagreen", "BF asymptotic (κ=100)": "darkorange"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(kappas_plot, bk_mean, color="steelblue", linewidth=2, marker="o", markersize=4, label="BK κ")
ax2.plot(kappas_plot, bk_rmse, color="steelblue", linewidth=2, marker="o", markersize=4, label="BK κ")
for label, arr in bk_ref.items():
    total = arr.sum(axis=1)
    c = ref_colors[label]
    ax1.axhline(total.mean(),                  color=c, linestyle=":", linewidth=1.5, label=label)
    ax2.axhline(np.sqrt((total**2).mean()),    color=c, linestyle=":", linewidth=1.5, label=label)
ax1.set_xlabel("κ"); ax1.set_ylabel("Mean total error"); ax1.set_title("Mean error vs κ"); ax1.legend()
ax2.set_xlabel("κ"); ax2.set_ylabel("RMSE");              ax2.set_title("RMSE vs κ");       ax2.legend()
plt.tight_layout()
# plt.savefig("finalfigures/bk_kappa.png", dpi=150)
# plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# plot_label.py plots  (labelresults.npy)
# ═══════════════════════════════════════════════════════════════════════════════

lb_total_CL  = lb_err_CL.sum(axis=1)
lb_total_BK1 = lb_err_BK[1.0].sum(axis=1)

# ── LB Plot 1: Histograms — CL (recovered, known labels), BK κ=1 ─────────────
all_vals = np.concatenate([lb_total_CL, lb_total_BK1])
xmin, xmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
bins = np.linspace(xmin, xmax, 61)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
plot_hist(axes[0], lb_total_CL,  "Total error – CL (recovered R, known labels)",          "steelblue",      bins)
axes[0].set_ylabel("Density")
plot_hist(axes[1], lb_total_BK1, "Total error – Benktander κ=1.0 (recovered R, known labels)", "mediumseagreen", bins)
for ax in axes:
    ax.set_xlim(xmin, xmax)
plt.tight_layout()
# plt.savefig("finalfigures/lb_hist_cl_bk.png", dpi=150)
# plt.show()

# ── LB Plot 2: RMSE and Mean error vs kappa (known labels) ───────────────────
lb_rmse = [np.sqrt((lb_err_BK[k].sum(axis=1) ** 2).mean()) for k in kappas_plot]
lb_mean = [lb_err_BK[k].sum(axis=1).mean()                 for k in kappas_plot]

lb_ref = {
    "CL (κ=0)":              lb_err_BK[0],
    "BK κ=mean(a)":          lb_err_BK_ma,
    "BF asymptotic (κ=100)": lb_err_BK[100.0],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(kappas_plot, lb_mean, color="steelblue", linewidth=2, marker="o", markersize=4, label="BK κ")
ax2.plot(kappas_plot, lb_rmse, color="steelblue", linewidth=2, marker="o", markersize=4, label="BK κ")
for label, arr in lb_ref.items():
    total = arr.sum(axis=1)
    c = ref_colors[label]
    ax1.axhline(total.mean(),               color=c, linestyle=":", linewidth=1.5, label=label)
    ax2.axhline(np.sqrt((total**2).mean()), color=c, linestyle=":", linewidth=1.5, label=label)
ax1.set_xlabel("κ"); ax1.set_ylabel("Mean total error"); ax1.set_title("Mean error vs κ (known labels)"); ax1.legend()
ax2.set_xlabel("κ"); ax2.set_ylabel("RMSE");              ax2.set_title("RMSE vs κ (known labels)");       ax2.legend()
plt.tight_layout()
# plt.savefig("finalfigures/lb_kappa.png", dpi=150)
# plt.show()
