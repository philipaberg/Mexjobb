"""
Usage: python plots.py etagammaresults.npy
REMEMBER TO ADD COMPARISON PLOT
AI-generated
"""

import sys
import numpy as np

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

# CL(P) and CL(R) do not depend on gamma, so we use any gamma to look them up
any_gamma = GAMMA_LIST[0]


# ── Helper: compute mean or RMSE for one (eta, gamma) pair ───────────────────

def get_mean(eta, gamma):
    data         = np.stack(errors[(eta, gamma)])  # shape: (N_SIM, n_methods, n_occ)
    total_errors = data.sum(axis=2)                # sum over accident years → (N_SIM, n_methods)
    return total_errors.mean(axis=0)               # average over simulations → (n_methods,)

def get_rmse(eta, gamma):
    data         = np.stack(errors[(eta, gamma)])
    total_errors = data.sum(axis=2)
    return np.sqrt((total_errors ** 2).mean(axis=0))


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

"""
Remember to add a comparison plot"""