import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from core import (
    J, n_occ, simulate_R, build_claims,
    simulate_queue, recover_reportings,
    compute_J_P, compute_J_P_min,
    compute_ab, build_triangle, chain_ladder,
)

# Parametrar
T_SIM = 250   # burn-in stationarity for B_t, when eta = 1.10
ETA = 1.1
GAMMA_J = 100
GAMMA_J_P = 0
GAMMA_J_P_MIN = 0
N_SIM = 40000


def run_sim(_):
    occ_start = T_SIM - n_occ
    t_eval = n_occ - 1

    R_full = simulate_R(T_SIM)
    claims, claims_by_cal, R_total = build_claims(R_full, T_SIM)
    P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, ETA, T_SIM)

    R = R_full[-n_occ:]
    true_ult = R[:, :J + 1].sum(axis=1)
    _, CL_R = chain_ladder(build_triangle(R, n_occ, J + 1, t_eval), n_occ, J + 1)
    err_R = CL_R[:, -1] - true_ult

    J_P = compute_J_P(P_full, n_occ)
    J_P_min = compute_J_P_min(P_full, n_occ, J_P)
    a, b = compute_ab(B_total, P_total, R_total, occ_start, n_occ)

    result = {
        "J_hat":         J_P,
        "J_P_min":       J_P_min,
        "a":             a,
        "b":             b,
        "B_t":           B_total[-n_occ:],
        "sum_err_P":     np.nan,
        "sum_err_J":     np.nan,
        "sum_err_JP":    np.nan,
        "sum_err_JPmin": np.nan,
        "rel_err_P":     np.nan,
        "rel_err_J":     np.nan,
        "rel_err_JP":    np.nan,
        "rel_err_JPmin": np.nan,
    }

    try:
        # CL(P)
        P = P_full[-n_occ:, :J_P + 1]
        _, CL_P = chain_ladder(build_triangle(P, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
        result["sum_err_P"] = (CL_P[:, -1] - true_ult).sum()
        result["rel_err_P"] = np.abs(result["sum_err_P"] - err_R.sum())

        # CL(Rhat, J)
        R_hat_J = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J, GAMMA_J)
        _, CL_J = chain_ladder(build_triangle(R_hat_J, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
        result["sum_err_J"] = (CL_J[:, -1] - true_ult).sum()
        result["rel_err_J"] = np.abs(result["sum_err_J"] - err_R.sum())

        # CL(Rhat, J_P)
        R_hat_JP = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J_P, GAMMA_J_P)
        _, CL_JP = chain_ladder(build_triangle(R_hat_JP, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
        result["sum_err_JP"] = (CL_JP[:, -1] - true_ult).sum()
        result["rel_err_JP"] = np.abs(result["sum_err_JP"] - err_R.sum())

        # CL(Rhat, J_P_min)
        R_hat_JPmin = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J_P_min, GAMMA_J_P_MIN)
        _, CL_JPmin = chain_ladder(build_triangle(R_hat_JPmin, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
        result["sum_err_JPmin"] = (CL_JPmin[:, -1] - true_ult).sum()
        result["rel_err_JPmin"] = np.abs(result["sum_err_JPmin"] - err_R.sum())

    except ValueError:
        pass

    return result


if __name__ == "__main__":

    with Pool() as pool:
        results = list(tqdm(pool.imap(run_sim, range(N_SIM)), total=N_SIM))

    skipped = sum(1 for r in results if np.isnan(r["sum_err_J"]))
    print(f"{skipped}/{N_SIM} simulations had solver failure (error values set to NaN)")

    np.save("diagresults.npy", {
        "J_hats":         np.array([r["J_hat"]        for r in results]),
        "J_P_mins":       np.array([r["J_P_min"]      for r in results]),
        "as":             np.array([r["a"]             for r in results]),
        "bs":             np.array([r["b"]             for r in results]),
        "B_ts":           np.array([r["B_t"]           for r in results]),
        "sum_err_P":      np.array([r["sum_err_P"]     for r in results]),
        "sum_err_J":      np.array([r["sum_err_J"]     for r in results]),
        "sum_err_JP":     np.array([r["sum_err_JP"]    for r in results]),
        "sum_err_JPmin":  np.array([r["sum_err_JPmin"] for r in results]),
        "rel_err_P":      np.array([r["rel_err_P"]     for r in results]),
        "rel_err_J":      np.array([r["rel_err_J"]     for r in results]),
        "rel_err_JP":     np.array([r["rel_err_JP"]    for r in results]),
        "rel_err_JPmin":  np.array([r["rel_err_JPmin"] for r in results]),
        "ETA":           ETA,
        "GAMMA_J":       GAMMA_J,
        "GAMMA_J_P":     GAMMA_J_P,
        "GAMMA_J_P_MIN": GAMMA_J_P_MIN,
        "N_SIM":         N_SIM,
        "T_SIM":         T_SIM,
    })

    print("Resultat sparat")
