import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from core import (
    J, T, n_occ, occ_start, t_eval,
    simulate_R, build_claims, simulate_queue,
    recover_reportings, compute_ab, benktander,
    compute_J_P, build_triangle, chain_ladder,
    known_reportings, recover_reportings_known_labels,
)

ETA = 1.1
GAMMA = 100
NSIM = 10000
KAPPAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 100.0]



def run_sim(_):
    R_full = simulate_R()
    claims, claims_by_cal, R_total = build_claims(R_full)
    P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, ETA)
    R_known = known_reportings(claims, P_total, T)

    R = R_full[-n_occ:]
    true_ult = R[:, :J + 1].sum(axis=1)

    a, b = compute_ab(B_total, P_total, R_total, occ_start, n_occ)
    J_P = compute_J_P(P_full, n_occ)

    results = {
        "a": a,
        "b": b,
        "err_CL": np.nan,
        **{f"err_BK_kappa_{k}": np.nan for k in KAPPAS},
        "err_BK_mean_a": np.nan,
    }

    try:
        # Recovery
        R_hat = recover_reportings_known_labels(P_full, B_total, R_total, P_total, R_known, occ_start, n_occ, max(J_P, J), J, GAMMA)
        triangle = build_triangle(R_hat, n_occ, max(J_P, J) + 1, t_eval)

        # CL on recovered
        factors, CL_R_hat = chain_ladder(triangle, n_occ, max(J_P, J) + 1)
        err_CL = CL_R_hat[:, -1] - true_ult
        results["err_CL"] = err_CL

        # weighted BK on recovered
        prior = R_total[occ_start:occ_start + n_occ].mean() * np.ones(n_occ)   # 1/17 sum R_t, t
        for kappa in KAPPAS:
            BK_R_hat = benktander(triangle, factors, prior, a, n_occ, max(J_P, J) + 1, kappa)
            err_BK = BK_R_hat[:, -1] - true_ult
            results[f"err_BK_kappa_{kappa}"] = err_BK

        # Regular BK on recovered
        BK_R_hat = benktander(triangle, factors, prior, a, n_occ, max(J_P, J) + 1, np.mean(a))
        results["err_BK_mean_a"] = BK_R_hat[:, -1] - true_ult

    except ValueError:
        pass

    return results

if __name__ == "__main__":

    with Pool() as pool:
        results = list(tqdm(pool.imap(run_sim, range(NSIM)), total=NSIM))

    skipped = sum(1 for r in results if np.isnan(r["err_CL"]).all())
    print(f"Skipped {skipped} simulations due to errors.")

    np.save("labelresults.npy", results)