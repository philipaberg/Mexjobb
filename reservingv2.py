import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from core import (
    J, T, n_occ, occ_start, t_eval,
    simulate_R, build_claims, simulate_queue,
    recover_reportings, compute_ab, benktander,
    compute_J_P, build_triangle, chain_ladder,
)

ETA = 1.1
GAMMA = 100
NSIM = 10000
KAPPAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 100.0]

def run_sim(_):

    n_occ_recover = 24
    occ_start_recover = T - n_occ_recover

    R_full = simulate_R()
    claims, claims_by_cal, R_total = build_claims(R_full)
    P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, ETA)

    R = R_full[-n_occ:]
    true_ult = R[:, :J + 1].sum(axis=1)
    _, CL_R = chain_ladder(build_triangle(R, n_occ, J + 1, t_eval), n_occ, J + 1)
    err_R = CL_R[:, -1] - true_ult

    a, b = compute_ab(B_total, P_total, R_total, occ_start, n_occ)
    J_P = compute_J_P(P_full, n_occ)

    # CL on P
    P = P_full[-n_occ:, :J_P + 1]
    pfactor, CL_P = chain_ladder(build_triangle(P, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
    err_P = CL_P[:, -1] - true_ult

    # BK on P
    prior = 0.9 * R_total[occ_start:occ_start + n_occ].mean() * np.ones(n_occ)   # 1/17 sum R_t, t
    BK_P = benktander(build_triangle(P, n_occ, J_P + 1, t_eval), pfactor, prior, a, n_occ, J_P + 1, np.mean(a))
    err_BK_P = BK_P[:, -1] - true_ult

    # Asymptotic BF on P
    BF_P = benktander(build_triangle(P, n_occ, J_P + 1, t_eval), pfactor, prior, a, n_occ, J_P + 1, np.inf)
    err_BF_P = BF_P[:, -1] - true_ult

    results = {
        "a": a,
        "b": b,
        "err_P": err_P,
        "err_BK_P": err_BK_P,
        "err_BF_P": err_BF_P,
        "err_R": err_R,
        "err_R_17": np.nan,
        "err_R_24": np.nan,
        "err_CL": np.nan,
        **{f"err_BK_kappa_{k}": np.nan for k in KAPPAS},
        "err_BK_mean_a": np.nan,
    }

    try:
        #recovery 1st 24 periods
        R_rec_24 = recover_reportings(P_full, B_total, R_total, P_total, occ_start_recover, n_occ_recover, J_P, J, GAMMA)
        R_hat_24 = R_rec_24[-n_occ:]
        triangle_24 = build_triangle(R_hat_24, n_occ, J_P + 1, t_eval)
        results["err_R_24"] = R_hat_24[:, :J + 1].sum(axis=1) - R[:, :J + 1].sum(axis=1)

        # Recovery using 17 periods
        R_hat = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J, GAMMA)
        results["err_R_17"] = R_hat[:, :J + 1].sum(axis=1) - R[:, :J + 1].sum(axis=1)

        # CL on recovered with 24 periods
        factors_24, CL_R_hat_24 = chain_ladder(triangle_24, n_occ, J_P + 1)
        err_CL_24 = CL_R_hat_24[:, -1] - true_ult
        results["err_CL"] = err_CL_24

        # weighted BK on recovered with 24 periods
        prior = 0.9 * R_total[occ_start:occ_start + n_occ].mean() * np.ones(n_occ)   # 1/17 sum R_t, t
        for kappa in KAPPAS:
            BK_R_hat_24 = benktander(triangle_24, factors_24, prior, a, n_occ, J_P + 1, kappa)
            err_BK_24 = BK_R_hat_24[:, -1] - true_ult
            results[f"err_BK_kappa_{kappa}"] = err_BK_24

        # Regular BK
        BK_R_hat = benktander(triangle_24, factors_24, prior, a, n_occ, J_P + 1, np.mean(a))
        results["err_BK_mean_a"] = BK_R_hat[:, -1] - true_ult

    except ValueError:
        pass

    return results

if __name__ == "__main__":

    with Pool() as pool:
        results = list(tqdm(pool.imap(run_sim, range(NSIM)), total=NSIM))

    skipped = sum(1 for r in results if np.isnan(r["err_CL"]).all())
    print(f"Skipped {skipped} simulations due to errors.")

    np.save("24reservresults.npy", results)