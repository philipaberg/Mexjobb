import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from core import (
    J, T, n_occ, occ_start, t_eval,
    simulate_R, build_claims, simulate_queue,
    compute_J_P, build_triangle, chain_ladder,
    known_reportings, recover_reportings_known_labels,
)

ETA_LIST = [1.1, 1.2, 1.4]
GAMMA_LIST = [0, 1.0, 10, 100]
NSIM = 6000


def run_sim(_):
    R_full = simulate_R()
    claims, claims_by_cal, R_total = build_claims(R_full)
    R = R_full[-n_occ:]
    true_ult = R[:, :J + 1].sum(axis=1)

    res = {(ETA, GAMMA): np.nan for ETA in ETA_LIST for GAMMA in GAMMA_LIST}
    for ETA in ETA_LIST:
        P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, ETA)
        R_known = known_reportings(claims, P_total, T)
        J_P = compute_J_P(P_full, n_occ)

        for GAMMA in GAMMA_LIST:
            try:
                R_hat = recover_reportings_known_labels(P_full, B_total, R_total, P_total, R_known, occ_start, n_occ, max(J_P, J), J, GAMMA)
                triangle = build_triangle(R_hat, n_occ, max(J_P, J) + 1, t_eval)
                _, CL_R_hat = chain_ladder(triangle, n_occ, max(J_P, J) + 1)
                res[(ETA, GAMMA)] = np.sum(CL_R_hat[:, -1] - true_ult)
            except ValueError:
                pass
    return res

if __name__ == "__main__":
    with Pool() as pool:
        results = list(tqdm(pool.imap(run_sim, range(NSIM)), total=NSIM))

    skipped = sum(1 for r in results for v in r.values() if np.isnan(v))
    print(f"Skipped {skipped} recovery(s) due to errors.")

    np.save("CL_labelresults.npy", results)