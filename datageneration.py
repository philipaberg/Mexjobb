import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from core import (
    J, T, n_occ, occ_start, t_eval,
    simulate_R, build_claims, simulate_queue,
    recover_reportings,
    compute_J_P, compute_J_P_min, build_triangle, chain_ladder,
)

# parameters to test and store errors
ETA_LIST   = [1.10, 1.20, 1.40]
GAMMA_LIST = [0, 1.0, 10, 100]
N_SIM      = 500

def run_sim(_):
    try:
        R_full = simulate_R()
        claims, claims_by_cal, R_total = build_claims(R_full)

        # CL on R
        R = R_full[-n_occ:]
        true_ult = R[:, :J + 1].sum(axis=1)
        _, CL_R = chain_ladder(build_triangle(R, n_occ, J + 1, t_eval), n_occ, J + 1)
        err_R = CL_R[:, -1] - true_ult

        res = {}
        for eta in ETA_LIST:
            P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, eta)

            J_P = compute_J_P(P_full, n_occ)
            J_P_min = compute_J_P_min(P_full, n_occ, J_P)

            P = P_full[-n_occ:, :J_P + 1]
            _, CL_P = chain_ladder(build_triangle(P, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
            err_P = CL_P[:, -1] - true_ult

            for gamma in GAMMA_LIST:
                # True J
                R_hat_trueJ = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J, gamma)
                _, CL_trueJ = chain_ladder(build_triangle(R_hat_trueJ, n_occ, J + 1, t_eval), n_occ, J + 1)

                # J_P
                R_hat_JP = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J_P, gamma)
                _, CL_JP = chain_ladder(build_triangle(R_hat_JP, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)

                # J_P_min
                R_hat_JPmin = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J_P_min, gamma)
                _, CL_JPmin = chain_ladder(build_triangle(R_hat_JPmin, n_occ, J_P_min + 1, t_eval), n_occ, J_P_min + 1)

                res[(eta, gamma)] = np.stack([
                    err_P,
                    err_R,
                    CL_trueJ[:, -1] - true_ult,
                    CL_JP[:, -1] - true_ult,
                    CL_JPmin[:, -1] - true_ult,
                ])
        return res

    except ValueError:
        return None

# Simulation loop
if __name__ == "__main__":
    errors = {(eta, gamma): [] for eta in ETA_LIST for gamma in GAMMA_LIST}
    skipped = 0

    with Pool() as pool:
        results = list(tqdm(pool.imap(run_sim, range(N_SIM)), total=N_SIM))

    for res in results:
        if res is None:
            skipped += 1
            continue
        for key in errors:
            errors[key].append(res[key])

    print(f"Done — {skipped}/{N_SIM} simulations skipped due to solver failure")
    np.save("braetagammaresults.npy", {"errors": errors, "ETA_LIST": ETA_LIST, "GAMMA_LIST": GAMMA_LIST})
