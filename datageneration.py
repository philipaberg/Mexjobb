import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from core import (
    J, T, n_occ, occ_start, t_eval,
    simulate_R, build_claims, simulate_queue,
    recover_reportings_known_j, recover_reportings_est_j,
    compute_J_P, build_triangle, chain_ladder,
)

# parameters to test and store errors
ETA_LIST   = [1.05, 1.10, 1.20, 1.50]
GAMMA_LIST = [0, 1.0, 10, 100]
N_SIM      = 10000

def run_sim(_):
    R_full = simulate_R()
    claims, claims_by_cal, R_total = build_claims(R_full)

    # CL on R 
    R = R_full[-n_occ:]
    true_ult = R[:, :J + 1].sum(axis=1)
    _, CL_R = chain_ladder(build_triangle(R, n_occ, J + 1, t_eval), n_occ, J + 1)
    err_R = CL_R[:, -1] - true_ult

    res_known_j, res_est_j = {}, {}
    for eta in ETA_LIST:
        P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, eta)

        J_P = compute_J_P(P_full, n_occ)
        P = P_full[-n_occ:, :J_P + 1]
        max_j_CL = J_P + 1

        _, CL_P = chain_ladder(build_triangle(P, n_occ, max_j_CL, t_eval), n_occ, max_j_CL)
        err_P = CL_P[:, -1] - true_ult

        for gamma in GAMMA_LIST:
            # Known J recovery
            R_hat_kj = recover_reportings_known_j(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, gamma)
            _, CL_Rhat_kj = chain_ladder(build_triangle(R_hat_kj, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
            res_known_j[(eta, gamma)] = np.stack([err_P, err_R, CL_Rhat_kj[:, -1] - true_ult])

            # Estimated J recovery
            R_hat_ej = recover_reportings_est_j(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, gamma)
            _, CL_Rhat_ej = chain_ladder(build_triangle(R_hat_ej, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
            res_est_j[(eta, gamma)] = np.stack([err_P, err_R, CL_Rhat_ej[:, -1] - true_ult])
    return res_known_j, res_est_j

# Simulation loop
if __name__ == "__main__":
    errors_known_j = {(eta, gamma): [] for eta in ETA_LIST for gamma in GAMMA_LIST}
    errors_est_j   = {(eta, gamma): [] for eta in ETA_LIST for gamma in GAMMA_LIST}

    with Pool() as pool:
        results = list(tqdm(pool.imap(run_sim, range(N_SIM)), total=N_SIM))

    for res_known_j, res_est_j in results:
        for key in errors_known_j:
            errors_known_j[key].append(res_known_j[key])
            errors_est_j[key].append(res_est_j[key])

    np.save("gammaresults_known_j.npy", {"errors": errors_known_j, "ETA_LIST": ETA_LIST, "GAMMA_LIST": GAMMA_LIST})
    np.save("gammaresults_est_j.npy",   {"errors": errors_est_j,   "ETA_LIST": ETA_LIST, "GAMMA_LIST": GAMMA_LIST})
    print("Done")