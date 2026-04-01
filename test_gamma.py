import numpy as np
from core import (
    n_occ, J, simulate_R, build_claims, simulate_queue,
    recover_reportings_known_j, recover_reportings_est_j, compute_J_P,
)

N_SIM = 5
ETA   = 1.2
T     = 120

for sim in range(N_SIM):
    occ_start = T - n_occ

    R_full = simulate_R(T)
    claims, claims_by_cal, R_total = build_claims(R_full, T)
    P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, ETA, T)

    R = R_full[-n_occ:]
    J_P = compute_J_P(P_full, n_occ)

    R_hat_10 = recover_reportings_known_j(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, gamma=10)
    R_hat_100 = recover_reportings_known_j(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, gamma=100)
    R_hat_ej = recover_reportings_est_j(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, gamma=0)

    P = P_full[-n_occ:, :J_P + 1]

    print(f"Simulation {sim + 1}")
    print(R[:, :J + 1])
    print(R_hat_10[:, :J + 1])
    print(R_hat_100[:, :J + 1])
    print(R_hat_ej[:, :J + 1])
    print(P)