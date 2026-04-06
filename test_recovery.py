import numpy as np
from core import (
    n_occ, J, simulate_R, build_claims, simulate_queue,
    recover_reportings, compute_J_P, compute_J_P_min,
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
    J_P_min = compute_J_P_min(P_full, n_occ, J_P)

    R_hat_J = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J, gamma=100)
    R_hat_JP = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J_P, gamma=0)
    R_hat_JPmin = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J_P_min, gamma=0)

    P = P_full[-n_occ:, :J_P + 1]

    print(f"\n=== Simulation {sim + 1}  (J={J}, J_P={J_P}, J_P_min={J_P_min}) ===\n")
    print("R:")
    print(R[:, :J + 1])
    print("\nR̂_J:")
    print(R_hat_J[:, :J + 1])
    print("\nR̂_J_P:")
    print(R_hat_JP[:, :J_P + 1])
    print("\nR̂_J_P_min:")
    print(R_hat_JPmin[:, :J_P_min + 1])
    print("\nP:")
    print(P)