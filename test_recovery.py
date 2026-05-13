import numpy as np
from core import (
    n_occ, J, simulate_R, build_claims, simulate_queue,
    recover_reportings, recover_reportings_known_labels,
    known_reportings, compute_J_P,
)

N_SIM = 5
ETA   = 1.1
T     = 200

for sim in range(N_SIM):
    occ_start = T - n_occ

    R_full = simulate_R(T)
    claims, claims_by_cal, R_total = build_claims(R_full, T)
    P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, ETA, T)
    R_known = known_reportings(claims, P_total, T)

    R = R_full[-n_occ:]
    J_P = compute_J_P(P_full, n_occ)

    R_hat_g100   = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J, gamma=100)
    R_hat_g0     = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J, gamma=0)
    R_hat_known  = recover_reportings_known_labels(P_full, B_total, R_total, P_total, R_known, occ_start, n_occ, J_P, J, gamma=100)

    P = P_full[-n_occ:, :J_P + 1]

    print(f"\nSimulation {sim + 1}/{N_SIM}")
    print("True R:")
    print(R)
    print("\nRecovered R (known labels):")
    print(R_hat_known)
    print("known R:")
    print(R_known[-n_occ:])

    print("\nP:")
    print(P)