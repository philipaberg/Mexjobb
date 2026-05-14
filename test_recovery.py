import numpy as np
from sympy import re
from core import (
    n_occ, J, simulate_R, build_claims, simulate_queue,
    recover_reportings, recover_reportings_known_labels,
    known_reportings, compute_J_P, compute_J_P_min
)

N_SIM = 1
ETA   = 1.2
T     = 250

for sim in range(N_SIM):
    n_occ_recover = 24
    occ_start_recover = T - n_occ_recover
    occ_start = T - n_occ

    R_full = simulate_R(T)
    claims, claims_by_cal, R_total = build_claims(R_full, T)
    P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, ETA, T)
    R_known = known_reportings(claims, P_total, T)

    R = R_full[-n_occ:]
    J_P = compute_J_P(P_full, n_occ)
    J_P_24 = compute_J_P(P_full, n_occ_recover)
    J_P_min = compute_J_P_min(P_full, n_occ, J_P)

    R_hat_J   = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, max(J_P, J), J, gamma=100)
    R_hat_JP     = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J_P, gamma=0)
    R_hat_JPmin     = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J_P_min, gamma=0)
    R_hat_24 = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, max(J_P_24, J), J, gamma=100)
    R_hat_more = R_hat_24[-n_occ:]
    R_hat_known  = recover_reportings_known_labels(P_full, B_total, R_total, P_total, R_known, occ_start, n_occ, J_P, J, gamma=100)


    P = P_full[-n_occ:, :J_P + 1]

    print(f"\nSimulation {sim + 1}/{N_SIM}")
    print("True R:")
    print(R)
    print("\nRecovered R (J):")
    print(R_hat_J)
    print("\nRecovered R (J_P):")
    print(R_hat_JP)
    print("\nRecovered R (J_P_min):")
    print(R_hat_JPmin)
    print("\nRecovered R (more periods):")
    print(R_hat_more)
    print("\nRecovered R (known labels):")
    print(R_hat_known)
    print("known R:")
    print(R_known[-n_occ:])
    print("\nP:")
    print(P)