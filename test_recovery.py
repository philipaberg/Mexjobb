import numpy as np
from core import (
    n_occ, J, simulate_R, build_claims, simulate_queue,
    recover_reportings, compute_J_P,
)

N_SIM = 5
ETA   = 1.1
T     = 200

for sim in range(N_SIM):
    occ_start = T - n_occ

    R_full = simulate_R(T)
    claims, claims_by_cal, R_total = build_claims(R_full, T)
    P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, ETA, T)

    R = R_full[-n_occ:]
    J_P = compute_J_P(P_full, n_occ)

    R_hat_g100 = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J, gamma=100)
    R_hat_g0   = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J, gamma=0)

    P = P_full[-n_occ:, :J_P + 1]

    for label, R_hat in [("gamma=100", R_hat_g100), ("gamma=0  ", R_hat_g0)]:
        print(f"\n=== Simulation {sim + 1}  (J={J}, J_P={J_P}, {label}) ===\n")

        # Side-by-side: R and R_hat, all columns
        header = f"{'i':>3}  " + "  ".join(f"R̂j{j:<2} Rj{j:<2}" for j in range(J + 1))
        print(header)
        print("-" * len(header))
        for i in range(n_occ):
            row = f"{i:>3}  "
            row += "  ".join(f"{R_hat[i, j]:>4}  {R[i, j]:>4}" for j in range(J + 1))
            print(row)

        # j=0 column comparison
        print(f"\n{'i':>3}  {'R̂_i0':>6}  {'R_i0':>6}  {'diff':>6}")
        print("-" * 28)
        for i in range(n_occ):
            diff = R_hat[i, 0] - R[i, 0]
            print(f"{i:>3}  {R_hat[i, 0]:>6}  {R[i, 0]:>6}  {diff:>+6}")

    print("\nP:")
    print(P)