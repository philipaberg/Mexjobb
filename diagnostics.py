import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from core import (
    J, n_occ, simulate_R, build_claims, 
    simulate_queue, recover_reportings_known_j,
    recover_reportings_est_j, compute_J_P, 
    compute_ab, build_triangle, chain_ladder,
)

# Parametrar
T_SIM = 250                # burn-in stationarity for B_t, when eta = 1.10
ETA = 1.1
GAMMA_KNOWN_J = 100        
GAMMA_EST_J = 0          
N_SIM = 50000

# Simulation structure
def run_sim(_):
    occ_start = T_SIM - n_occ
    t_eval = n_occ - 1

    R_full = simulate_R(T_SIM)
    claims, claims_by_cal, R_total = build_claims(R_full, T_SIM)
    P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, ETA, T_SIM)

    R = R_full[-n_occ:]
    true_ult = R[:, :J + 1].sum(axis=1)

    J_P = compute_J_P(P_full, n_occ)
    a, b = compute_ab(B_total, P_total, R_total, occ_start, n_occ)

    P = P_full[-n_occ:, :J_P + 1]
    _, CL_P = chain_ladder(build_triangle(P, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
    err_P = CL_P[:, -1] - true_ult

    R_hat_kj = recover_reportings_known_j(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, GAMMA_KNOWN_J)
    _, CL_Rhat_kj = chain_ladder(build_triangle(R_hat_kj, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
    err_Rhat_kj = CL_Rhat_kj[:, -1] - true_ult

    R_hat_ej = recover_reportings_est_j(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, GAMMA_EST_J)
    _, CL_Rhat_ej = chain_ladder(build_triangle(R_hat_ej, n_occ, J_P + 1, t_eval), n_occ, J_P + 1)
    err_Rhat_ej = CL_Rhat_ej[:, -1] - true_ult

    # Sum errors 
    sum_err_P = err_P.sum()
    sum_err_Rhat_kj = err_Rhat_kj.sum()
    sum_err_Rhat_ej = err_Rhat_ej.sum()

    return {
        "J_hat": J_P,
        "sum_a": a.sum(),
        "n_zeros_b": int(np.sum(b == 0)),
        "sum_b": float(b.sum()),
        "sum_err_P": sum_err_P,
        "sum_err_Rhat_kj": sum_err_Rhat_kj,
        "sum_err_Rhat_ej": sum_err_Rhat_ej,
        "worse_kj": abs(sum_err_P) < abs(sum_err_Rhat_kj),
        "worse_ej": abs(sum_err_P) < abs(sum_err_Rhat_ej),
    }


if __name__ == "__main__":

    # Step 1: Run simulations in parallel
    with Pool() as pool:
        results = list(tqdm(pool.imap(run_sim, range(N_SIM)), total=N_SIM))

    # Extract results
    J_hats = np.array([r["J_hat"] for r in results])
    sum_as = np.array([r["sum_a"] for r in results])
    n_zeros_bs = np.array([r["n_zeros_b"] for r in results])
    sum_bs = np.array([r["sum_b"] for r in results])
    sum_err_P = np.array([r["sum_err_P"] for r in results])
    sum_err_Rhat_kj = np.array([r["sum_err_Rhat_kj"] for r in results])
    sum_err_Rhat_ej = np.array([r["sum_err_Rhat_ej"] for r in results])
    worse_kj = np.array([r["worse_kj"] for r in results])
    worse_ej = np.array([r["worse_ej"] for r in results])

    # Step 2: Classify errors against CL(R) distribution
    err_CLR = np.load("errordistributionCLR.npy")      
    total_err_CLR = err_CLR.sum(axis=1)
    q_low, q_high = np.percentile(total_err_CLR, [2.5, 97.5])

    def classify_2x2(err_P, err_Rhat, worse, label):
        P_inside = (err_P >= q_low) & (err_P <= q_high)
        Rhat_inside = (err_Rhat >= q_low) & (err_Rhat <= q_high)

        both_inside = P_inside & Rhat_inside
        P_out_only = ~P_inside & Rhat_inside    # återställning räddar
        Rhat_out_only = P_inside & ~Rhat_inside    # återställning försämrar
        both_outside = ~P_inside & ~Rhat_inside

        def pct_worse(mask):
            n = mask.sum()
            if n == 0:
                return float("nan")
            return 100 * worse[mask].mean()
        
        print(f"\n{'='*67}")
        print(f"2x2 CLASSIFICATION — {label}")
        print(f"Reference interval CL(R): [{q_low:.1f}, {q_high:.1f}]")
        print(f"{'='*60}")
        print(f"  Both inside:              {both_inside.sum():>6}  ({100*both_inside.mean():.1f}%)  R_hat worse: {pct_worse(both_inside):.1f}%")
        print(f"  P outside only:           {P_out_only.sum():>6}  ({100*P_out_only.mean():.1f}%)  (recovery helps)")
        print(f"  R_hat outside only:       {Rhat_out_only.sum():>6}  ({100*Rhat_out_only.mean():.1f}%)  (recovery worsens)")
        print(f"  Both outside:             {both_outside.sum():>6}  ({100*both_outside.mean():.1f}%)  R_hat worse: {pct_worse(both_outside):.1f}%")

        return both_inside, P_out_only, Rhat_out_only, both_outside

    cells_kj = classify_2x2(sum_err_P, sum_err_Rhat_kj, worse_kj, f"Known J (gamma={GAMMA_KNOWN_J})")
    cells_ej = classify_2x2(sum_err_P, sum_err_Rhat_ej, worse_ej, f"Estimated J (gamma={GAMMA_EST_J})")

    # Save results
    
    np.save("when_to_use_results.npy", {
        "J_hats": J_hats,
        "sum_as": sum_as,
        "n_zeros_bs": n_zeros_bs,
        "sum_bs": sum_bs,
        "sum_err_P": sum_err_P,
        "sum_err_Rhat_kj": sum_err_Rhat_kj,
        "sum_err_Rhat_ej": sum_err_Rhat_ej,
        "worse_kj": worse_kj,
        "worse_ej": worse_ej,
        "q_low": q_low,
        "q_high": q_high,
        "cells_kj": cells_kj,
        "cells_ej": cells_ej,
        "ETA": ETA,
        "GAMMA_KNOWN_J": GAMMA_KNOWN_J,
        "GAMMA_EST_J": GAMMA_EST_J,
        "N_SIM": N_SIM,
        "T_SIM": T_SIM,
    })
    
    print(f"\nResultat sparat")
