import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from core import (
    J, T, n_occ, occ_start,
    simulate_R, build_claims, simulate_queue,
    recover_reportings, compute_J_P, compute_J_P_min, compute_ab,
)

GAMMA  = 10
ETA    = 1.1
N_SIM  = 2000

METHODS = ["J", "J_P", "J_P_min"]


def build_C_d(P_full, B_total, R_total, P_total, n_occ, occ_start, J_hat, J_est, gamma):
    a, b = compute_ab(B_total, P_total, R_total, occ_start, n_occ)
    dim = (J_hat + 1) * n_occ

    Mr = np.zeros((dim, dim))
    Mp = np.zeros((dim, dim))
    for i in range(n_occ):
        sl = slice(i * (J_hat + 1), (i + 1) * (J_hat + 1))
        Mr_block = np.zeros((J_hat + 1, J_hat + 1))
        Mp_block = np.zeros((J_hat + 1, J_hat + 1))
        for row in range(J_hat + 1):
            s = i + row
            if s >= n_occ:
                continue
            Mr_block[row, row] = b[s]
            Mp_block[row, row] = 1.0
            for col in range(row):
                Mr_block[row, col] = a[s]
                Mp_block[row, col] = a[s]
        Mr[sl, sl] = Mr_block
        Mp[sl, sl] = Mp_block

    p = np.zeros(dim)
    for i in range(n_occ):
        for j in range(J_hat + 1):
            if i + j < n_occ:
                p[i * (J_hat + 1) + j] = P_full[occ_start + i, j]

    C = Mr.copy()
    d = Mp @ p

    if gamma > 0:
        A = np.zeros((n_occ - J_est, dim))
        s_vec = np.zeros(n_occ - J_est)
        for k, s in enumerate(range(J_est, n_occ)):
            s_vec[k] = R_total[occ_start + s]
            for j in range(J_est + 1):
                i = s - j
                if 0 <= i < n_occ:
                    A[k, i * (J_hat + 1) + j] = 1.0
        C = np.vstack([C, np.sqrt(gamma) * A])
        d = np.concatenate([d, np.sqrt(gamma) * s_vec])

    return C, d, a, b


def residual(C, d, x):
    r = C @ x - d
    return r @ r


def run_sim(_):
    try:
        R_full = simulate_R()
        claims, claims_by_cal, R_total = build_claims(R_full)
        P_full, B_total, P_total = simulate_queue(claims, claims_by_cal, ETA)

        J_P = compute_J_P(P_full, n_occ)
        J_P_min = compute_J_P_min(P_full, n_occ, J_P)

        method_J_ests = {"J": J, "J_P": J_P, "J_P_min": J_P_min}

        results = {}
        for name, J_est in method_J_ests.items():
            R_hat = recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_P, J_est, gamma=GAMMA,)

            C, d, a, b = build_C_d(P_full, B_total, R_total, P_total, n_occ, occ_start, J_P, J_est, gamma=GAMMA,)

            x_hat = R_hat.flatten().astype(float)

            x_true_block = np.zeros((n_occ, J_P + 1))
            cols = min(J + 1, J_P + 1)
            x_true_block[:, :cols] = R_full[-n_occ:, :cols]
            x_true = x_true_block.flatten().astype(float)

            res_hat  = residual(C, d, x_hat)
            res_true = residual(C, d, x_true)
            succeeded = res_hat <= res_true

            rank_C = np.linalg.matrix_rank(C)

            results[name] = {
                "succeeded":  succeeded,
                "res_hat":    res_hat,
                "res_true":   res_true,
                "rank":       rank_C,
                "n_cols":     C.shape[1],
                "a":          a,
                "b":          b,
            }

        return results

    except ValueError:
        return None


if __name__ == "__main__":
    with Pool() as pool:
        all_results = list(tqdm(pool.imap(run_sim, range(N_SIM)), total=N_SIM))

    skipped = sum(1 for r in all_results if r is None)
    completed = [r for r in all_results if r is not None]
    n_done = len(completed)

    print(f"\n=== Results over {N_SIM} simulations (gamma={GAMMA}, eta={ETA}) ===")
    print(f"  Skipped (solver failure) : {skipped}/{N_SIM}\n")

    for name in METHODS:
        res_list        = [r[name] for r in completed]
        success         = sum(r["succeeded"] for r in res_list)
        ranks           = [r["rank"] for r in res_list]
        full_rank_count = sum(r["rank"] == r["n_cols"] for r in res_list)
        failures        = [r for r in res_list if not r["succeeded"]]

        print(f"--- Method: {name} ---")
        print(f"  Minimization succeeded   : {success}/{n_done}  ({100*success/n_done:.1f}%)")
        print(f"  Full column rank         : {full_rank_count}/{n_done}  ({100*full_rank_count/n_done:.1f}%)")
        print(f"  Rank  min/mean/max       : {min(ranks)} / {np.mean(ranks):.1f} / {max(ranks)}")

        if failures:
            print(f"  Failures ({len(failures)}):")
            for idx, f in enumerate(failures):
                print(f"    [{idx}] res_hat={f['res_hat']:.4f}  res_true={f['res_true']:.4f}")
                print(f"           a = {np.round(f['a'], 3)}")
                print(f"           b = {np.round(f['b'], 3)}")
        print()
