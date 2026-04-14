import numpy as np
from scipy.optimize import lsq_linear
from collections import defaultdict

# Parameters 
J = 3                                    # max development period
mu_j = np.array([500, 300, 150, 50])     # NegBin parameters
mu = mu_j.sum()                         
beta = 2 / 1000                          
alpha_j = mu_j * beta                    
alpha = alpha_j.sum()

T = 250                       # total simulation periods
n_occ = 17                    # occurrence periods analysed
occ_start = T - n_occ
t_eval = n_occ - 1                       

# Simulation functions =========================================================

# Draws R_{i,j} from a NegBin distribution for each occurrence period i and development period j.
def simulate_R(T=T):
    R_full = np.zeros((T, J + 1), dtype=int)
    for i in range(T):
        for j in range(J + 1):
            R_full[i, j] = np.random.negative_binomial(alpha_j[j], beta / (beta + 1))
    return R_full

# Gives label (occurrence i, development j, calendar period cal, uniform calender time in [cal-1, cal)) to each claim and sorts them by calendar time.
# Also returns R_t
def build_claims(R_full, T=T):
    claims = []
    for i in range(T):
        for j in range(J + 1):
            for _ in range(R_full[i, j]):
                cal = i + j
                claims.append((i, j, cal, (cal - 1) + np.random.uniform()))
    claims.sort(key=lambda x: x[3])

    claims_by_cal = defaultdict(list) # group claims by calendar period
    for idx, cl in enumerate(claims):
        claims_by_cal[cl[2]].append(idx)

    R_total = np.zeros(T + J + 1, dtype=int) # total number of reportings in each calendar period
    for t in range(T + J + 1):
        for j in range(J + 1):
            i = t - j
            if 0 <= i < T:
                R_total[t] += R_full[i, j]
    return claims, claims_by_cal, R_total

# Simulates the processing queue and returns P_{i,j}, B_t, P_t.
def simulate_queue(claims, claims_by_cal, eta, T=T):
    c = int(eta * mu) # capacity
    P_full = np.zeros((T, T + J + 1), dtype=int)
    B_total = np.zeros(T + J + 2, dtype=int) # backlog at the start of each period

    queue = [] # FCFS queue
    for t in range(T + J + 1):
        B_total[t] = len(queue)
        available = queue + claims_by_cal.get(t, [])
        for idx in available[:c]:
            P_full[claims[idx][0], t - claims[idx][0]] += 1
        queue = available[c:]
    B_total[T + J + 1] = len(queue)

    P_total = np.zeros(T + J + 1, dtype=int) #P_t
    for t in range(T + J + 1):
        for i in range(T):
            j = t - i
            if 0 <= j:
                P_total[t] += P_full[i, j]

    return P_full, B_total, P_total

# Checks last non-zero development period for observed processed claims, estimates J
def compute_J_P(P_full, n_occ):
    P = P_full[-n_occ:]
    best = 0
    for i in range(n_occ):
        max_col = n_occ - 1 - i  # last observed column for row i
        for j in range(max_col, best, -1):
            if P[i, j] > 0:
                best = j
                break
    return best

# Check smallest j s.t P_{i,j} > 0 for some i<n_occ-J_P
def compute_J_P_min(P_full, n_occ, J_P):
    P = P_full[-n_occ:]
    n_rows = n_occ - J_P
    last_nz = []
    for i in range(n_rows):
        max_col = n_occ - 1 - i
        col = 0
        for j in range(max_col, -1, -1):
            if P[i, j] > 0:
                col = j
                break
        last_nz.append(col)
    return min(last_nz)

# Gives a[s] and b[s]
def compute_ab(B_total, P_total, R_total, occ_start, n_occ):
    a = np.zeros(n_occ)
    b = np.zeros(n_occ)
    for s in range(n_occ):
        Bs, Ps, Rs = B_total[occ_start + s], P_total[occ_start + s], R_total[occ_start + s]
        a[s] = 1.0 if Bs <= Ps else Ps / Bs
        b[s] = (Ps - Bs) / Rs if (Bs <= Ps and Rs > 0) else 0.0
    return a, b


# Recovery function ===========================================================
""""
Recovers R_hat by minimizing eq. 16, assuming J_est is the true J.
J_hat controlls the information used"""
 
def recover_reportings(P_full, B_total, R_total, P_total, occ_start, n_occ, J_hat, J_est, gamma=0):
    a, b = compute_ab(B_total, P_total, R_total, occ_start, n_occ)

    def build_block_Mr(i): # build M_r^(i)
        M = np.zeros((J_hat + 1, J_hat + 1))
        for row in range(J_hat + 1):
            s = i + row
            if s >= n_occ:
                continue
            M[row, row] = b[s]
            for col in range(row):
                M[row, col] = a[s]
        return M

    def build_block_Mp(i):
        M = np.zeros((J_hat + 1, J_hat + 1))
        for row in range(J_hat + 1):
            s = i + row
            if s >= n_occ:
                continue
            M[row, row] = 1.0
            for col in range(row):
                M[row, col] = a[s]
        return M

    dim = (J_hat + 1) * n_occ
    Mr = np.zeros((dim, dim))
    Mp = np.zeros((dim, dim))
    for i in range(n_occ):
        sl = slice(i * (J_hat + 1), (i + 1) * (J_hat + 1))
        Mr[sl, sl] = build_block_Mr(i)
        Mp[sl, sl] = build_block_Mp(i)

    p = np.zeros(dim)
    for i in range(n_occ):
        for j in range(J_hat + 1):
            if i + j < n_occ:
                p[i * (J_hat + 1) + j] = P_full[occ_start + i, j]

    C = Mr
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

    lb = np.zeros(dim)
    ub = np.full(dim, np.inf)
    for i in range(n_occ):
        for j in range(J_hat + 1):
            if j > J_est or i + j >= n_occ:
                ub[i * (J_hat + 1) + j] = 0.4 
    result = lsq_linear(C, d, bounds=(lb, ub), method='bvls')

    if not np.isfinite(result.x).all():
        raise ValueError("lsq_linear failed: result contains absurd values")

    return np.round(result.x).reshape(n_occ, J_hat + 1).astype(int)


# Chain Ladder functions =======================================================

# Builds cummulative run off triangle
def build_triangle(data, n_rows, max_j, t_eval):
    cum = np.full((n_rows, max_j), np.nan)
    for row in range(n_rows):
        s = 0
        for j in range(max_j):
            if row + j > t_eval:
                break
            s += data[row, j]
            cum[row, j] = s
    return cum

# Ordinary CL, returns development factors and completed triangle
def chain_ladder(cum, n_rows, max_j):
    factors = np.ones(max_j - 1)
    for j in range(max_j - 1):
        num = den = 0
        for row in range(n_rows):
            if not np.isnan(cum[row, j]) and not np.isnan(cum[row, j + 1]):
                num += cum[row, j + 1]
                den += cum[row, j]
        factors[j] = num / den if den > 0 else 1.0

    completed = cum.copy()
    for row in range(n_rows):
        last = max(j for j in range(max_j) if not np.isnan(cum[row, j]))
        for j in range(last, max_j - 1):
            completed[row, j + 1] = np.round(completed[row, j] * factors[j]).astype(int)

    return factors, completed


# Benktander function =======================================================

# A priori expected ultimate must be provided. This function returns the completed triangle
def benktander(cum, factors, prior, a, n_rows, max_j, kappa = 1.0):
    ldf = np.ones(max_j)
    for j in range(max_j - 2, -1, -1):
        ldf[j] = ldf[j + 1] * factors[j]
    weights = 1.0 - 1.0 / ldf             # proportion left to be observed according to dev. factors

    completed = cum.copy()
    for row in range(n_rows):
        obs = [j for j in range(max_j) if not np.isnan(cum[row, j])]
        j_last = max(obs)
        if j_last == max_j - 1:
            continue
        q = weights[j_last]
        """This is a heuristic formula, which can be discussed"""
        delta = kappa / a.mean()
        z = (1.0 - q) ** delta
        u_cl = cum[row, j_last] * ldf[j_last]
        u_bf = cum[row, j_last] + q * prior[row]
        completed_val = z * u_cl + (1 - z) * u_bf
        for j in range(j_last + 1, max_j):
            completed[row, j] = np.round(completed_val / ldf[j]).astype(int)

    return completed
