import numpy as np
from tqdm import tqdm
from core import (J, n_occ, t_eval, alpha_j, beta, build_triangle, chain_ladder,)

N_SIM = 100_000

errors = []

for _ in tqdm(range(N_SIM)):
    R = np.random.negative_binomial(alpha_j, beta / (beta + 1), size=(n_occ, J + 1))
    true_ult = R.sum(axis=1)
    _, CL_R = chain_ladder(build_triangle(R, n_occ, J + 1, t_eval), n_occ, J + 1)
    errors.append(CL_R[:, -1] - true_ult)

np.save("errordistributionCLR.npy", np.array(errors))
print("Done — sparad som errordistributionCLR.npy")
