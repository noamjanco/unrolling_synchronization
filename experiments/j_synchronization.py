from common.math_utils import rel_error_so3
from data_generation.generate_data_so3 import generate_data_so3, apply_j
from synchronization.pim import pim_so3
import numpy as np
import tqdm
import time
import matplotlib.pyplot as plt

def plot_with_confidence(x,y,std, label, fig, ax):
    x = np.asarray(x)
    y = np.asarray(y)
    base_line, = ax.plot(x, y, label=label)
    ax.fill_between(x, (y - std), (y + std), color=base_line.get_color(), alpha=.1)

def j_conj_err(Rot_est: np.ndarray, Rot: np.ndarray):
    J = np.diag((1,1,-1))
    err_reg = rel_error_so3(Rot_est, Rot)
    N = int(Rot_est.shape[0]/3)
    Rot_est_j_conj = np.zeros_like(Rot_est, dtype=np.float64)
    for i in range(N):
        Rot_est_j_conj[3*i:3*i+3,:] = J @ Rot_est[3*i:3*i+3,:] @ J
    err_j_conj = rel_error_so3(Rot_est_j_conj, Rot)
    return min(err_reg, err_j_conj)

def power_iteration(Sigma: np.ndarray, max_iterations=100, tol=1e-5) -> np.ndarray:
    v = 1e-3 * np.ones((Sigma.shape[0],1))
    i = 0
    while i < max_iterations:
        v_prev = v
        v = Sigma @ v_prev
        v = v / np.sqrt(np.sum(v ** 2))
        if np.sum((v - v_prev) ** 2) < tol:
            break
        i += 1
    return v

def calc_pair_dict(N):
    d = {}
    idx = 0
    for i in range(N):
        for j in range(i+1,N):
            d[(i,j)] = idx
            d[(j,i)] = idx
            idx +=1
    return d

def j_synchronization(H: np.ndarray, J_conj_dict: dict) -> np.ndarray:
    def arg_min_c(H, i, j, k, J):
        H_ij = H[3 * i:3 * i + 3, 3 * j:3 * j + 3]
        H_jk = H[3 * j:3 * j + 3, 3 * k:3 * k + 3]
        H_ki = H[3 * k:3 * k + 3, 3 * i:3 * i + 3]
        H_ij_j_conj = J @ H_ij @ J
        H_jk_j_conj = J @ H_jk @ J
        H_ki_j_conj = J @ H_ki @ J
        mul = np.zeros((8, 3, 3))
        mul[0] = H_ij @ H_jk @ H_ki
        mul[1] = H_ij @ H_jk @ H_ki_j_conj
        mul[2] = H_ij @ H_jk_j_conj @ H_ki
        mul[3] = H_ij @ H_jk_j_conj @ H_ki_j_conj
        mul[4] = H_ij_j_conj @ H_jk @ H_ki
        mul[5] = H_ij_j_conj @ H_jk @ H_ki_j_conj
        mul[6] = H_ij_j_conj @ H_jk_j_conj @ H_ki
        mul[7] = H_ij_j_conj @ H_jk_j_conj @ H_ki_j_conj
        min_idx = np.argmin(np.linalg.norm(mul - np.eye(3),axis=(1,2)))
        mu_ki_opt = min_idx % 2
        mu_jk_opt = (min_idx >> 1) % 2
        mu_ij_opt = (min_idx >> 2) % 2

        # C_min = 1e9
        # mu_ij_opt, mu_jk_opt, mu_ki_opt = 0, 0, 0
        # for mu_ij in [0,1]:
        #     for mu_jk in [0,1]:
        #         for mu_ki in [0,1]:
        #             C = np.linalg.norm((np.linalg.matrix_power(J, mu_ij) @ H_ij @ np.linalg.matrix_power(J, mu_ij)) @
        #                                (np.linalg.matrix_power(J, mu_jk) @ H_jk @ np.linalg.matrix_power(J, mu_jk)) @
        #                                (np.linalg.matrix_power(J, mu_ki) @ H_ki @ np.linalg.matrix_power(J, mu_ki)) - np.eye(3))
        #             # print(C)
        #             if C < C_min:
        #                 C_min = C
        #                 mu_ij_opt, mu_jk_opt, mu_ki_opt = mu_ij, mu_jk, mu_ki
        # print('-' * 10)
        return mu_ij_opt, mu_jk_opt, mu_ki_opt

    J = np.diag((1,1,-1))
    N = int(H.shape[0] / 3)
    Sigma = np.zeros((int((N-1)*N / 2), int((N-1)*N / 2)))
    pair_dict = calc_pair_dict(N)

    # start = time.time()
    # for i in tqdm.tqdm(range(N)):
    for i in range(N):
        for j in range(i+1,N):
            for k in range(j+1, N):
                mu_ij, mu_jk, mu_ki = arg_min_c(H, i, j, k, J)

                Sigma[pair_dict[(i,j)],pair_dict[(j,k)]] = (-1.) ** (mu_ij - mu_jk)
                Sigma[pair_dict[(j,k)],pair_dict[(k,i)]] = (-1.) ** (mu_jk - mu_ki)
                Sigma[pair_dict[(k,i)],pair_dict[(i,j)]] = (-1.) ** (mu_ki - mu_ij)
    # dt = time.time() - start
    # print('took ', dt, ' seconds')

    Sigma = Sigma + Sigma.T

    # extract the eigenvector that corresponds to the leading eigenvalue of Sigma
    u_s = power_iteration(Sigma)
    # u_s = power_iteration(Sigma, max_iterations=100, tol=1e-7)

    # import matplotlib.pyplot as plt
    vals = []
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in J_conj_dict:
                vals.append(J_conj_dict[(i, j)])
            else:
                vals.append(0)
    vals = np.expand_dims(np.asarray(vals),axis=-1)
    err = np.min([np.mean(np.sign(vals-0.5) != np.sign(u_s)),np.mean(np.sign(vals-0.5) != -np.sign(u_s))])
    # print('j synch err = ', err)
    # plt.plot(vals, label='true')
    # plt.plot(u_s, label='estimated')
    # plt.show()

    # apply J()J to each relative measurement with u < 0
    for i in range(N):
        for j in range(i+1,N):
            if u_s[pair_dict[(i,j)]] < 0:
                H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = J @ H[3 * i:3 * i + 3, 3 * j:3 * j + 3] @ J
                H[3 * j:3 * j + 3, 3 * i:3 * i + 3] = J @ H[3 * j:3 * j + 3, 3 * i:3 * i + 3] @ J
    return H

np.random.seed(1)

# Lambda = 3
# N = 50
# Lambda_range = np.arange(1,10,1)
N = 20
R = 20
Lambda_range = np.arange(1,9,.25)
# Lambda_range = [10]
err_no_j_acc = []
err_with_j_acc = []
err_with_j_synch_acc = []
err_no_j_vec = []
err_no_j_vec_std = []
err_with_j_vec =[]
err_with_j_vec_std =[]
err_with_j_synch_vec = []
err_with_j_synch_vec_std = []
for Lambda in Lambda_range:
    err_no_j_acc = []
    err_with_j_acc = []
    err_with_j_synch_acc = []
    for r in range(R):
        # generate samples according to R_ij = {R_i^T R_j, J R_i^TR_jJ}, where J=diag(1,1,-1)
        Rot, H = generate_data_so3(N, Lambda)
        H = H * N / Lambda
        # err with no J conj
        Rot_est = pim_so3(H)
        Rot_est = np.real(Rot_est)
        err_no_j = j_conj_err(Rot_est, Rot)
        err_no_j_acc.append(err_no_j)
        H, J_conj_dict = apply_j(H, p=0.5)

        # err with J conj without J-synch
        Rot_est = pim_so3(H)
        Rot_est = np.real(Rot_est)
        err_with_j = j_conj_err(Rot_est, Rot)
        err_with_j_acc.append(err_with_j)

        # error with J conj and J-synch
        H = j_synchronization(H, J_conj_dict)
        Rot_est = pim_so3(H)
        Rot_est = np.real(Rot_est)
        err_with_j_synch = j_conj_err(Rot_est, Rot)
        err_with_j_synch_acc.append(err_with_j_synch)

    err_no_j_vec.append(np.mean(err_no_j_acc))
    err_no_j_vec_std.append(np.std(err_no_j_acc))
    err_with_j_vec.append(np.mean(err_with_j_acc))
    err_with_j_vec_std.append(np.std(err_with_j_acc))
    err_with_j_synch_vec.append(np.mean(err_with_j_synch_acc))
    err_with_j_synch_vec_std.append(np.std(err_with_j_synch_acc))

fig, ax = plt.subplots()
plot_with_confidence(Lambda_range, err_no_j_vec,err_no_j_vec_std, 'err without J ambiguity', fig, ax)
plot_with_confidence(Lambda_range, err_with_j_vec, err_with_j_vec_std, 'err with J ambiguity', fig, ax)
plot_with_confidence(Lambda_range, err_with_j_synch_vec, err_with_j_synch_vec_std, 'err with J ambiguity using J-Synch', fig, ax)
plt.xlabel('$SNR$')
plt.ylabel('Alignment Error')
plt.title(f'Error comparison with N={N}')
plt.legend()
plt.show()
# print('err = ', err_pim)
