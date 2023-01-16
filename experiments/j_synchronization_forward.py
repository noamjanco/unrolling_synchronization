from common.math_utils import rel_error_so3
from data_generation.generate_data_so3 import generate_data_so3, apply_j
from synchronization.pim import pim_so3
import numpy as np
import tqdm
import time
import matplotlib.pyplot as plt
import tensorflow as tf


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


def j_synch_forward(H: np.ndarray):
    batchsize = H.shape[0]
    N = int(H.shape[1]/3)
    shape = (int((N - 1) * N / 2), int((N - 1) * N / 2))

    J = tf.expand_dims(tf.linalg.diag((1., 1., -1.)), axis=0)
    J_block = tf.expand_dims(tf.linalg.diag(tf.tile((1., 1., -1.), [N])), axis=0)

    H_j_conj = J_block @ H @ J_block



    Hij = tf.reshape(tf.gather_nd(H, ijs), (batchsize, -1, 3, 3))
    Hij_j_conj = tf.reshape(tf.gather_nd(H_j_conj, ijs), (batchsize, -1, 3, 3))
    Hjk = tf.reshape(tf.gather_nd(H, jks), (batchsize, -1, 3, 3))
    Hjk_j_conj = tf.reshape(tf.gather_nd(H_j_conj, jks), (batchsize, -1, 3, 3))
    Hki = tf.reshape(tf.gather_nd(H, kis), (batchsize, -1, 3, 3))
    Hki_j_conj = tf.reshape(tf.gather_nd(H_j_conj, kis), (batchsize, -1, 3, 3))

    H_ij = tf.reshape(Hij, (-1, 3, 3))
    H_ij_j_conj = tf.reshape(Hij_j_conj, (-1, 3, 3))
    H_jk = tf.reshape(Hjk, (-1, 3, 3))
    H_jk_j_conj = tf.reshape(Hjk_j_conj, (-1, 3, 3))
    H_ki = tf.reshape(Hki, (-1, 3, 3))
    H_ki_j_conj = tf.reshape(Hki_j_conj, (-1, 3, 3))

    err0 = tf.linalg.norm(H_ij @ H_jk @ H_ki - tf.eye(3), axis=[1, 2])
    err1 = tf.linalg.norm(H_ij @ H_jk @ H_ki_j_conj - tf.eye(3), axis=[1, 2])
    err2 = tf.linalg.norm(H_ij @ H_jk_j_conj @ H_ki - tf.eye(3), axis=[1, 2])
    err3 = tf.linalg.norm(H_ij @ H_jk_j_conj @ H_ki_j_conj - tf.eye(3), axis=[1, 2])
    err4 = tf.linalg.norm(H_ij_j_conj @ H_jk @ H_ki - tf.eye(3), axis=[1, 2])
    err5 = tf.linalg.norm(H_ij_j_conj @ H_jk @ H_ki_j_conj - tf.eye(3), axis=[1, 2])
    err6 = tf.linalg.norm(H_ij_j_conj @ H_jk_j_conj @ H_ki - tf.eye(3), axis=[1, 2])
    err7 = tf.linalg.norm(H_ij_j_conj @ H_jk_j_conj @ H_ki_j_conj - tf.eye(3), axis=[1, 2])

    err = tf.stack([err0, err1, err2, err3, err4, err5, err6, err7], axis=-1)

    err_rs = tf.reshape(err, (batchsize, -1, 8))

    min_ind = tf.argmin(err_rs, axis=-1)
    mu_ki_opt = tf.math.floormod(min_ind / 1, 2)
    mu_jk_opt = tf.math.floormod(tf.floor(min_ind / 2), 2)
    mu_ij_opt = tf.math.floormod(tf.floor(min_ind / 4), 2)


    res1 = tf.map_fn(lambda x: tf.scatter_nd(indices=_ij_jk, updates=x, shape=shape),
                     tf.pow(-1., mu_ij_opt - mu_jk_opt))
    res2 = tf.map_fn(lambda x: tf.scatter_nd(indices=_jk_ki, updates=x, shape=shape),
                     tf.pow(-1., mu_jk_opt - mu_ki_opt))
    res3 = tf.map_fn(lambda x: tf.scatter_nd(indices=_ki_ij, updates=x, shape=shape),
                     tf.pow(-1., mu_ki_opt - mu_ij_opt))

    Sigma = res1 + res2 + res3
    Sigma = Sigma + tf.transpose(Sigma,perm=[0,2,1])

    #todo: power iteration over batch
    u_s = power_iteration(Sigma[0])

    # apply J()J to each relative measurement with u < 0
    for i in range(N):
        for j in range(i + 1, N):
            if u_s[pair_dict[(i, j)]] < 0:
                H[0,3 * i:3 * i + 3, 3 * j:3 * j + 3] = J @ H[0,3 * i:3 * i + 3, 3 * j:3 * j + 3] @ J
                H[0,3 * j:3 * j + 3, 3 * i:3 * i + 3] = J @ H[0,3 * j:3 * j + 3, 3 * i:3 * i + 3] @ J
    return H


np.random.seed(1)
# Lambda = 3
# N = 50
# Lambda_range = np.arange(1,10,1)
N = 20
R = 1
# Lambda_range = np.arange(1,9,2)
Lambda_range = np.arange(1,9,2)
# Lambda_range = [3.]
# Lambda_range = [9.]
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

# ---------------------------------------------------------------------------- #
# Fixed computation for N and batchsize
batchsize = 1
pair_dict = calc_pair_dict(N)
_ijs = []
_jks = []
_kis = []
indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _ijs.append(pair_dict[(i, j)])
            _jks.append(pair_dict[(j, k)])
            _kis.append(pair_dict[(k, i)])
            indices.append((i, j, k))

_ij_jk = []
_jk_ki = []
_ki_ij = []
for i in range(len(_ijs)):
    _ij_jk.append((_ijs[i], _jks[i]))
    _jk_ki.append((_jks[i], _kis[i]))
    _ki_ij.append((_kis[i], _ijs[i]))

ijs = []
jks = []
kis = []
for b in range(batchsize):
    for i, idx in enumerate(indices):
        for x in range(3):
            for y in range(3):
                ijs.append([b, 3 * idx[0] + x, 3 * idx[1] + y])
                jks.append([b, 3 * idx[1] + x, 3 * idx[2] + y])
                kis.append([b, 3 * idx[2] + x, 3 * idx[0] + y])


ijs = tf.constant(np.asarray(ijs))
jks = tf.constant(np.asarray(jks))
kis = tf.constant(np.asarray(kis))
# ---------------------------------------------------------------------------- #


for Lambda in tqdm.tqdm(Lambda_range):
    err_no_j_acc = []
    err_with_j_acc = []
    err_with_j_synch_acc = []
    for r in range(R):
        # generate samples according to R_ij = {R_i^T R_j, J R_i^TR_jJ}, where J=diag(1,1,-1)
        Rot, H = generate_data_so3(N, Lambda)
        # H = H * N / Lambda
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
        # H = j_synchronization(H, J_conj_dict, N, Lambda)
        H = j_synch_forward(np.asarray(np.expand_dims(H,axis=0), np.float32))
        H = H[0]
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
